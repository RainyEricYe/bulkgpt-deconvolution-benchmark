"""Resource tracking for test runs: CPU, GPU, memory, time."""

import json
import os
import platform
import subprocess
import threading
import time as _time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class GPUMonitor:
    """Real-time GPU memory peak tracker.

    Polls ``nvidia-smi --query-gpu=index,memory.used`` every 2 seconds during
    method execution to capture the peak per-GPU memory usage. Computes the
    delta (peak minus baseline) so the reported value approximates what the
    method actually used.

    Usage::

        monitor = GPUMonitor()
        monitor.start()
        ... run method ...
        monitor.stop()
        gpu_info = monitor.get_delta()
    """

    _stop: threading.Event = field(default_factory=threading.Event, repr=False)
    _thread: threading.Thread | None = field(default=None, repr=False)
    _peak: dict = field(default_factory=dict, repr=False)
    _baseline: dict = field(default_factory=dict, repr=False)

    def _snapshot(self) -> dict:
        """Query current GPU memory per device, return {idx: {name, memory_used_gb}}."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,name,memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                return {}
            snap = {}
            for line in result.stdout.strip().split("\n"):
                line = line.strip()
                if not line:
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    snap[parts[0]] = {
                        "name": parts[1],
                        "memory_used_gb": round(float(parts[2]) / 1024, 2),
                    }
            return snap
        except Exception:
            return {}

    def start(self):
        """Record baseline and start background polling thread."""
        self._stop.clear()
        self._baseline = self._snapshot()
        self._peak = dict(self._baseline)

        def _poll():
            while not self._stop.wait(2):
                snap = self._snapshot()
                for idx, dev in snap.items():
                    cur = dev["memory_used_gb"]
                    if idx not in self._peak or cur > self._peak[idx]["memory_used_gb"]:
                        self._peak[idx] = dev

        self._thread = threading.Thread(target=_poll, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop polling and record final snapshot."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        # Final snapshot ensures peak catches any late spike
        final = self._snapshot()
        for idx, dev in final.items():
            cur = dev["memory_used_gb"]
            if idx not in self._peak or cur > self._peak[idx]["memory_used_gb"]:
                self._peak[idx] = dev

    def get_peak(self) -> dict:
        """Return peak per-device GPU memory during monitoring window."""
        devices = [{"index": int(k), **v} for k, v in sorted(self._peak.items(), key=lambda x: int(x[0]))]
        return {"available": bool(devices), "count": len(devices), "devices": devices}

    def get_delta(self) -> dict:
        """Return peak minus baseline per-device GPU memory (method's contribution)."""
        devices = []
        for idx, dev in sorted(self._peak.items(), key=lambda x: int(x[0])):
            baseline = self._baseline.get(idx, {}).get("memory_used_gb", 0)
            delta = round(max(0, dev["memory_used_gb"] - baseline), 2)
            devices.append({"index": int(idx), "name": dev["name"], "memory_used_gb": delta})
        return {"available": bool(devices), "count": len(devices), "devices": devices}


def _get_cpu_info(pid: int | None = None) -> dict:
    """CPU info: model, core count, and process/child CPU time + RSS.

    When *pid* is given, tracks that specific process (the method/apptainer).
    When *pid* is None, tracks current process via getrusage.
    """
    import psutil

    info = {"count": psutil.cpu_count(logical=True)}
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    info["model"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        info["model"] = platform.processor()

    if pid is not None:
        try:
            proc = psutil.Process(pid)
            with proc.oneshot():
                cpu_t = proc.cpu_times()
                mem = proc.memory_info()
                info["user_time_s"] = round(cpu_t.user, 2)
                info["system_time_s"] = round(cpu_t.system, 2)
                info["max_rss_kb"] = mem.rss // 1024 if mem.rss else 0
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            info["user_time_s"] = 0
            info["system_time_s"] = 0
            info["max_rss_kb"] = 0
    else:
        import resource

        ru = resource.getrusage(resource.RUSAGE_SELF)
        info["user_time_s"] = round(ru.ru_utime, 2)
        info["system_time_s"] = round(ru.ru_stime, 2)
        info["max_rss_kb"] = ru.ru_maxrss
    return info


def _get_children_cpu(pid: int) -> dict:
    """Read children CPU time from /proc/<pid>/stat (cutime, cstime)."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            parts = f.read().split()
        clk_tck = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
        cutime = int(parts[15]) / clk_tck
        cstime = int(parts[16]) / clk_tck
        return {
            "user_time_s": round(cutime, 2),
            "system_time_s": round(cstime, 2),
            "max_rss_kb": 0,
        }
    except (OSError, IndexError, ValueError, AttributeError):
        return {"user_time_s": 0, "system_time_s": 0, "max_rss_kb": 0}


def _get_cpu_for_children() -> dict:
    """Read children CPU time from /proc/self/stat (cutime, cstime).

    The shell that ran the method has waited-for children whose CPU time
    accumulates in the c* fields of /proc/self/stat. Also captures max RSS
    from getrusage(RUSAGE_CHILDREN) (works when the waited child is an
    apptainer subprocess or a direct Python subprocess).

    This function is typically called from the main process (e.g., the Python
    script that ran the method via subprocess.run) where getrusage captures
    the waited-for child's peak RSS.
    """
    result = {"user_time_s": 0, "system_time_s": 0, "max_rss_kb": 0}
    try:
        with open("/proc/self/stat") as f:
            parts = f.read().split()
        clk_tck = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
        result["user_time_s"] = round(int(parts[15]) / clk_tck, 2)
        result["system_time_s"] = round(int(parts[16]) / clk_tck, 2)
    except (OSError, IndexError, ValueError, AttributeError):
        pass
    try:
        import resource
        ru = resource.getrusage(resource.RUSAGE_CHILDREN)
        if ru.ru_maxrss > 0:
            result["max_rss_kb"] = ru.ru_maxrss
    except ImportError:
        pass
    return result


def _get_memory_info() -> dict:
    """System and process memory info."""
    import psutil

    vm = psutil.virtual_memory()
    return {
        "total_gb": round(vm.total / 1e9, 2),
        "available_gb": round(vm.available / 1e9, 2),
    }


def _get_gpu_for_pid(pid: int) -> dict:
    """Query per-process GPU memory for a specific PID."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory,gpu_name,gpu_bus_id",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            devices = []
            seen_bus = set()
            pid_str = str(pid)
            for line in result.stdout.strip().split("\n"):
                if not line.strip():
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 4 and parts[0] == pid_str:
                    bus = parts[3]
                    if bus not in seen_bus:
                        seen_bus.add(bus)
                        devices.append({
                            "name": parts[2],
                            "memory_used_gb": round(float(parts[1]) / 1024, 2),
                            "bus_id": bus,
                        })
            return {"available": bool(devices), "count": len(devices), "devices": devices}
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, IndexError):
        pass
    return {"available": False, "count": 0, "devices": []}


def _get_gpu_system() -> dict:
    """System-wide GPU info from nvidia-smi."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            devices = []
            for line in result.stdout.strip().split("\n"):
                if not line.strip():
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 4:
                    devices.append({
                        "index": int(parts[0]),
                        "name": parts[1],
                        "memory_total_gb": round(float(parts[2]) / 1024, 2),
                        "memory_used_gb": round(float(parts[3]) / 1024, 2),
                    })
            return {"available": True, "count": len(devices), "devices": devices}
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return {"available": False, "count": 0, "devices": []}


def collect_resources(pid: int | None = None) -> dict:
    """Resource snapshot. If *pid* is given, tracks that process."""
    info = {
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": _get_cpu_info(pid),
        "memory": _get_memory_info(),
        "gpu": _get_gpu_for_pid(pid) if pid is not None else _get_gpu_system(),
    }
    try:
        import torch
        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["cuda"] = torch.version.cuda
    except ImportError:
        pass
    return info


def collect_resources_for_method(shell_pid: int | None = None) -> dict:
    """Collect resources by reading the shell's children CPU stats.

    Used by run_real_bulk.py after a method finishes. The shell's /proc/<pid>/stat
    has accumulated cutime/cstime from the method process tree.
    If *shell_pid* is None, reads /proc/self/stat (caller must be the shell).
    """
    cpu_info = _get_cpu_for_children() if shell_pid is None else _get_children_cpu(shell_pid)
    info = {
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": cpu_info,
        "memory": _get_memory_info(),
        "gpu": _get_gpu_system(),
    }
    try:
        import torch
        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["cuda"] = torch.version.cuda
    except ImportError:
        pass
    return info


def write_resources_json(output_dir: str | Path, pid: int | None = None) -> None:
    """Snapshot resources and write to ``output_dir/resources.json``."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    info = collect_resources(pid)
    with open(out / "resources.json", "w") as f:
        json.dump(info, f, indent=2)
