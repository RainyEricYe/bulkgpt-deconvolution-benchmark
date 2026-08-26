from __future__ import annotations
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import trange

from core.deconv.config import TrainingConfig
from core.deconv.domain_adaptation import DomainAdaptationModule
from core.deconv.model import DeconvLoss, DeconvHead


class Trainer:
    """Generic trainer for deconvolution models.

    Works with any model that has a ``forward(gene_ids, values, mask) -> dict``
    interface returning ``{"proportions": ...}``.
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        device: str = "cuda",
        da_module: DomainAdaptationModule | None = None,
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.da_module = da_module

        self.criterion = DeconvLoss(loss_type=config.loss_type)

        # Support differential learning rates: backbone low, head high
        if config.backbone_lr is None:
            trainable_params = filter(lambda p: p.requires_grad, model.parameters())
            param_groups = [{"params": trainable_params, "lr": config.lr}]
        else:
            backbone_params = []
            head_params = []
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                if name.startswith("backbone."):
                    backbone_params.append(param)
                else:
                    head_params.append(param)
            param_groups = [
                {"params": backbone_params, "lr": config.backbone_lr},
                {"params": head_params, "lr": config.lr},
            ]
            print(f"Differential LR: backbone={config.backbone_lr}, head={config.lr}")
            print(f"  Backbone params: {sum(p.numel() for p in backbone_params):,}")
            print(f"  Head params: {sum(p.numel() for p in head_params):,}")

        self.optimizer = torch.optim.AdamW(
            param_groups,
            lr=config.lr,
            weight_decay=1e-5,
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=10,
            gamma=0.9,
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.device == "cuda"))

        self.best_val_loss = float("inf")
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self, loader: DataLoader) -> dict:
        self.model.train()
        total_loss = 0.0
        total_mse = 0.0
        total_kl = 0.0
        total_cos = 0.0
        n_batches = len(loader)

        for batch in loader:
            gene_ids = batch["gene_ids"].to(self.device)
            values = batch["values"].to(self.device)
            mask = batch["src_key_padding_mask"].to(self.device)
            true_props = batch["proportions"].to(self.device)

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=(self.device == "cuda")):
                output = self.model(gene_ids, values, mask)
                losses = self.criterion(output["proportions"], true_props)

            self.scaler.scale(losses["loss"]).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, self.model.parameters()), 1.0
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += losses["loss"].item()
            total_mse += losses["mse"].item()
            total_kl += losses["kl"].item()
            total_cos += losses["cos"].item()

        return {
            "loss": total_loss / n_batches,
            "mse": total_mse / n_batches,
            "kl": total_kl / n_batches,
            "cos": total_cos / n_batches,
            "lr": self.scheduler.get_last_lr()[0],
        }

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict:
        self.model.eval()
        total_loss = 0.0
        total_mse = 0.0
        total_kl = 0.0
        total_cos = 0.0
        n_batches = len(loader)

        for batch in loader:
            gene_ids = batch["gene_ids"].to(self.device)
            values = batch["values"].to(self.device)
            mask = batch["src_key_padding_mask"].to(self.device)
            true_props = batch["proportions"].to(self.device)

            with torch.cuda.amp.autocast(enabled=(self.device == "cuda")):
                output = self.model(gene_ids, values, mask)
                losses = self.criterion(output["proportions"], true_props)

            total_loss += losses["loss"].item()
            total_mse += losses["mse"].item()
            total_kl += losses["kl"].item()
            total_cos += losses["cos"].item()

        return {
            "loss": total_loss / n_batches,
            "mse": total_mse / n_batches,
            "kl": total_kl / n_batches,
            "cos": total_cos / n_batches,
        }

    def fit(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader | None = None,
        target_loader: DataLoader | None = None,
    ):
        if valid_loader is not None:
            val_metrics = self.evaluate(valid_loader)
            print(f"Initial val loss: {val_metrics['loss']:.4f}")

        train_metrics = {"loss": 0.0, "mse": 0.0, "kl": 0.0, "cos": 0.0, "lr": 0.0}

        use_da = target_loader is not None and self.da_module is not None

        for epoch in trange(1, self.config.epochs + 1, desc="Training"):
            if use_da:
                train_metrics = self.train_epoch_da(train_loader, target_loader, epoch)
            else:
                train_metrics = self.train_epoch(train_loader)

            if valid_loader is not None:
                val_metrics = self.evaluate(valid_loader)
                if val_metrics["loss"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["loss"]
                    self._save_checkpoint("best_model.pt", epoch, val_metrics)

            self.scheduler.step()

            if epoch % 10 == 0:
                self._log(epoch, train_metrics, val_metrics if valid_loader else None)

            if epoch % 10 == 0:
                self._save_checkpoint(f"model_epoch_{epoch}.pt", epoch, train_metrics)

        self._save_checkpoint("final_model.pt", self.config.epochs, train_metrics)

    def train_epoch_da(
        self,
        loader: DataLoader,
        target_loader: DataLoader,
        epoch: int,
    ) -> dict:
        """Training epoch with unsupervised domain adaptation.

        Each step processes one source batch and one target batch.
        Source: standard deconvolution loss.
        Target: based on *da_method* (GRL, MMD, or entropy), an
        unsupervised loss aligns the target-domain cell embeddings or
        predictions.
        """
        self.model.train()
        total_loss = 0.0
        total_mse = 0.0
        total_kl = 0.0
        total_cos = 0.0
        total_da = 0.0
        n_batches = len(loader)

        target_iter = iter(target_loader)
        da_lambda = self._get_da_lambda(
            epoch, self.config.epochs, self.config.da_grl_lambda
        )

        for batch in loader:
            gene_ids = batch["gene_ids"].to(self.device)
            values = batch["values"].to(self.device)
            mask = batch["src_key_padding_mask"].to(self.device)
            true_props = batch["proportions"].to(self.device)

            # ── Target batch ─────────────────────────────────────────
            try:
                target_batch = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                target_batch = next(target_iter)

            t_gene_ids = target_batch["gene_ids"].to(self.device)
            t_values = target_batch["values"].to(self.device)
            t_mask = target_batch["src_key_padding_mask"].to(self.device)

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=(self.device == "cuda")):
                # Source forward
                output = self.model(gene_ids, values, mask)
                losses = self.criterion(output["proportions"], true_props)

                # Target forward (no ground truth for target)
                target_output = self.model(t_gene_ids, t_values, t_mask)

                # Domain adaptation loss
                method = self.config.da_method
                if method == "grl":
                    da_loss = self.da_module.compute_grl_loss(
                        output["cell_emb"],
                        target_output["cell_emb"],
                        lambda_=da_lambda,
                    )
                elif method == "mmd":
                    da_loss = self.da_module.compute_mmd_loss(
                        output["cell_emb"],
                        target_output["cell_emb"],
                    )
                elif method == "entropy":
                    da_loss = self.da_module.compute_entropy_loss(
                        target_output["proportions"],
                    )
                else:
                    da_loss = torch.tensor(0.0, device=self.device)

                total_loss_val = losses["loss"] + self.config.da_lambda * da_loss

            self.scaler.scale(total_loss_val).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, self.model.parameters()), 1.0
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += total_loss_val.item()
            total_mse += losses["mse"].item()
            total_kl += losses["kl"].item()
            total_cos += losses["cos"].item()
            total_da += da_loss.item()

        return {
            "loss": total_loss / n_batches,
            "mse": total_mse / n_batches,
            "kl": total_kl / n_batches,
            "cos": total_cos / n_batches,
            "da_loss": total_da / n_batches,
            "lr": self.scheduler.get_last_lr()[0],
        }

    @staticmethod
    def _get_da_lambda(
        current: int,
        total: int,
        max_lambda: float,
    ) -> float:
        """Linearly ramp GRL lambda from 0 to *max_lambda* over training."""
        return max_lambda * min(1.0, current / max(1, total // 2))

    def _save_checkpoint(self, name: str, epoch: int, metrics: dict):
        path = self.checkpoint_dir / name
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metrics": metrics,
                "config": self.config,
            },
            path,
        )

    @staticmethod
    def _format_loss_components(metrics: dict) -> str:
        parts = [f"mse: {metrics['mse']:.4f}"]
        if metrics.get("kl", 0.0) > 1e-10:
            parts.append(f"kl: {metrics['kl']:.4f}")
        if metrics.get("cos", 0.0) > 1e-10:
            parts.append(f"cos: {metrics['cos']:.4f}")
        if metrics.get("da_loss", 0.0) > 1e-10:
            parts.append(f"da: {metrics['da_loss']:.4f}")
        return ", ".join(parts)

    def _log(self, epoch: int, train: dict, val: dict | None = None):
        msg = f"Epoch {epoch:3d} | train loss: {train['loss']:.4f} ({self._format_loss_components(train)})"
        if val:
            msg += f" | val loss: {val['loss']:.4f} ({self._format_loss_components(val)})"
        msg += f" | lr: {train['lr']:.6f}"
        print(msg)

        if self.config.use_wandb:
            import wandb
            log_dict = {f"train/{k}": v for k, v in train.items()}
            if val:
                log_dict.update({f"val/{k}": v for k, v in val.items()})
            log_dict["epoch"] = epoch
            wandb.log(log_dict)
