"""
Sweetwater data utilities: pseudo-bulk generation, normalization.
Adapted from https://github.com/ML4BM-Lab/Sweetwater
"""
import math
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm


def generate_synthetic(scRNA, nsamples=1000):
    """Generate pseudo-bulk samples from single-cell reference.

    Parameters
    ----------
    scRNA : pd.DataFrame
        Single-cell expression matrix. Index = cell type labels, columns = genes.
    nsamples : int
        Number of pseudo-bulk samples to generate.

    Returns
    -------
    xpseudo : np.ndarray
        Pseudo-bulk expression (nsamples x n_genes).
    ypseudo : np.ndarray
        True proportions (nsamples x n_cell_types).
    celltypes : list
        List of cell type names.
    """
    # Build dict of cell type -> row indices
    celltypes = sorted(set(scRNA.index))
    ctdict = {ct: i for i, ct in enumerate(celltypes)}
    dfdict = {}
    for i, ct in enumerate(scRNA.index):
        idx = ctdict[ct]
        if idx in dfdict:
            dfdict[idx].append(i)
        else:
            dfdict[idx] = [i]

    # Generate Dirichlet proportions
    def gen_fractions(samples, nct):
        combl = []
        ncombl = sum(len(list(combinations(range(nct), i))) for i in range(1, nct + 1))
        l = max(1, math.ceil(samples / ncombl))
        for i in range(1, nct + 1):
            cmb = list(combinations(range(nct), i))
            for e in cmb:
                combelm = np.zeros((l, nct))
                combelm[:, e] = np.random.dirichlet(alpha=np.ones(len(e)), size=l)
                combl.append(combelm)
        mat = np.vstack(combl)
        return mat[:samples]

    props = gen_fractions(samples=nsamples, nct=len(celltypes))

    # Generate expression
    sc_values = np.ascontiguousarray(scRNA.values, dtype=np.float32)
    ncells = np.random.randint(100, max(101, scRNA.shape[0] // 10), props.shape[0])

    xpseudo = np.zeros((nsamples, scRNA.shape[1]), dtype=np.float32)
    ypseudo = np.zeros((nsamples, len(celltypes)), dtype=np.float32)

    for i in tqdm(range(nsamples), desc='Generating pseudo-bulk'):
        props_int = np.int32(props[i] * ncells[i])
        props_int = np.maximum(props_int, 0)
        total = max(1, props_int.sum())
        props_adj = props_int.astype(float) / total
        ypseudo[i] = props_adj

        sample_expr = np.zeros(scRNA.shape[1], dtype=np.float32)
        for k in range(len(celltypes)):
            n = props_int[k]
            if n > 0 and k in dfdict:
                indices = np.random.choice(dfdict[k], size=min(n, len(dfdict[k])), replace=False)
                sample_expr += sc_values[indices].sum(axis=0)
        xpseudo[i] = sample_expr

    return xpseudo, ypseudo, celltypes


def transform_and_normalize(*args):
    """Apply log2(x+1) and MinMaxScaler to each matrix."""
    results = []
    for x in args:
        x_norm = np.log2(x + 1).T
        scaler = MinMaxScaler(feature_range=(0, 1))
        x_norm = scaler.fit_transform(x_norm).T
        results.append(x_norm)
    return results


def convert_to_float_tensors(*args):
    """Convert arrays to float torch tensors."""
    import torch
    return [torch.tensor(x).float() for x in args]
