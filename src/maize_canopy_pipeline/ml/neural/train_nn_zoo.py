#!/usr/bin/env python3
"""
NN Model Zoo for Helios surrogate (leaf-structured tabular) — v2

Adds:
- fixed Sobol-order 80/10/10 split (first 10% test, next 10% val, remaining 80% train)
- canopy-aware derived leaf features (area, projected-area proxy, normalized height, rank, cumulative-above)
- stable geo encodings while KEEPING raw lat/lon (ECEF recommended)
- leaf_transformer: ordered-leaf Transformer encoder (learns inter-leaf shading interactions)
- cumulative_shade: top-down transmittance style network (explicit shading inductive bias)

Still includes:
- deepsets
- deepsets_layerpool
- resnet_mlp
- moe_mlp

Assumptions:
- 12 leaves indexed 0..11
- per-leaf columns: interleaf_pos_i,length_cm_i,width_cm_i,theta_deg_i,phi_deg_i,curv_i,twist_i
- global columns: lat_deg, lon_deg, stalk_scale
- target: net_PAR (daily-integrated PAR; Aug 7 2020 constant across dataset)

Run:
  python nn_model_zoo_v2.py --data_csv results/ml_dataset_scaled.csv --models leaf_transformer,cumulative_shade,deepsets_layerpool
"""

import argparse
import math
import os
import json
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# -------------------------
# Utilities / Metrics
# -------------------------
def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


@dataclass
class RunResult:
    model: str
    test_rmse: float
    test_mae: float
    test_r2: float
    search: str
    params: dict


# -------------------------
# Feature parsing / encoding
# -------------------------
def build_feature_spec(n_leaves: int = 12) -> Tuple[List[str], Dict[int, Dict[str, str]]]:
    global_cols = ["lat_deg", "lon_deg", "stalk_scale"]
    leaf_cols = {}
    for i in range(n_leaves):
        leaf_cols[i] = {
            "interleaf_pos": f"interleaf_pos_{i}",
            "length_cm": f"length_cm_{i}",
            "width_cm": f"width_cm_{i}",
            "theta_deg": f"theta_deg_{i}",
            "phi_deg": f"phi_deg_{i}",
            "curv": f"curv_{i}",
            "twist": f"twist_{i}",
        }
    return global_cols, leaf_cols


def encode_angles_deg(theta_deg: np.ndarray, phi_deg: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    theta = np.deg2rad(theta_deg)
    phi = np.deg2rad(phi_deg)
    return np.sin(theta), np.cos(theta), np.sin(phi), np.cos(phi)


def ecef_from_latlon(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    lat = np.deg2rad(lat_deg.astype(float))
    lon = np.deg2rad(lon_deg.astype(float))
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return np.stack([x, y, z], axis=1).astype(np.float32)


def dataframe_to_arrays(
    df: pd.DataFrame,
    target_col: str,
    n_leaves: int = 12,
    use_log1p: bool = False,
    geo_mode: str = "raw+ecef",  # raw | raw+ecef | raw+sincos | raw+ecef+sincos
    sort_by_height: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      X_global: (N, G)
      X_leaf:   (N, L, F_leaf)
      y:        (N,)
    Leaf features (v2):
      base: [pos, length, width, sin(th), cos(th), sin(phi), cos(phi), curv, twist] = 9
      derived: [area, proj, hnorm, rank, cum_area_above, cum_proj_above] = 6
      total: 15
    """
    global_cols, leaf_cols = build_feature_spec(n_leaves)

    # Check columns
    missing = [c for c in global_cols + [leaf_cols[i][k] for i in range(n_leaves) for k in leaf_cols[i]] + [target_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns (showing up to 20): {missing[:20]}")

    # Global
    g = df[global_cols].to_numpy(dtype=np.float32)  # lat, lon, stalk_scale
    lat = g[:, 0]
    lon = g[:, 1]

    global_feats = [g]
    if "ecef" in geo_mode:
        global_feats.append(ecef_from_latlon(lat, lon))
    if "sincos" in geo_mode:
        lonr = np.deg2rad(lon.astype(float))
        global_feats.append(np.stack([np.sin(lonr), np.cos(lonr)], axis=1).astype(np.float32))

    X_global = np.concatenate(global_feats, axis=1).astype(np.float32)

    # Build leaf matrices: (N, L)
    pos = np.stack([df[leaf_cols[i]["interleaf_pos"]].to_numpy(dtype=np.float32) for i in range(n_leaves)], axis=1)
    length = np.stack([df[leaf_cols[i]["length_cm"]].to_numpy(dtype=np.float32) for i in range(n_leaves)], axis=1)
    width = np.stack([df[leaf_cols[i]["width_cm"]].to_numpy(dtype=np.float32) for i in range(n_leaves)], axis=1)
    theta = np.stack([df[leaf_cols[i]["theta_deg"]].to_numpy(dtype=np.float32) for i in range(n_leaves)], axis=1)
    phi = np.stack([df[leaf_cols[i]["phi_deg"]].to_numpy(dtype=np.float32) for i in range(n_leaves)], axis=1)
    curv = np.stack([df[leaf_cols[i]["curv"]].to_numpy(dtype=np.float32) for i in range(n_leaves)], axis=1)
    twist = np.stack([df[leaf_cols[i]["twist"]].to_numpy(dtype=np.float32) for i in range(n_leaves)], axis=1)

    s_th, c_th, s_ph, c_ph = encode_angles_deg(theta, phi)

    area = (length * width).astype(np.float32)
    proj = (area * np.abs(c_th)).astype(np.float32)  # shading proxy

    if sort_by_height:
        order = np.argsort(pos, axis=1)  # bottom -> top

        def take(a):
            return np.take_along_axis(a, order, axis=1)

        pos = take(pos); length = take(length); width = take(width)
        s_th = take(s_th); c_th = take(c_th); s_ph = take(s_ph); c_ph = take(c_ph)
        curv = take(curv); twist = take(twist)
        area = take(area); proj = take(proj)

    # Derived: normalized height and rank
    eps = 1e-9
    hmin = pos.min(axis=1, keepdims=True)
    hmax = pos.max(axis=1, keepdims=True)
    hnorm = ((pos - hmin) / (hmax - hmin + eps)).astype(np.float32)
    rank = np.linspace(0.0, 1.0, n_leaves, dtype=np.float32)[None, :].repeat(pos.shape[0], axis=0)

    # Derived: cumulative above (higher index is above after sorting)
    suffix_area = np.cumsum(area[:, ::-1], axis=1)[:, ::-1]
    suffix_proj = np.cumsum(proj[:, ::-1], axis=1)[:, ::-1]
    cum_area_above = (suffix_area - area).astype(np.float32)
    cum_proj_above = (suffix_proj - proj).astype(np.float32)

    # Stack leaf tensor: (N, L, F)
    X_leaf = np.stack([
        pos, length, width,
        s_th.astype(np.float32), c_th.astype(np.float32),
        s_ph.astype(np.float32), c_ph.astype(np.float32),
        curv, twist,
        area, proj, hnorm, rank, cum_area_above, cum_proj_above
    ], axis=2).astype(np.float32)

    y = df[target_col].to_numpy(dtype=np.float32)
    if use_log1p:
        y = np.log1p(np.maximum(y, 0.0)).astype(np.float32)

    return X_global, X_leaf, y


def flatten_for_tabular(X_global: np.ndarray, X_leaf: np.ndarray) -> np.ndarray:
    N, L, Fd = X_leaf.shape
    X_leaf_flat = X_leaf.reshape(N, L * Fd)
    return np.concatenate([X_global, X_leaf_flat], axis=1).astype(np.float32)


# -------------------------
# Torch datasets
# -------------------------
class LeafDataset(Dataset):
    def __init__(self, X_global: np.ndarray, X_leaf: np.ndarray, y: np.ndarray):
        self.Xg = torch.from_numpy(X_global)
        self.Xl = torch.from_numpy(X_leaf)
        self.y = torch.from_numpy(y).unsqueeze(1)

    def __len__(self) -> int:
        return self.Xg.shape[0]

    def __getitem__(self, idx: int):
        return self.Xg[idx], self.Xl[idx], self.y[idx]


class TabDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y).unsqueeze(1)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


# -------------------------
# Modules / models
# -------------------------
def make_mlp(in_dim: int, hidden: List[int], dropout: float, act: str = "gelu") -> nn.Sequential:
    acts = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "tanh": nn.Tanh,
    }
    if act not in acts:
        raise ValueError(f"Unknown activation: {act}")
    layers: List[nn.Module] = []
    d = in_dim
    for h in hidden:
        layers.append(nn.Linear(d, h))
        layers.append(acts[act]())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        d = h
    return nn.Sequential(*layers)


class DeepSetsRegressor(nn.Module):
    def __init__(
        self,
        g_dim: int,
        leaf_dim: int,
        emb_dim: int = 64,
        leaf_hidden: List[int] = [128, 128],
        head_hidden: List[int] = [256, 128],
        dropout: float = 0.1,
        act: str = "gelu",
        pooling: str = "mean_max",  # "mean", "mean_max"
    ):
        super().__init__()
        self.pooling = pooling
        self.leaf_enc = make_mlp(leaf_dim, leaf_hidden, dropout=dropout, act=act)
        last_leaf = leaf_hidden[-1] if leaf_hidden else leaf_dim
        self.leaf_proj = nn.Linear(last_leaf, emb_dim)
        pooled_dim = emb_dim * (2 if pooling == "mean_max" else 1)
        self.head = nn.Sequential(
            make_mlp(g_dim + pooled_dim, head_hidden, dropout=dropout, act=act),
            nn.Linear(head_hidden[-1], 1),
        )

    def forward(self, Xg: torch.Tensor, Xl: torch.Tensor) -> torch.Tensor:
        B, L, Fd = Xl.shape
        h = self.leaf_enc(Xl.reshape(B * L, Fd))
        h = self.leaf_proj(h).reshape(B, L, -1)  # (B, L, E)

        if self.pooling == "mean":
            pooled = h.mean(dim=1)
        elif self.pooling == "mean_max":
            pooled = torch.cat([h.mean(dim=1), h.max(dim=1).values], dim=1)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        x = torch.cat([Xg, pooled], dim=1)
        return self.head(x)


class DeepSetsLayerPoolRegressor(nn.Module):
    def __init__(
        self,
        g_dim: int,
        leaf_dim: int,
        groups: List[List[int]],
        emb_dim: int = 64,
        leaf_hidden: List[int] = [128, 128],
        head_hidden: List[int] = [256, 128],
        dropout: float = 0.1,
        act: str = "gelu",
        pooling: str = "mean_max",
    ):
        super().__init__()
        self.groups = groups
        self.pooling = pooling
        self.leaf_enc = make_mlp(leaf_dim, leaf_hidden, dropout=dropout, act=act)
        last_leaf = leaf_hidden[-1] if leaf_hidden else leaf_dim
        self.leaf_proj = nn.Linear(last_leaf, emb_dim)
        pooled_per_group = emb_dim * (2 if pooling == "mean_max" else 1)
        pooled_dim = pooled_per_group * len(groups)

        self.head = nn.Sequential(
            make_mlp(g_dim + pooled_dim, head_hidden, dropout=dropout, act=act),
            nn.Linear(head_hidden[-1], 1),
        )

    def _pool_group(self, h_group: torch.Tensor) -> torch.Tensor:
        if self.pooling == "mean":
            return h_group.mean(dim=1)
        elif self.pooling == "mean_max":
            return torch.cat([h_group.mean(dim=1), h_group.max(dim=1).values], dim=1)
        raise ValueError(f"Unknown pooling: {self.pooling}")

    def forward(self, Xg: torch.Tensor, Xl: torch.Tensor) -> torch.Tensor:
        B, L, Fd = Xl.shape
        h = self.leaf_enc(Xl.reshape(B * L, Fd))
        h = self.leaf_proj(h).reshape(B, L, -1)

        pooled_groups = []
        for idxs in self.groups:
            pooled_groups.append(self._pool_group(h[:, idxs, :]))
        pooled = torch.cat(pooled_groups, dim=1)

        x = torch.cat([Xg, pooled], dim=1)
        return self.head(x)


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float, act: str = "gelu"):
        super().__init__()
        act_layer = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU, "tanh": nn.Tanh}[act]
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.norm = nn.LayerNorm(dim)
        self.act = act_layer()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)
        h = self.act(h)
        h = self.drop(h)
        h = self.fc2(h)
        x = x + h
        return self.norm(x)


class ResNetMLPRegressor(nn.Module):
    def __init__(
        self,
        in_dim: int,
        width: int = 512,
        n_blocks: int = 4,
        block_hidden: int = 512,
        head_hidden: int = 256,
        dropout: float = 0.1,
        act: str = "gelu",
    ):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, width),
            nn.LayerNorm(width),
            nn.GELU() if act == "gelu" else nn.ReLU(),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(width, block_hidden, dropout=dropout, act=act) for _ in range(n_blocks)])
        self.head = nn.Sequential(
            nn.Linear(width, head_hidden),
            nn.GELU() if act == "gelu" else nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x)
        x = self.blocks(x)
        return self.head(x)


class MoEMLPRegressor(nn.Module):
    def __init__(
        self,
        in_dim: int,
        n_experts: int = 4,
        expert_hidden: List[int] = [512, 256],
        gate_hidden: List[int] = [256],
        dropout: float = 0.1,
        act: str = "gelu",
    ):
        super().__init__()
        self.n_experts = n_experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                make_mlp(in_dim, expert_hidden, dropout=dropout, act=act),
                nn.Linear(expert_hidden[-1], 1),
            )
            for _ in range(n_experts)
        ])
        self.gate = nn.Sequential(
            make_mlp(in_dim, gate_hidden, dropout=dropout, act=act),
            nn.Linear(gate_hidden[-1], n_experts),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.gate(x)
        w = torch.softmax(logits, dim=1)
        preds = torch.cat([exp(x) for exp in self.experts], dim=1)
        out = (w * preds).sum(dim=1, keepdim=True)
        return out


class LeafTransformerRegressor(nn.Module):
    """
    Ordered-leaf Transformer encoder.

    Leaves are assumed already ordered bottom->top in Xl (we sort by interleaf_pos in dataframe_to_arrays).
    """
    def __init__(
        self,
        g_dim: int,
        leaf_dim: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        ff_dim: int = 256,
        dropout: float = 0.1,
        act: str = "gelu",
        use_cls: bool = True,
        head_hidden: List[int] = [256, 128],
    ):
        super().__init__()
        self.use_cls = use_cls
        self.leaf_in = nn.Linear(leaf_dim, d_model)

        # Simple learned positional embedding for 12 leaves
        self.pos_emb = nn.Parameter(torch.zeros(1, 12 + (1 if use_cls else 0), d_model))

        if use_cls:
            self.cls = nn.Parameter(torch.zeros(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu" if act == "gelu" else "relu",
            batch_first=True,
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        pooled_dim = d_model
        self.head = nn.Sequential(
            make_mlp(g_dim + pooled_dim, head_hidden, dropout=dropout, act=act),
            nn.Linear(head_hidden[-1], 1),
        )

        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        if use_cls:
            nn.init.trunc_normal_(self.cls, std=0.02)

    def forward(self, Xg: torch.Tensor, Xl: torch.Tensor) -> torch.Tensor:
        B, L, Fd = Xl.shape
        h = self.leaf_in(Xl)  # (B, L, d)
        if self.use_cls:
            cls = self.cls.expand(B, -1, -1)
            h = torch.cat([cls, h], dim=1)  # (B, 1+L, d)
        h = h + self.pos_emb[:, :h.shape[1], :]
        h = self.enc(h)
        pooled = h[:, 0, :] if self.use_cls else h.mean(dim=1)
        x = torch.cat([Xg, pooled], dim=1)
        return self.head(x)


class CumulativeShadeRegressor(nn.Module):
    """
    Explicit top-down shading inductive bias.
    Computes a learned transmittance down the canopy, then sums per-leaf capture.

    Leaves are assumed ordered bottom->top; we internally flip to top->bottom for cumulative attenuation.
    """
    def __init__(
        self,
        g_dim: int,
        leaf_dim: int,
        leaf_hidden: List[int] = [128, 128],
        dropout: float = 0.1,
        act: str = "gelu",
        diffuse_hidden: List[int] = [128],
    ):
        super().__init__()
        self.leaf_enc = make_mlp(leaf_dim, leaf_hidden, dropout=dropout, act=act)
        last = leaf_hidden[-1] if leaf_hidden else leaf_dim

        # absorb >=0 and atten >=0
        self.to_absorb = nn.Linear(last, 1)
        self.to_atten = nn.Linear(last, 1)

        # diffuse term depends on global + canopy pooled stats
        self.diffuse = nn.Sequential(
            make_mlp(g_dim + last, diffuse_hidden, dropout=dropout, act=act),
            nn.Linear(diffuse_hidden[-1], 1),
        )

    def forward(self, Xg: torch.Tensor, Xl: torch.Tensor) -> torch.Tensor:
        B, L, Fd = Xl.shape
        h = self.leaf_enc(Xl.reshape(B * L, Fd)).reshape(B, L, -1)  # (B, L, H)

        absorb = F.softplus(self.to_absorb(h)).squeeze(-1)  # (B, L)
        atten = F.softplus(self.to_atten(h)).squeeze(-1)    # (B, L)

        # Flip to top->bottom for attenuation cascade
        absorb_tb = absorb.flip(dims=[1])
        atten_tb = atten.flip(dims=[1])

        # cumulative transmittance: T0=1, T_{i+1}=T_i*exp(-atten_i)
        T = torch.ones((B, 1), device=Xg.device, dtype=absorb.dtype)
        contribs = []
        for i in range(L):
            # light available at this layer
            contribs.append(T.squeeze(1) * absorb_tb[:, i])
            T = T * torch.exp(-atten_tb[:, i:i+1])

        captured_topdown = torch.stack(contribs, dim=1).sum(dim=1, keepdim=True)  # (B,1)

        # Diffuse / scattered term: use pooled leaf hidden + global
        pooled = h.mean(dim=1)
        diffuse = self.diffuse(torch.cat([Xg, pooled], dim=1))
        return captured_topdown + diffuse


# -------------------------
# Training
# -------------------------
def huber_loss(pred: torch.Tensor, y: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    return F.huber_loss(pred, y, delta=delta)


def train_one_fold(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    lr: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
    loss_name: str,
    huber_delta: float,
    grad_clip: Optional[float] = None,
) -> Tuple[nn.Module, Dict[str, float]]:
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5, verbose=False)

    best_val = float("inf")
    best_state = None
    best_epoch = 0
    bad = 0
    final_epoch = 0

    for _epoch in range(1, max_epochs + 1):
        final_epoch = _epoch
        model.train()
        for batch in train_loader:
            opt.zero_grad(set_to_none=True)
            if len(batch) == 3:
                Xg, Xl, y = batch
                Xg = Xg.to(device); Xl = Xl.to(device); y = y.to(device)
                pred = model(Xg, Xl)
            else:
                X, y = batch
                X = X.to(device); y = y.to(device)
                pred = model(X)

            if loss_name == "mse":
                loss = F.mse_loss(pred, y)
            elif loss_name == "huber":
                loss = huber_loss(pred, y, delta=huber_delta)
            else:
                raise ValueError("loss_name must be 'mse' or 'huber'")

            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

        # validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 3:
                    Xg, Xl, y = batch
                    Xg = Xg.to(device); Xl = Xl.to(device); y = y.to(device)
                    pred = model(Xg, Xl)
                else:
                    X, y = batch
                    X = X.to(device); y = y.to(device)
                    pred = model(X)

                vloss = F.mse_loss(pred, y).item() if loss_name == "mse" else huber_loss(pred, y, delta=huber_delta).item()
                val_losses.append(vloss)

        val_loss = float(np.mean(val_losses))
        sched.step(val_loss)

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_epoch = _epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, {
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "final_epoch": final_epoch,
        "best_state_dict": best_state,
        "last_state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "optimizer_state_dict": opt.state_dict(),
        "scheduler_state_dict": sched.state_dict(),
    }


@torch.no_grad()
def predict_model(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    preds = []
    for batch in loader:
        if len(batch) == 3:
            Xg, Xl, _y = batch
            Xg = Xg.to(device); Xl = Xl.to(device)
            pred = model(Xg, Xl)
        else:
            X, _y = batch
            X = X.to(device)
            pred = model(X)
        preds.append(pred.detach().cpu().numpy().reshape(-1))
    return np.concatenate(preds, axis=0)


# -------------------------
# Hyperparam sampling
# -------------------------
def sample_params(model_name: str, rng: random.Random) -> dict:
    act = rng.choice(["gelu", "silu", "relu"])
    dropout = rng.uniform(0.0, 0.2)
    lr = 10 ** rng.uniform(-4.2, -2.8)
    wd = 10 ** rng.uniform(-5.0, -2.0)
    loss = rng.choice(["huber", "mse"])
    huber_delta = rng.choice([0.5, 1.0, 2.0, 5.0])
    batch_size = rng.choice([128, 256, 512, 1024])

    if model_name in ("deepsets", "deepsets_layerpool"):
        emb_dim = rng.choice([64, 96, 128])
        leaf_hidden = rng.choice([[128, 128], [256, 128], [256, 256]])
        head_hidden = rng.choice([[256, 128], [512, 256], [256, 256]])
        pooling = rng.choice(["mean", "mean_max"])
        return dict(
            act=act, dropout=dropout, lr=lr, weight_decay=wd, loss=loss, huber_delta=huber_delta,
            batch_size=batch_size, emb_dim=emb_dim, leaf_hidden=leaf_hidden, head_hidden=head_hidden, pooling=pooling
        )

    if model_name == "leaf_transformer":
        d_model = rng.choice([96, 128, 160, 192])
        n_heads = 4 if d_model % 4 == 0 else 8
        n_layers = rng.choice([2, 3, 4])
        ff_dim = rng.choice([256, 384, 512])
        use_cls = rng.choice([True, True, False])
        head_hidden = rng.choice([[256, 128], [512, 256], [256, 256]])
        return dict(
            act=act, dropout=dropout, lr=lr, weight_decay=wd, loss=loss, huber_delta=huber_delta,
            batch_size=batch_size, d_model=d_model, n_heads=n_heads, n_layers=n_layers, ff_dim=ff_dim,
            use_cls=use_cls, head_hidden=head_hidden
        )

    if model_name == "cumulative_shade":
        leaf_hidden = rng.choice([[128, 128], [256, 128], [256, 256]])
        diffuse_hidden = rng.choice([[64], [128], [128, 64]])
        return dict(
            act=act, dropout=dropout, lr=lr, weight_decay=wd, loss=loss, huber_delta=huber_delta,
            batch_size=batch_size, leaf_hidden=leaf_hidden, diffuse_hidden=diffuse_hidden
        )

    if model_name == "resnet_mlp":
        width = rng.choice([256, 384, 512, 768])
        n_blocks = rng.choice([3, 4, 5, 6])
        block_hidden = rng.choice([width, int(width * 1.5), int(width * 2)])
        head_hidden = rng.choice([128, 256, 384])
        return dict(
            act=act, dropout=dropout, lr=lr, weight_decay=wd, loss=loss, huber_delta=huber_delta,
            batch_size=batch_size, width=width, n_blocks=n_blocks, block_hidden=block_hidden, head_hidden=head_hidden
        )

    if model_name == "moe_mlp":
        n_experts = rng.choice([3, 4, 5])
        expert_hidden = rng.choice([[512, 256], [768, 384], [512, 512, 256]])
        gate_hidden = rng.choice([[256], [384], [256, 128]])
        return dict(
            act=act, dropout=dropout, lr=lr, weight_decay=wd, loss=loss, huber_delta=huber_delta,
            batch_size=batch_size, n_experts=n_experts, expert_hidden=expert_hidden, gate_hidden=gate_hidden
        )

    raise ValueError(f"Unknown model_name: {model_name}")


# -------------------------
# Checkpoint / manifest helpers
# -------------------------
def build_scaler_state(scaler: StandardScaler) -> Dict[str, object]:
    state = {
        "mean_": scaler.mean_.tolist(),
        "scale_": scaler.scale_.tolist(),
        "var_": scaler.var_.tolist(),
        "n_features_in_": int(scaler.n_features_in_),
    }
    if hasattr(scaler, "n_samples_seen_"):
        n_seen = scaler.n_samples_seen_
        state["n_samples_seen_"] = n_seen.tolist() if hasattr(n_seen, "tolist") else int(n_seen)
    return state


def save_json(path: str, payload: Dict[str, object]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_trial_artifacts(
    out_dir: str,
    model_name: str,
    trial_idx: int,
    trial_seed: int,
    params: dict,
    train_info: dict,
    metrics: dict,
    split_indices: Dict[str, np.ndarray],
    scaler_states: Dict[str, Dict[str, object]],
    run_meta: Dict[str, object],
) -> None:
    trial_dir = os.path.join(out_dir, model_name, f"trial_{trial_idx:03d}")
    os.makedirs(trial_dir, exist_ok=True)

    common_ckpt = {
        "model_name": model_name,
        "trial_index": trial_idx,
        "trial_seed": trial_seed,
        "params": params,
        "metrics": metrics,
        "best_val_loss": train_info["best_val_loss"],
        "best_epoch": train_info["best_epoch"],
        "final_epoch": train_info["final_epoch"],
        "optimizer_state_dict": train_info["optimizer_state_dict"],
        "scheduler_state_dict": train_info["scheduler_state_dict"],
        "scalers": scaler_states,
        "split": {k: v.tolist() for k, v in split_indices.items()},
        "run_meta": run_meta,
    }
    best_ckpt = dict(common_ckpt)
    best_ckpt["model_state_dict"] = train_info["best_state_dict"]
    last_ckpt = dict(common_ckpt)
    last_ckpt["model_state_dict"] = train_info["last_state_dict"]

    torch.save(best_ckpt, os.path.join(trial_dir, "best_checkpoint.pt"))
    torch.save(last_ckpt, os.path.join(trial_dir, "last_checkpoint.pt"))

    manifest = {
        "model_name": model_name,
        "trial_index": trial_idx,
        "trial_seed": trial_seed,
        "params": params,
        "metrics": metrics,
        "best_val_loss": train_info["best_val_loss"],
        "best_epoch": train_info["best_epoch"],
        "final_epoch": train_info["final_epoch"],
        "run_meta": run_meta,
        "split_sizes": {k: int(len(v)) for k, v in split_indices.items()},
    }
    save_json(os.path.join(trial_dir, "manifest.json"), manifest)


# -------------------------
# Main CV runner
# -------------------------
def run_cv_random_search(
    df: pd.DataFrame,
    model_name: str,
    target_col: str,
    n_leaves: int,
    n_trials: int,
    n_splits: int,
    seed: int,
    device: torch.device,
    out_dir: str,
    use_log1p: bool,
    max_epochs: int,
    patience: int,
    num_workers: int,
    geo_mode: str,
    data_csv: str,
    save_every_trial: bool,
    deterministic: bool,
) -> List[RunResult]:
    """
    Uses a fixed Sobol-order split instead of shuffled K-fold CV:
      - first 10%   -> test
      - next 10%    -> validation
      - remaining   -> training

    This preserves the original sequence ordering, which is useful when the
    dataset was generated sequentially from a Sobol design and the user wants
    deterministic, reproducible evaluation on the earliest samples.
    """
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(seed)

    Xg, Xl, y = dataframe_to_arrays(
        df, target_col=target_col, n_leaves=n_leaves, use_log1p=use_log1p,
        geo_mode=geo_mode, sort_by_height=True
    )

    g_dim = Xg.shape[1]
    leaf_dim = Xl.shape[2]
    Xtab = flatten_for_tabular(Xg, Xl)
    tab_dim = Xtab.shape[1]

    N = len(Xg)
    test_end = int(0.10 * N)
    val_end = int(0.20 * N)

    if test_end < 1 or val_end <= test_end or val_end >= N:
        raise ValueError(
            f"Dataset too small for fixed 80/10/10 split: N={N}, "
            f"test_end={test_end}, val_end={val_end}"
        )

    te_idx = np.arange(0, test_end)         # first 10%
    val_idx = np.arange(test_end, val_end)  # next 10%
    tr_idx = np.arange(val_end, N)          # remaining 80%

    Xg_tr, Xg_val, Xg_te = Xg[tr_idx], Xg[val_idx], Xg[te_idx]
    Xl_tr, Xl_val, Xl_te = Xl[tr_idx], Xl[val_idx], Xl[te_idx]
    y_tr, y_val, y_te = y[tr_idx], y[val_idx], y[te_idx]

    # Fit scalers on train only
    sg = StandardScaler()
    Xg_tr_s = sg.fit_transform(Xg_tr)
    Xg_val_s = sg.transform(Xg_val)
    Xg_te_s = sg.transform(Xg_te)

    sl = StandardScaler()
    Xl_tr_2d = Xl_tr.reshape(-1, leaf_dim)
    Xl_val_2d = Xl_val.reshape(-1, leaf_dim)
    Xl_te_2d = Xl_te.reshape(-1, leaf_dim)
    Xl_tr_s = sl.fit_transform(Xl_tr_2d).reshape(Xl_tr.shape)
    Xl_val_s = sl.transform(Xl_val_2d).reshape(Xl_val.shape)
    Xl_te_s = sl.transform(Xl_te_2d).reshape(Xl_te.shape)

    st = StandardScaler()
    Xtab_tr = flatten_for_tabular(Xg_tr, Xl_tr)
    Xtab_val = flatten_for_tabular(Xg_val, Xl_val)
    Xtab_te = flatten_for_tabular(Xg_te, Xl_te)
    Xtab_tr_s = st.fit_transform(Xtab_tr).astype(np.float32)
    Xtab_val_s = st.transform(Xtab_val).astype(np.float32)
    Xtab_te_s = st.transform(Xtab_te).astype(np.float32)

    scaler_states = {
        "global": build_scaler_state(sg),
        "leaf": build_scaler_state(sl),
        "tabular": build_scaler_state(st),
    }
    split_indices = {"train_idx": tr_idx, "val_idx": val_idx, "test_idx": te_idx}
    run_meta = {
        "data_csv": data_csv,
        "target_col": target_col,
        "n_leaves": n_leaves,
        "geo_mode": geo_mode,
        "use_log1p": bool(use_log1p),
        "device": str(device),
        "base_seed": int(seed),
        "deterministic": bool(deterministic),
        "g_dim": int(g_dim),
        "leaf_dim": int(leaf_dim),
        "tab_dim": int(tab_dim),
        "split_strategy": "fixed_80_10_10_sobol_order",
    }

    results: List[RunResult] = []
    for t in range(n_trials):
        trial_seed = int(seed + t)
        set_seed(trial_seed, deterministic=deterministic)
        params = sample_params(model_name, rng)
        trial_tag = "fixed_80_10_10_sobol_order"

        bs = int(params["batch_size"])
        torch_gen = torch.Generator()
        torch_gen.manual_seed(trial_seed)

        if model_name in ("deepsets", "deepsets_layerpool", "leaf_transformer", "cumulative_shade"):
            train_ds = LeafDataset(Xg_tr_s, Xl_tr_s, y_tr)
            val_ds = LeafDataset(Xg_val_s, Xl_val_s, y_val)
            test_ds = LeafDataset(Xg_te_s, Xl_te_s, y_te)

            train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=num_workers, pin_memory=True, worker_init_fn=seed_worker if deterministic else None, generator=torch_gen)
            val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=num_workers, pin_memory=True, worker_init_fn=seed_worker if deterministic else None, generator=torch_gen)
            test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=num_workers, pin_memory=True, worker_init_fn=seed_worker if deterministic else None, generator=torch_gen)

            if model_name == "deepsets":
                model = DeepSetsRegressor(
                    g_dim=g_dim, leaf_dim=leaf_dim,
                    emb_dim=params["emb_dim"],
                    leaf_hidden=params["leaf_hidden"],
                    head_hidden=params["head_hidden"],
                    dropout=params["dropout"],
                    act=params["act"],
                    pooling=params["pooling"],
                )
            elif model_name == "deepsets_layerpool":
                groups = [list(range(0, 4)), list(range(4, 8)), list(range(8, 12))]
                model = DeepSetsLayerPoolRegressor(
                    g_dim=g_dim, leaf_dim=leaf_dim, groups=groups,
                    emb_dim=params["emb_dim"],
                    leaf_hidden=params["leaf_hidden"],
                    head_hidden=params["head_hidden"],
                    dropout=params["dropout"],
                    act=params["act"],
                    pooling=params["pooling"],
                )
            elif model_name == "leaf_transformer":
                model = LeafTransformerRegressor(
                    g_dim=g_dim, leaf_dim=leaf_dim,
                    d_model=params["d_model"],
                    n_heads=params["n_heads"],
                    n_layers=params["n_layers"],
                    ff_dim=params["ff_dim"],
                    dropout=params["dropout"],
                    act=params["act"],
                    use_cls=params["use_cls"],
                    head_hidden=params["head_hidden"],
                )
            else:
                model = CumulativeShadeRegressor(
                    g_dim=g_dim, leaf_dim=leaf_dim,
                    leaf_hidden=params["leaf_hidden"],
                    diffuse_hidden=params["diffuse_hidden"],
                    dropout=params["dropout"],
                    act=params["act"],
                )

        elif model_name in ("resnet_mlp", "moe_mlp"):
            train_ds = TabDataset(Xtab_tr_s, y_tr)
            val_ds = TabDataset(Xtab_val_s, y_val)
            test_ds = TabDataset(Xtab_te_s, y_te)

            train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=num_workers, pin_memory=True, worker_init_fn=seed_worker if deterministic else None, generator=torch_gen)
            val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=num_workers, pin_memory=True, worker_init_fn=seed_worker if deterministic else None, generator=torch_gen)
            test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=num_workers, pin_memory=True, worker_init_fn=seed_worker if deterministic else None, generator=torch_gen)

            if model_name == "resnet_mlp":
                model = ResNetMLPRegressor(
                    in_dim=tab_dim,
                    width=params["width"],
                    n_blocks=params["n_blocks"],
                    block_hidden=params["block_hidden"],
                    head_hidden=params["head_hidden"],
                    dropout=params["dropout"],
                    act=params["act"],
                )
            else:
                model = MoEMLPRegressor(
                    in_dim=tab_dim,
                    n_experts=params["n_experts"],
                    expert_hidden=params["expert_hidden"],
                    gate_hidden=params["gate_hidden"],
                    dropout=params["dropout"],
                    act=params["act"],
                )
        else:
            raise ValueError(model_name)

        trained, train_info = train_one_fold(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            lr=float(params["lr"]),
            weight_decay=float(params["weight_decay"]),
            max_epochs=max_epochs,
            patience=patience,
            loss_name=params["loss"],
            huber_delta=float(params["huber_delta"]),
            grad_clip=1.0,
        )

        y_pred = predict_model(trained, test_loader, device=device)

        if use_log1p:
            y_te_eval = np.expm1(y_te)
            y_pred_eval = np.expm1(y_pred)
        else:
            y_te_eval = y_te
            y_pred_eval = y_pred

        rmse_mean = rmse(y_te_eval, y_pred_eval)
        mae_mean = float(mean_absolute_error(y_te_eval, y_pred_eval))
        r2_mean = float(r2_score(y_te_eval, y_pred_eval))

        results.append(RunResult(
            model=model_name,
            test_rmse=rmse_mean,
            test_mae=mae_mean,
            test_r2=r2_mean,
            search=trial_tag,
            params=params,
        ))

        if save_every_trial:
            save_trial_artifacts(
                out_dir=out_dir,
                model_name=model_name,
                trial_idx=t + 1,
                trial_seed=trial_seed,
                params=params,
                train_info=train_info,
                metrics={"test_rmse": rmse_mean, "test_mae": mae_mean, "test_r2": r2_mean, "search": trial_tag},
                split_indices=split_indices,
                scaler_states=scaler_states,
                run_meta=run_meta,
            )

        print(
            f"[{model_name}] trial {t+1}/{n_trials}: "
            f"RMSE={rmse_mean:.4f}, MAE={mae_mean:.4f}, R2={r2_mean:.4f}"
        )

    out_csv = os.path.join(out_dir, f"{model_name}_results.csv")
    pd.DataFrame([{
        "model": r.model,
        "test_rmse": r.test_rmse,
        "test_mae": r.test_mae,
        "test_r2": r.test_r2,
        "search": r.search,
        "params": r.params,
    } for r in results]).to_csv(out_csv, index=False)
    print(f"Saved {model_name} results to: {out_csv}")
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_csv", required=True, help="Path to dataset CSV")
    ap.add_argument("--target", default="net_PAR", help="Target column")
    ap.add_argument("--models", default="leaf_transformer,cumulative_shade,deepsets_layerpool,resnet_mlp,moe_mlp,deepsets",
                    help="Comma-separated models")
    ap.add_argument("--n_leaves", type=int, default=12)
    ap.add_argument("--trials", type=int, default=25)
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="results/nn_model_zoo_v2")
    ap.add_argument("--use_log1p", action="store_true")
    ap.add_argument("--max_epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--geo_mode", default="raw+ecef", choices=["raw", "raw+ecef", "raw+sincos", "raw+ecef+sincos"])
    ap.add_argument("--save_every_trial", action="store_true", help="Save best/last checkpoints and manifests for every trial")
    ap.add_argument("--deterministic", action="store_true", help="Enable more reproducible CUDA/DataLoader behavior")
    args = ap.parse_args()

    set_seed(args.seed, deterministic=args.deterministic)

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = pd.read_csv(args.data_csv)
    if "simulation_id" in df.columns:
        df = df.drop(columns=["simulation_id"])

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    all_results: List[RunResult] = []
    os.makedirs(args.out_dir, exist_ok=True)

    run_manifest = {
        "data_csv": args.data_csv,
        "target": args.target,
        "models": models,
        "n_leaves": args.n_leaves,
        "trials": args.trials,
        "seed": args.seed,
        "out_dir": args.out_dir,
        "use_log1p": bool(args.use_log1p),
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "num_workers": args.num_workers,
        "device": str(device),
        "geo_mode": args.geo_mode,
        "save_every_trial": bool(args.save_every_trial),
        "deterministic": bool(args.deterministic),
        "split_strategy": "fixed_80_10_10_sobol_order",
    }
    save_json(os.path.join(args.out_dir, "run_manifest.json"), run_manifest)

    for m in models:
        res = run_cv_random_search(
            df=df,
            model_name=m,
            target_col=args.target,
            n_leaves=args.n_leaves,
            n_trials=args.trials,
            n_splits=args.cv,
            seed=args.seed,
            device=device,
            out_dir=args.out_dir,
            use_log1p=args.use_log1p,
            max_epochs=args.max_epochs,
            patience=args.patience,
            num_workers=args.num_workers,
            geo_mode=args.geo_mode,
            data_csv=args.data_csv,
            save_every_trial=args.save_every_trial,
            deterministic=args.deterministic,
        )
        all_results.extend(res)

    ranked = sorted(all_results, key=lambda r: r.test_rmse)
    ranked_csv = os.path.join(args.out_dir, "nn_model_zoo_results_ranked.csv")
    pd.DataFrame([{
        "model": r.model,
        "test_rmse": r.test_rmse,
        "test_mae": r.test_mae,
        "test_r2": r.test_r2,
        "search": r.search,
        "params": r.params,
    } for r in ranked]).to_csv(ranked_csv, index=False)

    print("===== DONE =====")
    print(f"Saved ranked results to: {ranked_csv}")
    print("Top 10 by test RMSE:")
    top10 = ranked[:10]
    tdf = pd.DataFrame([{
        "model": r.model,
        "test_rmse": r.test_rmse,
        "test_mae": r.test_mae,
        "test_r2": r.test_r2,
        "search": r.search,
    } for r in top10])
    with pd.option_context("display.max_rows", 20, "display.width", 140):
        print(tdf.to_string(index=False))


if __name__ == "__main__":
    main()
