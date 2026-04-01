"""
Helper module for FSDP/ZeRO sharding notebook.

Functions and classes used by mp.spawn must be defined in an importable module,
not in Jupyter notebook cells, because the "spawn" start method pickles the
target function by reference (module + qualname). Notebook cells define objects
in __main__, which doesn't exist in spawned child processes.
"""

import os
import math
import time

import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
from dataclasses import dataclass
from functools import partial
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    vocab_size:  int = 32000
    seq_len:     int = 512
    n_layers:    int = 6
    d_model:     int = 512
    n_heads:     int = 8
    d_ff:        int = 2048
    dropout:     float = 0.1


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.d_head  = cfg.d_model // cfg.n_heads
        self.qkv     = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj    = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.drop    = nn.Dropout(cfg.dropout)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(cfg.seq_len, cfg.seq_len))
                .view(1, 1, cfg.seq_len, cfg.seq_len),
        )

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).split(C, dim=-1)
        def reshape(t):
            return t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        q, k, v = map(reshape, qkv)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = self.drop(torch.softmax(att, dim=-1))
        out = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1  = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2  = nn.LayerNorm(cfg.d_model)
        self.ff   = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff, cfg.d_model),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class ToyGPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg    = cfg
        self.embed  = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos    = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )
        self.ln_f   = nn.LayerNorm(cfg.d_model)
        self.head   = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.shape
        pos  = torch.arange(T, device=idx.device)
        x    = self.embed(idx) + self.pos(pos)
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def setup_process_group(rank: int, world_size: int, backend: str = "nccl"):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def cleanup_process_group():
    dist.destroy_process_group()


def get_batch(batch_size, seq_len, vocab_size, device):
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    y = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    return x, y


# ---------------------------------------------------------------------------
# Worker functions (passed to mp.spawn)
# ---------------------------------------------------------------------------

def ddp_train_fn(rank: int, world_size: int, results_queue, use_gpu, cfg, n_steps: int = 5):
    backend = "nccl" if use_gpu else "gloo"
    setup_process_group(rank, world_size, backend)
    device = torch.device(f"cuda:{rank}") if use_gpu else torch.device("cpu")

    m = ToyGPT(cfg).to(device)
    m = DDP(m, device_ids=[rank] if use_gpu else None)

    optimizer = torch.optim.AdamW(m.parameters(), lr=1e-4)
    loss_fn   = nn.CrossEntropyLoss()

    if use_gpu:
        torch.cuda.reset_peak_memory_stats(device)

    step_times = []
    for step in range(n_steps):
        t0 = time.time()

        x, y    = get_batch(2, cfg.seq_len, cfg.vocab_size, device)
        logits  = m(x)
        loss    = loss_fn(logits.view(-1, cfg.vocab_size), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step_times.append(time.time() - t0)

    peak_mb = 0.0
    if use_gpu:
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    if rank == 0:
        results_queue.put({
            "method":         "DDP",
            "world_size":     world_size,
            "mean_step_ms":   np.mean(step_times[1:]) * 1000,
            "peak_memory_mb": peak_mb,
        })
    cleanup_process_group()


def fsdp_train_fn(rank, world_size, results_queue, sharding_strategy, use_gpu, cfg, n_steps=5):
    backend = "nccl" if use_gpu else "gloo"
    setup_process_group(rank, world_size, backend)
    device = torch.device(f"cuda:{rank}") if use_gpu else torch.device("cpu")

    m = ToyGPT(cfg).to(device)

    wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={TransformerBlock},
    )

    m = FSDP(
        m,
        auto_wrap_policy=wrap_policy,
        sharding_strategy=sharding_strategy,
        device_id=rank if use_gpu else None,
    )

    optimizer = torch.optim.AdamW(m.parameters(), lr=1e-4)
    loss_fn   = nn.CrossEntropyLoss()

    if use_gpu:
        torch.cuda.reset_peak_memory_stats(device)

    step_times = []
    for step in range(n_steps):
        t0 = time.time()

        x, y   = get_batch(2, cfg.seq_len, cfg.vocab_size, device)
        logits = m(x)
        loss   = loss_fn(logits.view(-1, cfg.vocab_size), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step_times.append(time.time() - t0)

    peak_mb = 0.0
    if use_gpu:
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    if rank == 0:
        results_queue.put({
            "method":         f"FSDP/{sharding_strategy.name}",
            "world_size":     world_size,
            "mean_step_ms":   np.mean(step_times[1:]) * 1000,
            "peak_memory_mb": peak_mb,
        })
    cleanup_process_group()
