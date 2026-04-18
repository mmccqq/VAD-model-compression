#!/usr/bin/env python3
"""
Inference speed benchmark: Teacher CRDNN vs Student CRDNN (KD) on CPU.

Measures per-chunk latency, real-time factor, parameter count, and model size.
Designed to evaluate whether the student model is practical for edge/CPU devices.

Usage:
    python measure_speed.py
    python measure_speed.py --teacher_ckpt /path/to/teacher/model.ckpt \
                            --student_ckpt /path/to/student/model.ckpt
"""

import argparse
import os
import time

import torch
import torch.nn as nn

import speechbrain as sb
from speechbrain.lobes.features import Fbank
from speechbrain.nnet import RNN, containers, linear, normalization
from speechbrain.lobes.models.CRDNN import CNN_Block, DNN_Block
from speechbrain.processing.features import InputNormalization

# ── Default checkpoint paths ──────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TEACHER_CKPT = os.path.join(
    _HERE, "/Users/xiaoluochun/Documents/NEU/6140_machine_learning/src/project/backup_result/pretrained_models/vad-crdnn-libriparty/model.ckpt"
)
DEFAULT_STUDENT_CKPT = os.path.join(
    _HERE, "/Users/xiaoluochun/Documents/NEU/6140_machine_learning/src/project/backup_result/alpha0.5t2/VAD_CRDNN_KD/1986/save/CKPT+epoch_93/model.ckpt"
)

# ── Audio / feature constants (must match training hparams) ───────────────────
SAMPLE_RATE     = 16000
EXAMPLE_LENGTH  = 5          # seconds
N_FFT           = 400
N_MELS          = 40
HOP_LENGTH_MS   = 10.0       # ms  →  hop = 160 samples


# ─────────────────────────────────────────────────────────────────────────────
# Model builders
# ─────────────────────────────────────────────────────────────────────────────

def build_teacher():
    cnn = containers.Sequential(
        input_shape=[None, None, N_MELS],
        norm1=normalization.LayerNorm,
        cnn1=lambda input_shape: CNN_Block(input_shape, channels=16, kernel_size=(3, 3)),
        cnn2=lambda input_shape: CNN_Block(input_shape, channels=32, kernel_size=(3, 3)),
    )
    rnn = RNN.GRU(
        input_shape=[None, None, 320],
        hidden_size=32,
        num_layers=2,
        bidirectional=True,
    )
    dnn = containers.Sequential(
        input_shape=[None, None, 64],
        dnn1=lambda input_shape: DNN_Block(input_shape, neurons=16),
        dnn2=lambda input_shape: DNN_Block(input_shape, neurons=16),
        lin=lambda input_shape: linear.Linear(input_shape=input_shape,
                                              n_neurons=1, bias=False),
    )
    return nn.ModuleList([cnn, rnn, dnn])


def build_student():
    cnn = containers.Sequential(
        input_shape=[None, None, N_MELS],
        norm1=normalization.LayerNorm,
        cnn1=lambda input_shape: CNN_Block(input_shape, channels=8,  kernel_size=(3, 3)),
        cnn2=lambda input_shape: CNN_Block(input_shape, channels=16, kernel_size=(3, 3)),
    )
    rnn = RNN.GRU(
        input_shape=[None, None, 160],
        hidden_size=16,
        num_layers=2,
        bidirectional=True,
    )
    dnn = containers.Sequential(
        input_shape=[None, None, 32],
        dnn1=lambda input_shape: DNN_Block(input_shape, neurons=8),
        lin=lambda input_shape: linear.Linear(input_shape=input_shape,
                                              n_neurons=1, bias=False),
    )
    return nn.ModuleList([cnn, rnn, dnn])


# ─────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ─────────────────────────────────────────────────────────────────────────────

def run_model(model, feats):
    """Run CNN → reshape → RNN → DNN and return logits."""
    cnn, rnn, dnn = model[0], model[1], model[2]
    out = cnn(feats)
    out = out.reshape(out.shape[0], out.shape[1], out.shape[2] * out.shape[3])
    out, _ = rnn(out)
    logit = dnn(out)
    return logit


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def model_size_mb(model):
    """Estimate model size from parameter bytes."""
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return total_bytes / (1024 ** 2)


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark
# ─────────────────────────────────────────────────────────────────────────────

def benchmark(model, feats, n_warmup=20, n_runs=200):
    """
    Run n_warmup passes to warm up caches, then time n_runs passes.
    Returns list of latencies in seconds.
    """
    model.eval()
    with torch.no_grad():
        for _ in range(n_warmup):
            run_model(model, feats)

        latencies = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            run_model(model, feats)
            latencies.append(time.perf_counter() - t0)

    return latencies


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device("cpu")
    audio_duration = EXAMPLE_LENGTH  # seconds

    # ── Prepare input ─────────────────────────────────────────────────────────
    print("Preparing input features...")
    n_samples = SAMPLE_RATE * EXAMPLE_LENGTH
    wavs = torch.randn(1, n_samples)                          # [1, 80000]
    lens = torch.ones(1)

    fbank = Fbank(sample_rate=SAMPLE_RATE, n_fft=N_FFT,
                  n_mels=N_MELS, hop_length=HOP_LENGTH_MS)
    normalizer = InputNormalization(norm_type="sentence")

    with torch.no_grad():
        feats = fbank(wavs)                                   # [1, T, 40]
        feats = normalizer(feats, lens)
        feats = feats.detach()

    print(f"Input feature shape: {feats.shape}  "
          f"(batch=1, frames={feats.shape[1]}, mels={feats.shape[2]})\n")

    # ── Build models ──────────────────────────────────────────────────────────
    print("Building models...")
    teacher = build_teacher().to(device)
    student = build_student().to(device)

    # ── Load parameters at checkpoints ──────────────────────────────────────────────────────
    for name, model, ckpt_path in [("Teacher", teacher, args.teacher_ckpt),
                                    ("Student", student, args.student_ckpt)]:
        if os.path.exists(ckpt_path):
            state = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(state)
            print(f"  {name}: loaded from {ckpt_path}")
        else:
            print(f"  {name}: checkpoint not found at {ckpt_path}, "
                  f"using random weights (speed result still valid)")

    print()

    # ── Run benchmarks ────────────────────────────────────────────────────────
    print(f"Benchmarking ({args.n_warmup} warmup + {args.n_runs} timed runs, "
          f"batch_size=1, CPU only)...\n")

    teacher_times = benchmark(teacher, feats, args.n_warmup, args.n_runs)
    student_times = benchmark(student, feats, args.n_warmup, args.n_runs)

    # ── Compute statistics ────────────────────────────────────────────────────
    def stats(times):
        t = torch.tensor(times)
        return {
            "mean_ms":  t.mean().item() * 1000,
            "std_ms":   t.std().item()  * 1000,
            "min_ms":   t.min().item()  * 1000,
            "max_ms":   t.max().item()  * 1000,
            "rtf":      t.mean().item() / audio_duration,
        }

    ts = stats(teacher_times)
    ss = stats(student_times)

    t_params = count_parameters(teacher)
    s_params = count_parameters(student)
    t_mb     = model_size_mb(teacher)
    s_mb     = model_size_mb(student)

    speedup  = ts["mean_ms"] / ss["mean_ms"]
    #rtf_improvement = ts["rtf"] / ss["rtf"]

    # ── Print results ─────────────────────────────────────────────────────────
    sep = "─" * 62
    print(sep)
    print(f"{'':30s} {'Teacher':>14s} {'Student':>14s}")
    print(sep)
    print(f"{'Parameters':30s} {t_params:>14,d} {s_params:>14,d}")
    print(f"{'Model size (MB)':30s} {t_mb:>14.3f} {s_mb:>14.3f}")
    print(sep)
    print(f"{'Mean latency (ms)':30s} {ts['mean_ms']:>14.2f} {ss['mean_ms']:>14.2f}")
    print(f"{'Std  latency (ms)':30s} {ts['std_ms']:>14.2f} {ss['std_ms']:>14.2f}")
    print(f"{'Min  latency (ms)':30s} {ts['min_ms']:>14.2f} {ss['min_ms']:>14.2f}")
    print(f"{'Max  latency (ms)':30s} {ts['max_ms']:>14.2f} {ss['max_ms']:>14.2f}")
    print(sep)
    print(f"{'Real-time factor (RTF)':30s} {ts['rtf']:>14.4f} {ss['rtf']:>14.4f}")
    print(sep)
    print()
    print(f"  Speedup:              {speedup:.2f}×  (student is {speedup:.2f}× faster)")
    print(f"  Parameter reduction:  {t_params / s_params:.2f}×")
    print(f"  Model size reduction: {t_mb / s_mb:.2f}×")
    print()

    # ── RTF interpretation ────────────────────────────────────────────────────
    print("RTF interpretation:")
    print(f"  RTF < 1.0 → model can process audio faster than real time (real-time capable)")
    print(f"  RTF > 1.0 → model is too slow for real-time processing on this device")
    for name, rtf in [("Teacher", ts["rtf"]), ("Student", ss["rtf"])]:
        status = "✓ real-time capable" if rtf < 1.0 else "✗ too slow for real-time"
        print(f"  {name}: RTF={rtf:.4f}  →  {status}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark teacher vs student VAD model inference speed on CPU"
    )
    parser.add_argument("--teacher_ckpt", default=DEFAULT_TEACHER_CKPT,
                        help="Path to teacher model.ckpt")
    parser.add_argument("--student_ckpt", default=DEFAULT_STUDENT_CKPT,
                        help="Path to student model.ckpt")
    parser.add_argument("--n_warmup", type=int, default=20,
                        help="Number of warmup runs before timing (default: 20)")
    parser.add_argument("--n_runs",   type=int, default=200,
                        help="Number of timed runs (default: 200)")
    args = parser.parse_args()
    main(args)