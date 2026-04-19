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
DEFAULT_TEACHER_CKPT = "/Users/xiaoluochun/Documents/NEU/6140_machine_learning/src/project/pretrained_models/vad-crdnn-libriparty/model.ckpt"
DEFAULT_STUDENT_CKPT = "/Users/xiaoluochun/Documents/NEU/6140_machine_learning/src/project/backup_result/alpha0.5t4/VAD_CRDNN_KD/1986/save/CKPT+epoch_99/model.ckpt"
DEFAULT_PRUNED_CKPT     = "/Users/xiaoluochun/Documents/NEU/6140_machine_learning/src/project/VAD-model-compression/VAD-Pruning-GRU-Layer/artifacts/pruned_finetuned_vad/vad_pruned_finetuned_state.pt"
DEFAULT_PRUNED_KD_CKPT  = "/Users/xiaoluochun/Documents/NEU/6140_machine_learning/src/project/backup_result/pruning_KDalpha0.5t2/1986/save/CKPT+epoch_79/model.ckpt"


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


def build_from_checkpoint(state):
    """Infer architecture from checkpoint weight shapes and build matching model."""
    # Normalize: strip 'model.' prefix if present
    if any(k.startswith("model.") for k in state):
        state = {k.replace("model.", "", 1): v for k, v in state.items() if k.startswith("model.")}

    # CNN channel size from first conv weight: shape [out_ch, in_ch, kH, kW]
    cnn1_ch = state["0.cnn1.conv_1.conv.weight"].shape[0]
    cnn2_ch = state["0.cnn2.conv_1.conv.weight"].shape[0]

    # GRU hidden size: weight_hh_l0 shape [3*hidden, hidden]
    gru_hidden = state["1.rnn.weight_hh_l0"].shape[1]
    gru_input  = state["1.rnn.weight_ih_l0"].shape[1]   # cnn2_ch * freq_bins
    rnn_input  = gru_input

    # DNN: check if dnn2 exists
    has_dnn2  = "2.dnn2.linear.w.weight" in state
    dnn1_neur = state["2.dnn1.linear.w.weight"].shape[0]
    dnn_input = state["2.dnn1.linear.w.weight"].shape[1]

    cnn = containers.Sequential(
        input_shape=[None, None, N_MELS],
        norm1=normalization.LayerNorm,
        cnn1=lambda input_shape, c=cnn1_ch: CNN_Block(input_shape, channels=c, kernel_size=(3, 3)),
        cnn2=lambda input_shape, c=cnn2_ch: CNN_Block(input_shape, channels=c, kernel_size=(3, 3)),
    )
    rnn = RNN.GRU(
        input_shape=[None, None, rnn_input],
        hidden_size=gru_hidden,
        num_layers=2,
        bidirectional=True,
    )
    if has_dnn2:
        dnn2_neur = state["2.dnn2.linear.w.weight"].shape[0]
        dnn = containers.Sequential(
            input_shape=[None, None, dnn_input],
            dnn1=lambda input_shape, n=dnn1_neur: DNN_Block(input_shape, neurons=n),
            dnn2=lambda input_shape, n=dnn2_neur: DNN_Block(input_shape, neurons=n),
            lin=lambda input_shape: linear.Linear(input_shape=input_shape, n_neurons=1, bias=False),
        )
    else:
        dnn = containers.Sequential(
            input_shape=[None, None, dnn_input],
            dnn1=lambda input_shape, n=dnn1_neur: DNN_Block(input_shape, neurons=n),
            lin=lambda input_shape: linear.Linear(input_shape=input_shape, n_neurons=1, bias=False),
        )
    return nn.ModuleList([cnn, rnn, dnn])

def build_pruned():
    cnn = containers.Sequential(
        input_shape=[None, None, N_MELS],
        norm1=normalization.LayerNorm,
        cnn1=lambda input_shape: CNN_Block(input_shape, channels=16, kernel_size=(3, 3)),
        cnn2=lambda input_shape: CNN_Block(input_shape, channels=32, kernel_size=(3, 3)),
    )
    rnn = RNN.GRU(input_shape=[None, None, 320], hidden_size=22, num_layers=2, bidirectional=True)
    dnn = containers.Sequential(
        input_shape=[None, None, 44],
        dnn1=lambda input_shape: DNN_Block(input_shape, neurons=16),
        dnn2=lambda input_shape: DNN_Block(input_shape, neurons=16),
        lin=lambda input_shape: linear.Linear(input_shape=input_shape, n_neurons=1, bias=False),
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

def load_model(model, ckpt_path, strip_prefix=False):
    if not ckpt_path or not os.path.exists(ckpt_path):
        return False
    state = torch.load(ckpt_path, map_location="cpu")
    if strip_prefix:
        state = {k.replace("model.", "", 1): v
                 for k, v in state.items() if k.startswith("model.")}
    model.load_state_dict(state)
    return True

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device("cpu")

    print("Preparing input features...")
    wavs = torch.randn(1, SAMPLE_RATE * EXAMPLE_LENGTH)
    fbank      = Fbank(sample_rate=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS, hop_length=HOP_LENGTH_MS)
    normalizer = InputNormalization(norm_type="sentence")
    with torch.no_grad():
        feats = normalizer(fbank(wavs), torch.ones(1)).detach()
    print(f"Input shape: {feats.shape}\n")

    print("Building models...")
    teacher = build_teacher().to(device)
    student = build_student().to(device)

    def load_auto(ckpt_path, label):
        """Load checkpoint, auto-detect architecture, return (model, loaded)."""
        if not ckpt_path or not os.path.exists(ckpt_path):
            print(f"  {label}: checkpoint not found, using random weights")
            return build_pruned().to(device), False
        raw = torch.load(ckpt_path, map_location="cpu")
        # Strip 'model.' prefix if present so keys are 0.xxx / 1.xxx / 2.xxx
        if any(k.startswith("model.") for k in raw):
            state = {k.replace("model.", "", 1): v for k, v in raw.items() if k.startswith("model.")}
        else:
            state = raw
        model = build_from_checkpoint(state).to(device)
        model.load_state_dict(state)
        print(f"  {label}: loaded from {ckpt_path}")
        return model, True

    pruned    = load_auto(args.pruned_ckpt,    "Pruned")[0]
    pruned_kd = load_auto(args.pruned_kd_ckpt, "Pruned+KD")[0]

    for name, model, ckpt in [
        ("Teacher", teacher, args.teacher_ckpt),
        ("Student", student, args.student_ckpt),
    ]:
        if ckpt and os.path.exists(ckpt):
            state = torch.load(ckpt, map_location="cpu")
            model.load_state_dict(state)
            print(f"  {name}: loaded from {ckpt}")
        else:
            print(f"  {name}: checkpoint not found, using random weights")
    print()

    print(f"Benchmarking ({args.n_warmup} warmup + {args.n_runs} timed runs, CPU)...\n")
    teacher_times   = benchmark(teacher,   feats, args.n_warmup, args.n_runs)
    student_times   = benchmark(student,   feats, args.n_warmup, args.n_runs)
    pruned_times    = benchmark(pruned,    feats, args.n_warmup, args.n_runs)
    pruned_kd_times = benchmark(pruned_kd, feats, args.n_warmup, args.n_runs)

    def stats(times):
        t = torch.tensor(times)
        return {
            "mean_ms": t.mean().item() * 1000,
            "std_ms":  t.std().item()  * 1000,
            "min_ms":  t.min().item()  * 1000,
            "max_ms":  t.max().item()  * 1000,
            "rtf":     t.mean().item() / EXAMPLE_LENGTH,
        }

    ts  = stats(teacher_times)
    ss  = stats(student_times)
    ps  = stats(pruned_times)
    pks = stats(pruned_kd_times)

    t_params  = count_parameters(teacher)
    s_params  = count_parameters(student)
    p_params  = count_parameters(pruned)
    pk_params = count_parameters(pruned_kd)
    t_mb  = model_size_mb(teacher)
    s_mb  = model_size_mb(student)
    p_mb  = model_size_mb(pruned)
    pk_mb = model_size_mb(pruned_kd)

    sep = "─" * 90
    print(sep)
    print(f"{'':30s} {'Teacher':>14s} {'Student(KD)':>14s} {'Pruned':>14s} {'Pruned+KD':>14s}")
    print(sep)
    print(f"{'Parameters':30s} {t_params:>14,d} {s_params:>14,d} {p_params:>14,d} {pk_params:>14,d}")
    print(f"{'Model size (MB)':30s} {t_mb:>14.3f} {s_mb:>14.3f} {p_mb:>14.3f} {pk_mb:>14.3f}")
    print(sep)
    print(f"{'Mean latency (ms)':30s} {ts['mean_ms']:>14.2f} {ss['mean_ms']:>14.2f} {ps['mean_ms']:>14.2f} {pks['mean_ms']:>14.2f}")
    print(f"{'Std  latency (ms)':30s} {ts['std_ms']:>14.2f} {ss['std_ms']:>14.2f} {ps['std_ms']:>14.2f} {pks['std_ms']:>14.2f}")
    print(f"{'Min  latency (ms)':30s} {ts['min_ms']:>14.2f} {ss['min_ms']:>14.2f} {ps['min_ms']:>14.2f} {pks['min_ms']:>14.2f}")
    print(f"{'Max  latency (ms)':30s} {ts['max_ms']:>14.2f} {ss['max_ms']:>14.2f} {ps['max_ms']:>14.2f} {pks['max_ms']:>14.2f}")
    print(sep)
    print(f"{'Real-time factor (RTF)':30s} {ts['rtf']:>14.4f} {ss['rtf']:>14.4f} {ps['rtf']:>14.4f} {pks['rtf']:>14.4f}")
    print(sep)
    print()
    print(f"  vs Teacher — Student speedup:    {ts['mean_ms']/ss['mean_ms']:.2f}×   params: {t_params/s_params:.2f}×   size: {t_mb/s_mb:.2f}×")
    print(f"  vs Teacher — Pruned speedup:     {ts['mean_ms']/ps['mean_ms']:.2f}×   params: {t_params/p_params:.2f}×   size: {t_mb/p_mb:.2f}×")
    print(f"  vs Teacher — Pruned+KD speedup:  {ts['mean_ms']/pks['mean_ms']:.2f}×   params: {t_params/pk_params:.2f}×   size: {t_mb/pk_mb:.2f}×")
    print()
    print("RTF interpretation:")
    for name, rtf in [("Teacher", ts["rtf"]), ("Student(KD)", ss["rtf"]),
                      ("Pruned", ps["rtf"]), ("Pruned+KD", pks["rtf"])]:
        status = "real-time capable" if rtf < 1.0 else "too slow for real-time"
        print(f"  {name}: RTF={rtf:.4f}  → {status}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_ckpt", default=DEFAULT_TEACHER_CKPT)
    parser.add_argument("--student_ckpt", default=DEFAULT_STUDENT_CKPT)
    parser.add_argument("--pruned_ckpt",    default=DEFAULT_PRUNED_CKPT,
                        help="Path to pruned model checkpoint")
    parser.add_argument("--pruned_kd_ckpt", default=DEFAULT_PRUNED_KD_CKPT,
                        help="Path to pruned+KD model checkpoint")
    parser.add_argument("--n_warmup", type=int, default=100)
    parser.add_argument("--n_runs",   type=int, default=2000)
    args = parser.parse_args()
    main(args)