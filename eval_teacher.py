#!/usr/bin/env python3
"""
Evaluate the pretrained teacher CRDNN VAD model on the LibriParty test set.
Reports: F-score, FAR, FRR, DER, Precision, Recall
"""

import json
import numpy as np
import torch

import speechbrain as sb
from speechbrain.lobes.features import Fbank
from speechbrain.lobes.models.CRDNN import CNN_Block, DNN_Block
from speechbrain.nnet import RNN, containers, linear, normalization
from speechbrain.processing.features import InputNormalization
from speechbrain.utils.metric_stats import BinaryMetricStats

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_CKPT   = "/Users/xiaoluochun/Documents/NEU/6140_machine_learning/src/pretrained_models/vad-crdnn-libriparty/model.ckpt"
NORM_CKPT    = "/Users/xiaoluochun/Documents/NEU/6140_machine_learning/src/pretrained_models/vad-crdnn-libriparty/mean_var_norm.ckpt"
TEST_JSON    = "/Users/xiaoluochun/Documents/NEU/6140_machine_learning/src/project/VAD-model-compression/VAD_kd/results/VAD_CRDNN_KD/1986/save/test.json"
DATA_ROOT    = "/Users/xiaoluochun/Documents/NEU/6140_machine_learning/voice_dataset/LibriParty/dataset"

# ── Hparams (from pretrained hyperparams.yaml) ────────────────────────────────
SAMPLE_RATE     = 16000
N_FFT           = 400
N_MELS          = 40
HOP_LENGTH_MS   = 10.0
EXAMPLE_LENGTH  = 5
TIME_RESOLUTION = 0.01


# ── Build teacher model ───────────────────────────────────────────────────────
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
    return torch.nn.ModuleList([cnn, rnn, dnn])

# ── Build pruning teacher model ───────────────────────────────────────────────────────
def build_pruned():
    cnn = containers.Sequential(
        input_shape=[None, None, N_MELS],
        norm1=normalization.LayerNorm,
        cnn1=lambda input_shape: CNN_Block(input_shape, channels=16, kernel_size=(3, 3)),
        cnn2=lambda input_shape: CNN_Block(input_shape, channels=32, kernel_size=(3, 3)),
    )
    rnn = RNN.GRU(
        input_shape=[None, None, 320],
        hidden_size=22,        # pruned from 32
        num_layers=2,
        bidirectional=True,
    )
    dnn = containers.Sequential(
        input_shape=[None, None, 44],   # pruned from 64
        dnn1=lambda input_shape: DNN_Block(input_shape, neurons=16),
        dnn2=lambda input_shape: DNN_Block(input_shape, neurons=16),
        lin=lambda input_shape: linear.Linear(input_shape=input_shape,
                                              n_neurons=1, bias=False),
    )
    return torch.nn.ModuleList([cnn, rnn, dnn])

# ── Inference ─────────────────────────────────────────────────────────────────
def run_model(model, feats):
    cnn, rnn, dnn = model[0], model[1], model[2]
    out = cnn(feats)
    out = out.reshape(out.shape[0], out.shape[1], out.shape[2] * out.shape[3])
    out, _ = rnn(out)
    logit = dnn(out)                      # [1, T, 1]
    return torch.sigmoid(logit[0, :, 0])  # [T]  probabilities


# ── Target builder ────────────────────────────────────────────────────────────
def get_target(speech_intervals, n_frames):
    gt = torch.zeros(n_frames)
    for start_s, end_s in speech_intervals:
        start_f = int(start_s / TIME_RESOLUTION)
        end_f   = int(end_s   / TIME_RESOLUTION)
        gt[start_f:end_f] = 1
    return gt


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cpu")

    # Ask user which model to evaluate
    print("Select model to evaluate:")
    print("  1. Normal model (original CRDNN teacher)")
    print("  2. Pruning model (pruned CRDNN, GRU hidden=22)")
    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        ckpt_path  = MODEL_CKPT
        model      = build_teacher().to(device)
        label      = "Normal CRDNN (Teacher)"
        strip_prefix = False
    elif choice == "2":
        ckpt_path  = input("Enter path to pruned model .pt file: ").strip()
        model      = build_pruned().to(device)
        label      = "Pruned CRDNN"
        strip_prefix = True
    else:
        print("Invalid choice. Enter 1 or 2.")
        exit(1)

    # Build and load model
    print(f"\nLoading {label}...")
    full_state = torch.load(ckpt_path, map_location="cpu")
    if strip_prefix:
        state = {
            k.replace("model.", "", 1): v
            for k, v in full_state.items()
            if k.startswith("model.")
        }
    else:
        state = full_state
    model.load_state_dict(state)
    model.eval()
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Build and load feature extractor + normalizer
    fbank = Fbank(sample_rate=SAMPLE_RATE, n_fft=N_FFT,
                  n_mels=N_MELS, hop_length=HOP_LENGTH_MS)
    normalizer = InputNormalization(norm_type="sentence")
    # norm_state = torch.load(NORM_CKPT, map_location="cpu")
    # normalizer.load_state_dict(norm_state)
    normalizer.eval()

    # Load test manifest
    print("Loading test manifest...")
    with open(TEST_JSON) as f:
        test_data = json.load(f)
    print(f"  Test examples: {len(test_data)}")

    # Evaluate
    print("Running inference...")
    metrics  = BinaryMetricStats()
    n_frames = int(np.ceil(EXAMPLE_LENGTH / TIME_RESOLUTION))

    with torch.no_grad():
        for utt_id, entry in test_data.items():
            wav_path = entry["wav"]["file"].replace("{data_root}", DATA_ROOT)
            start    = entry["wav"].get("start", 0)
            stop     = entry["wav"].get("stop",  None)

            # Load audio
            wav = sb.dataio.dataio.read_audio(
                {"file": wav_path, "start": start, "stop": stop}
            ).unsqueeze(0)                            # [1, T_samples]

            # Feature extraction
            feats = fbank(wav)                        # [1, T, 40]
            feats = normalizer(feats, torch.ones(1))  # [1, T, 40]

            # Inference
            prob = run_model(model, feats)[:n_frames]  # [T]

            # Pad if needed
            if prob.shape[0] < n_frames:
                prob = torch.nn.functional.pad(
                    prob, (0, n_frames - prob.shape[0])
                )

            # Ground truth
            target = get_target(entry.get("speech", []), n_frames)  # [T]

            metrics.append(
                ids=[utt_id],
                scores=prob.unsqueeze(0),    # [1, T]
                labels=target.unsqueeze(0),  # [1, T]
            )

    # Results
    summary = metrics.summarize(threshold=0.5)
    sep = "─" * 40
    print(f"\n{sep}")
    print(f"{label} — LibriParty test set")
    print(sep)
    print(f"  F-score:   {summary['F-score']:.4f}")
    print(f"  DER:       {summary['DER']:.4f}")
    print(f"  FAR:       {summary['FAR']:.4f}")
    print(f"  FRR:       {summary['FRR']:.4f}")
    print(f"  Precision: {summary['precision']:.4f}")
    print(f"  Recall:    {summary['recall']:.4f}")
    print(f"  MCC:       {summary['MCC']:.4f}")
    print(sep)
