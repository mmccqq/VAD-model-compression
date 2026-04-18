import torch
import numpy as np
import speechbrain as sb
from speechbrain.utils.metric_stats import BinaryMetricStats
import json

# ── Parameters (must match your model's settings) ─────────────────────────────
SAMPLE_RATE     = 16000
FRAME_SIZE      = int(0.01 * SAMPLE_RATE)   # 10ms = 160 samples
EXAMPLE_LENGTH  = 5
TIME_RESOLUTION = 0.01
THRESHOLD_DB    = -40.0   # dB — tune this for best performance

# ── Load test manifest ────────────────────────────────────────────────────────
TEST_JSON = "/Users/xiaoluochun/Documents/NEU/6140_machine_learning/src/project/VAD-model-compression/VAD_kd/results/VAD_CRDNN_KD/1986/save/test.json"
DATA_ROOT = "/Users/xiaoluochun/Documents/NEU/6140_machine_learning/voice_dataset/LibriParty/dataset"

with open(TEST_JSON) as f:
    test_data = json.load(f)

# ── Energy VAD ────────────────────────────────────────────────────────────────
def energy_vad(wav, frame_size, threshold_db):
    """
    Compute frame-level energy and apply threshold.
    Returns binary prediction [T] where 1=speech, 0=silence.
    """
    n_frames = wav.shape[0] // frame_size
    wav = wav[:n_frames * frame_size].reshape(n_frames, frame_size)

    # RMS energy per frame
    rms = wav.pow(2).mean(dim=1).sqrt()
    # Prevents log(0) in the next line by ensuring RMS is never exactly zero.
    rms = torch.clamp(rms, min=1e-10)
    energy_db = 20 * torch.log10(rms)

    return (energy_db > threshold_db).float()   # [T]


def get_target(speech_intervals, n_frames):
    """Convert speech intervals (seconds) to binary frame labels."""
    gt = torch.zeros(n_frames)
    for start_s, end_s in speech_intervals:
        start_f = int(start_s / TIME_RESOLUTION)
        end_f   = int(end_s   / TIME_RESOLUTION)
        gt[start_f:end_f] = 1
    return gt

for thresh in [-60, -50, -40, -35, -30, -25, -20]:
  THRESHOLD_DB = thresh
# ── Evaluate over test set ────────────────────────────────────────────────────
  metrics = BinaryMetricStats()

  for utt_id, entry in test_data.items():
      wav_path = entry["wav"]["file"].replace("{data_root}", DATA_ROOT)
      start    = entry["wav"].get("start", 0)
      stop     = entry["wav"].get("stop",  None)

      wav = sb.dataio.dataio.read_audio(
          {"file": wav_path, "start": start, "stop": stop}
      )                                                 # [T_samples]

      n_frames = int(np.ceil(EXAMPLE_LENGTH / TIME_RESOLUTION))
      pred     = energy_vad(wav, FRAME_SIZE, THRESHOLD_DB)[:n_frames]  # [T]
      target   = get_target(entry.get("speech", []), n_frames)         # [T]

      # Pad if needed
      if pred.shape[0] < n_frames:
          pred = torch.nn.functional.pad(pred, (0, n_frames - pred.shape[0]))

      metrics.append(
          ids=[utt_id],
          scores=pred.unsqueeze(0),     # [1, T]
          labels=target.unsqueeze(0),   # [1, T]
      )

  summary = metrics.summarize(threshold=0.5)
  print(f"Energy VAD (threshold={THRESHOLD_DB} dB)")
  print(f"  F-score:   {summary['F-score']:.4f}")
  print(f"  FAR:       {summary['FAR']:.4f}")
  print(f"  FRR:       {summary['FRR']:.4f}")
  print(f"  DER:       {summary['DER']:.4f}")
  print(f"  Precision: {summary['precision']:.4f}")
  print(f"  Recall:    {summary['recall']:.4f}")
  print(f"  MCC:    {summary['MCC']:.4f}")
