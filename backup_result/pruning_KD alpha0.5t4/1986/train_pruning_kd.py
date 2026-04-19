#!/usr/bin/env python3
"""
Knowledge-distillation training for a smaller CRDNN VAD student model.

Teacher  : CNN(16,32) + biGRU(hidden=32,layers=2) + DNN(16,16)  -- frozen
Student  : CNN( 8,16) + biGRU(hidden=16,layers=2) + DNN( 8   )  -- trained

KD loss  : alpha * MSE(sigmoid(s_logit/T), sigmoid(t_logit/T))
         + (1-alpha) * BCE(s_logit, hard_label)

Usage:
    python train_kd.py hparams/train_kd.yaml \
        --data_folder=/path/to/LibriParty \
        --musan_folder=/path/to/musan \
        --commonlanguage_folder=/path/to/common_voice_kpd \
        --teacher_ckpt_dir=/path/to/CKPT+epoch_85
"""

import sys

import numpy as np
import torch
import torch.nn.functional as F
from data_augment import augment_data
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.utils.distributed import run_on_main
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Helper: load teacher weights and freeze all parameters
# ──────────────────────────────────────────────────────────────────────────────

def load_and_freeze_teacher(hparams):
    """Load the teacher checkpoint and freeze its weights."""
    # Load the teacher architecture objects from the yaml with random weights at this point
    teacher_model = hparams["teacher_model"]
    teacher_cnn   = hparams["teacher_cnn"]
    teacher_rnn   = hparams["teacher_rnn"]
    teacher_dnn   = hparams["teacher_dnn"]


    ckpt_path = hparams["teacher_ckpt_dir"]  # now points to .pt file

    # # Creates a SpeechBrain Checkpointer pointing at the model folder and use recover_if_possible() to load the checkpoint if found.
    # # ckpt_dir = hparams["teacher_ckpt_dir"]
    # teacher_checkpointer = sb.utils.checkpoints.Checkpointer(
    #     checkpoints_dir=ckpt_dir,
    #     recoverables={"model": teacher_model},
    # )
    # # Load exactly this checkpoint folder (not "best" — the folder IS the ckpt)
    # teacher_checkpointer.recover_if_possible()


    # Load raw state dict — keys look like "model.0.*", "model.1.*", "model.2.*"
    full_state = torch.load(ckpt_path, map_location="cpu")
    # Strip the "model." prefix to match ModuleList keys "0.*", "1.*", "2.*"
    model_state = {
        k.replace("model.", "", 1): v
        for k, v in full_state.items()
        if k.startswith("model.")
    }
    teacher_model.load_state_dict(model_state)




    # Freeze every teacher parameter
    for param in teacher_model.parameters():
        param.requires_grad = False
    # Switch the model to eval mode which disables dropout and fixes batchnorm)
    teacher_model.eval()

    #logger.info("Teacher loaded from %s and frozen.", ckpt_dir)
    logger.info("Teacher loaded from %s and frozen.", ckpt_path)
    return teacher_cnn, teacher_rnn, teacher_dnn


# ──────────────────────────────────────────────────────────────────────────────
# KD Brain
# ──────────────────────────────────────────────────────────────────────────────

class KDVADBrain(sb.Brain):
    """VAD brain that trains the student with knowledge distillation."""

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def compute_forward(self, batch, stage):
        """Run augmentation, then student forward. In TRAIN also run teacher."""
        # Same data augmentation as the baseline model
        batch = batch.to(self.device)
        wavs, lens = batch.signal
        targets, lens_targ = batch.target
        self.targets = targets

        if stage == sb.Stage.TRAIN:
            wavs, targets, lens = augment_data(
                self.noise_datasets,
                self.speech_datasets,
                wavs,
                targets,
                lens_targ,
            )
            self.lens    = lens
            self.targets = targets

        # Shared feature extraction (detached — not part of any graph)
        # Same as the baseline model
        feats = self.hparams.compute_features(wavs)
        feats = self.modules.mean_var_norm(feats, lens)
        feats = feats.detach()

        # ── Student forward ──
        s_out = self.modules.student_cnn(feats)
        # Flattens the last two dimensions into a single feature vector per frame
        s_out = s_out.reshape(s_out.shape[0], s_out.shape[1],
                              s_out.shape[2] * s_out.shape[3])
        s_out, _ = self.modules.student_rnn(s_out)
        s_logit   = self.modules.student_dnn(s_out)   # [B, T, 1]

        # ── Teacher forward (no gradient, only during training) ──
        t_logit = None
        if stage == sb.Stage.TRAIN:
            with torch.no_grad():
                # - Uses the same feats tensor as the student — teacher and student see identical input features
                t_out = self.teacher_cnn(feats)
                t_out = t_out.reshape(t_out.shape[0], t_out.shape[1],
                                      t_out.shape[2] * t_out.shape[3])
                t_out, _ = self.teacher_rnn(t_out)
                t_logit   = self.teacher_dnn(t_out)   # [B, T, 1]

        return s_logit, t_logit, lens

    # ------------------------------------------------------------------
    # Objectives
    # ------------------------------------------------------------------

    def compute_objectives(self, predictions, batch, stage):
        """alpha * KD_loss + (1-alpha) * BCE_loss."""
        s_logit, t_logit, lens = predictions
        targets = self.targets

        # Trim time axis to match target length
        T = targets.shape[-1]
        s_logit_2d = s_logit[:, :T, 0]   # [B, T]

        # Hard-label BCE loss
        bce_loss = self.hparams.compute_BCE_cost(s_logit_2d, targets, lens)

        # KD loss (only during training, where teacher output is available)
        if stage == sb.Stage.TRAIN and t_logit is not None:
            alpha = self.hparams.kd_alpha
            T_temp = self.hparams.kd_temperature

            t_logit_2d = t_logit[:, :T, 0]   # [B, T]

            # Soften with temperature, then compare probabilities via MSE
            s_prob = torch.sigmoid(s_logit_2d / T_temp)
            t_prob = torch.sigmoid(t_logit_2d / T_temp)
            # Scale by T^2 to keep gradient magnitudes comparable (Hinton et al.)
            kd_loss = F.mse_loss(s_prob, t_prob) * (T_temp ** 2)

            loss = alpha * kd_loss + (1.0 - alpha) * bce_loss
        else:
            loss = bce_loss

        # Metrics
        # Same as the baseline model
        self.train_metrics.append(
            batch.id, torch.sigmoid(s_logit_2d), targets
        )
        if stage != sb.Stage.TRAIN:
            self.valid_metrics.append(
                batch.id, torch.sigmoid(s_logit_2d), targets
            )

        return loss

    # ------------------------------------------------------------------
    # Stage hooks
    # ------------------------------------------------------------------

    def on_stage_start(self, stage, epoch=None):
        "Gets called when a stage (either training, validation, test) starts."
        # Same as the baseline model
        self.train_metrics = self.hparams.train_stats()

        self.noise_datasets = [
            self.hparams.add_noise,
            self.hparams.add_noise_musan,
            self.hparams.add_music_musan,
        ]
        self.speech_datasets = [
            self.hparams.add_speech_musan,
            self.hparams.add_speech_musan,
            self.hparams.add_speech_musan,
        ]

        if stage != sb.Stage.TRAIN:
            self.valid_metrics = self.hparams.test_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of a stage."""
        # Same as the baseline model
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss

        else:
            summary = self.valid_metrics.summarize(threshold=0.5)

        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(epoch)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats={"loss": self.train_loss},
                valid_stats={"loss": stage_loss, "summary": summary},
            )
            self.checkpointer.save_and_keep_only(
                meta={"loss": stage_loss, "summary": summary},
                num_to_keep=1,
                min_keys=["loss"],
                name=f"epoch_{epoch}",
            )

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats={"loss": stage_loss, "summary": summary},
            )


# ──────────────────────────────────────────────────────────────────────────────
# Data pipeline 
# ──────────────────────────────────────────────────────────────────────────────
# identical to original train.py
def dataio_prep(hparams):
    "Creates the datasets and their data processing pipelines."

    # 1. Declarations:
    data_folder = hparams["data_folder"]
    train = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["annotation_train"],
        replacements={"data_root": data_folder},
    )
    validation = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["annotation_valid"],
        replacements={"data_root": data_folder},
    )
    test = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["annotation_test"],
        replacements={"data_root": data_folder},
    )
    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("signal")
    def audio_pipeline(wav):
        return sb.dataio.dataio.read_audio(wav)
    
    # 3. Define text pipeline
    @sb.utils.data_pipeline.takes("speech")
    @sb.utils.data_pipeline.provides("target")
    def vad_targets(speech, hparams=hparams):
        boundaries = (
            [
                (
                    int(interval[0] / hparams["time_resolution"]),
                    int(interval[1] / hparams["time_resolution"]),
                )
                for interval in speech
            ]
            if len(speech) > 0
            else []
        )
        gt = torch.zeros(
            int(np.ceil(hparams["example_length"] * (1 / hparams["time_resolution"])))
        )
        for start, stop in boundaries:
            gt[start:stop] = 1
        return gt
    
    # Create dataset
    datasets = [train, validation, test]
    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)
    sb.dataio.dataset.add_dynamic_item(datasets, vad_targets)
    sb.dataio.dataset.set_output_keys(datasets, ["id", "signal", "target", "speech"])

    return train, validation, test


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    
    # Load hyperparameters file with command-line overrides
    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # ── Data preparation ──────────────────────────────────────────────
    from libriparty_prepare import prepare_libriparty

    # LibriParty preparation
    run_on_main(
        prepare_libriparty,
        kwargs={
            "data_folder":      hparams["data_folder"],
            "save_json_folder": hparams["save_folder"],
            "sample_rate":      hparams["sample_rate"],
            "window_size":      hparams["example_length"],
            "skip_prep":        hparams["skip_prep"],
        },
    )

    # Prepare openrir
    run_on_main(hparams["prepare_noise_data"])

    # Prepare Musan
    from musan_prepare import prepare_musan

    if not hparams["skip_prep"]:
        run_on_main(
            prepare_musan,
            kwargs={
                "folder":        hparams["musan_folder"],
                "music_csv":     hparams["music_csv"],
                "noise_csv":     hparams["noise_csv"],
                "speech_csv":    hparams["speech_csv"],
                "max_noise_len": hparams["example_length"],
            },
        )

    # Prepare common
    from commonlanguage_prepare import prepare_commonlanguage

    if not hparams["skip_prep"]:
        run_on_main(
            prepare_commonlanguage,
            kwargs={
                "folder":   hparams["commonlanguage_folder"],
                "csv_file": hparams["multilang_speech_csv"],
            },
        )

    # Dataset IO prep: creating Dataset objects
    train_data, valid_data, test_data = dataio_prep(hparams)

    # ── Load and freeze teacher ───────────────────────────────────────
    teacher_cnn, teacher_rnn, teacher_dnn = load_and_freeze_teacher(hparams)

    # ── Build student brain ───────────────────────────────────────────
    kd_brain = KDVADBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Attach frozen teacher modules as plain attributes (not tracked by optimizer)
    kd_brain.teacher_cnn = teacher_cnn.to(kd_brain.device)
    kd_brain.teacher_rnn = teacher_rnn.to(kd_brain.device)
    kd_brain.teacher_dnn = teacher_dnn.to(kd_brain.device)

    # ── Training ──────────────────────────────────────────────────────
    kd_brain.fit(
        kd_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )

    # ── Test ──────────────────────────────────────────────────────────
    kd_brain.evaluate(
        test_data,
        min_key="loss",
        test_loader_kwargs=hparams["test_dataloader_opts"],
    )
