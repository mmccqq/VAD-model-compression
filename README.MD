## Folder:

backup_results: trained models

pretrained_models: the original CRDNN trained by SpeechBrain.

VAD-model-compression/VAD_kd: use Knowledge Distillation to compress a model

VAD-model-compression/VAD-Pruning-GRU-Layer: use pruning technique to compress a model

## Files:

energy_VAD_validation: evaluate the performance of the energy_VAD, the baseline.

eval_teacher: evaluate the performance of the original CRDNN and the pruned CRDNN model.

measure_speed.py: measure the inference speed of a model.

VAD-model-compression/VAD_kd: 

1. train.py train a CRDNN from scrath
2. train_kd.py, kd the original CRDNN to a student model
3. train_pruning_kd.py, kd the pruned CRDNN to a student model

to run a training:

  python train_pruning_kd.py hparams/train_pruning_kd.yaml \
  --data_folder=/path/to/LibriParty/dataset \
  --musan_folder=/path/to/voice_data/musan \
  --commonlanguage_folder=/path/to/common_voice_kpd \
  --teacher_ckpt_dir=/path/to/teacher_model