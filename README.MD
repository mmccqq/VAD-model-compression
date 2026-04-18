Folder:
backup_results: trained models
pretrained_models: the original CRDNN trained by SpeechBrain.
VAD-model-compression/VAD_kd: use Knowledge Distillation to compress a model
VAD-model-compression/VAD-Pruning-GRU-Layer: use pruning technique to compress a model

Files:
energy_VAD_validation: evaluate the performance of the energy_VAD, the baseline.
eval_teacher: evaluate the performance of the original CRDNN and the pruned CRDNN model.
measure_speed.py: measure the inference speed of a model.

1. train.py train a CRDNN from scrath
2. train_kd.py, kd the original CRDNN to a student model
3. train_pruning_kd.py, kd the pruned CRDNN to a student model
4. 