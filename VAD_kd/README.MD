download voice data set.

update the path:
python train_kd.py hparams/train_kd.yaml \
    --data_folder=./voice_data/LibriParty/dataset \
    --musan_folder=./voice_data/musan \
    --commonlanguage_folder=./voice_data/common_voice_kpd \
    --teacher_ckpt_dir=results/VAD_CRDNN/1986/save/CKPT+epoch_85 \
    --skip_prep=True \
    --annotation_train=results/VAD_CRDNN/1986/save/train.json \
    --annotation_valid=results/VAD_CRDNN/1986/save/valid.json \
    --annotation_test=results/VAD_CRDNN/1986/save/test.json \
    --music_csv=results/VAD_CRDNN/1986/save/music.csv \
    --noise_csv=results/VAD_CRDNN/1986/save/noise.csv \
    --speech_csv=results/VAD_CRDNN/1986/save/speech.csv \
    --noise_csv_openrir=results/VAD_CRDNN/1986/save/noise_openrir.csv \
    --device=cuda