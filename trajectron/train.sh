#!/bin/bash

python train.py \
--eval_every 1 \
--conf ../experiments/nuScenes/train_configs/config_ph_10_maxhl_4_minhl_0.json \
--data_dir /media/cds-k/data/nuScenes/traj++_processed_data/processed_boris_new_no_kalman \
--train_data_dict nuScenes_train_full.pkl \
--eval_data_dict nuScenes_val_full.pkl \
--offline_scene_graph yes \
--preprocess_workers 10 \
--batch_size 256 \
--log_dir ../experiments/nuScenes/models \
--train_epochs 20 \
--log_tag _int_ee_ph_10_maxhl_4_minhl_0 \
--node_freq_mult_train \
--augment

