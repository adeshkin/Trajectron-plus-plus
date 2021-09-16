#!/bin/bash

python train.py \
--eval_every 1 \
--vis_every 1 \
--conf ../experiments/nuScenes/models/vel_ee/config.json \
--data_dir ../experiments/processed_new \
--train_data_dict nuScenes_train_full.pkl \
--eval_data_dict nuScenes_val_full.pkl \
--offline_scene_graph yes \
--preprocess_workers 10 \
--batch_size 256 \
--log_dir ../experiments/nuScenes/models \
--train_epochs 20 \
--node_freq_mult_train \
--log_tag _vel_ee_new \
--augment
