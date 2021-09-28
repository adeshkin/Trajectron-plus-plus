#!/bin/bash

python train.py \
--eval_every 1 \
--vis_every 1 \
--conf ../experiments/nuScenes/models/int_ee_old/config_ph_10_maxhl_4.json \
--data_dir /media/cds-k/data/nuScenes/traj++_processed_data/processed_centerpoint_out_augment \
--train_data_dict centerpoint_out_train.pkl \
--eval_data_dict centerpoint_out_val.pkl \
--offline_scene_graph yes \
--preprocess_workers 10 \
--batch_size 256 \
--log_dir ../experiments/nuScenes/models \
--train_epochs 20 \
--node_freq_mult_train \
--log_tag _int_ee_centerpoint_out_ph_10_maxhl_4 \
--augment
