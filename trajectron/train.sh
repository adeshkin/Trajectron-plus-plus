#!/bin/bash

python train.py \
--eval_every 1 \
--vis_every 1 \
--conf ../experiments/nuScenes/models/int_ee_old/config_ph_25_maxhl_25_minhl_24.json \
--data_dir /media/cds-k/data/nuScenes/traj++_processed_data/processed_sdc \
--train_data_dict sdc_train.pkl \
--eval_data_dict sdc_validation.pkl \
--offline_scene_graph no \
--preprocess_workers 10 \
--batch_size 256 \
--log_dir ../experiments/nuScenes/models \
--train_epochs 100 \
--node_freq_mult_train \
--log_tag _int_ee_sdc_ph_25_maxhl_25_min_hl_24
