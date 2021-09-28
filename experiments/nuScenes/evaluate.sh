#!/bin/bash

#'models/vel_ee_old' 'models/int_ee_old' 'models/my_int_ee_old' 'models/my_int_ee' 'models/my_int_ee_new'
for model in 'models/int_ee_new_no_kalman_ph_10_maxhl_4'
do
    data_dir=/media/cds-k/data/nuScenes/traj++_processed_data/processed_boris/nuScenes_test_full.pkl

    if [[ $model == *"_old" ]]; then
      data_dir=/media/cds-k/data/nuScenes/traj++_processed_data/processed_boris_old/nuScenes_test_full.pkl
    fi

    if [[ $model == *"_new" ]]; then
      data_dir=/media/cds-k/data/nuScenes/traj++_processed_data/processed_boris_new/nuScenes_test_full.pkl
    fi

    model_name=$(basename $model)
    python evaluate.py \
      --model=$model \
      --checkpoint=12 \
      --data=$data_dir \
      --output_path=/media/cds-k/data/nuScenes/traj++_results_ped/$model_name \
      --output_tag=$model_name
done

