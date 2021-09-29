#!/bin/bash

for mode in 'train' 'validation'
do
  python process_sdc.py \
    --data=/home/adeshkin/Desktop/sdc_data \
    --version=$mode \
    --output_path=../../processed_sdc
done