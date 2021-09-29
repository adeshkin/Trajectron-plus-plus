#!/bin/bash

for mode in 'train' 'val' 'test'
do
  python process_sdc.py \
    --data=/home/adeshkin/Desktop/sdc_data \
    --version="validation" \
    --output_path=../../processed_sdc
done