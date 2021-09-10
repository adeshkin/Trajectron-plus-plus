#!/bin/bash

python process_waymo.py \
	--data=/media/cds-k/Data_2/waymo_motion/validation/validation_tfexample.tfrecord-00000-of-00150 \
	--version="val_old" \
	--output_path=../../processed_waymo