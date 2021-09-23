import cv2
import os
from tqdm import tqdm

fourcc = cv2.VideoWriter_fourcc(*'XVID')


def main():
    img_width = 1024
    img_height = 1130
    fps = 2
    model_name = 'int_ee'
    scene_idx = 99
    ph = 6
    num_samples = 100

    video_path = f"./stone_flower_add_2021-08-20-11-33-48_4.avi"
    data_dir = f"../visualizations/centerpoint_out_no_minus_50_dur_180s/stone_flower_add_2021-08-20-11-33-48_4/model_int_ee_ph_10_maxhl_4_scene_0_ns_1_ph_10_max_h_4_Z_GMM/bev_maps"

    out = cv2.VideoWriter(video_path, fourcc, fps, (img_width, img_height), True)

    filepaths = sorted([os.path.join(data_dir, x) for x in os.listdir(data_dir)], key=lambda x: x.split('/')[-1])

    for path in tqdm(filepaths):
        if os.path.isfile(path):
            frame = cv2.imread(path)
            out.write(frame)


if __name__ == '__main__':
    main()
