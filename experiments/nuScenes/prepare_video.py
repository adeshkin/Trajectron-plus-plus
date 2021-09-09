import cv2
import os
from tqdm import tqdm

fourcc = cv2.VideoWriter_fourcc(*'XVID')


def main():
    img_width = 1000
    img_height = 1000
    fps = 2
    model_name = 'int_ee'
    scene_idx = 99
    ph = 6
    num_samples = 100

    video_path = f"videos/waymo_ml_{model_name}_sc_{scene_idx}_ns_{num_samples}_ph_{ph}_bev.avi"
    data_dir = f"visualizations/waymo/model_{model_name}_scene_{scene_idx}_ns_{num_samples}_ph_{ph}/bev_maps"

    out = cv2.VideoWriter(video_path, fourcc, fps, (img_width, img_height), True)

    filepaths = sorted([os.path.join(data_dir, x) for x in os.listdir(data_dir)], key=lambda x: x.split('/')[-1])

    for path in tqdm(filepaths):
        if os.path.isfile(path):
            frame = cv2.imread(path)
            out.write(frame)


if __name__ == '__main__':
    main()
