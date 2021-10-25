import cv2
import os
import numpy as np
from tqdm import tqdm

fourcc = cv2.VideoWriter_fourcc(*'XVID')


def main():
    img_width = 583
    img_height = 575
    fps = 0.5
    ph = 6
    num_samples = 100

    #dataset = 'nuscenes_old'
    #model_name1 = 'vel_ee'
    #model_name2 = 'int_ee'
    #scene_idx = 10

    #video_path = f"./{dataset}_ml_{model_name1}_{model_name2}_sc_{scene_idx}_ns_{num_samples}_ph_{ph}_bev.avi"
    #data_dir1 = f"../visualizations/{dataset}/model_{model_name1}_scene_{scene_idx}_ns_{num_samples}_ph_{ph}/bev_maps"
    #data_dir2 = f"../visualizations/{dataset}/model_{model_name2}_scene_{scene_idx}_ns_{num_samples}_ph_{ph}/bev_maps"

    dataset1 = 'waymo_old'
    dataset2 = 'waymo'
    model_name = 'int_ee'
    scene_idx = 10

    video_path = f"/home/cds-k/Desktop/shifts_pred/no_map_vs_map.avi"
    data_dir1 = f"/home/cds-k/Desktop/shifts_pred/no_map"
    data_dir2 = f"/home/cds-k/Desktop/shifts_pred/map"

    out = cv2.VideoWriter(video_path, fourcc, fps, (img_width*2, img_height), True)

    filepaths1 = sorted([os.path.join(data_dir1, x) for x in os.listdir(data_dir1)], key=lambda x: x.split('/')[-1])
    filepaths2 = sorted([os.path.join(data_dir2, x) for x in os.listdir(data_dir2)], key=lambda x: x.split('/')[-1])
    for path1, path2 in tqdm(zip(filepaths1, filepaths2)):
        if os.path.isfile(path1) and os.path.isfile(path2):
            frame1 = cv2.imread(path1)
            frame2 = cv2.imread(path2)
            frame = np.concatenate((frame1, frame2), axis=1)
            out.write(frame)


if __name__ == '__main__':
    main()
