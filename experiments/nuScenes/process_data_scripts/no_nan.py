import numpy as np

yaws = np.array([np.nan, 1, np.nan, 1, 2, 3, 4, np.nan, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                 np.nan, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, np.nan])
yaws = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, np.nan, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])


def get_correct_ids(yaws, len_timesteps=11):
    start_no_nan = -1
    count = 0
    end_no_nan = -1
    no_nan_ids = []
    for frame_id, yaw in enumerate(yaws):
        if not np.isnan(yaw) and count == 0:
            start_no_nan = frame_id
            count += 1
        elif not np.isnan(yaw):
            count += 1

        if np.isnan(yaw) or frame_id == len(yaws)-1:
            if np.isnan(yaw):
                end_no_nan = frame_id - 1

            elif frame_id == len(yaws)-1:
                end_no_nan = frame_id

            if count >= len_timesteps:
                no_nan_ids.append((start_no_nan, end_no_nan))

            count = 0

    return no_nan_ids
