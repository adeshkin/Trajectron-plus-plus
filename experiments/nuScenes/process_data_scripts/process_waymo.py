import sys
import os
import dill
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import tensorflow as tf

from kalman_filter import NonlinearKinematicBicycle

nu_path = '../devkit/python-sdk/'
sys.path.append(nu_path)
sys.path.append("../../../trajectron")
from environment import Environment, Scene, Node, derivative_of

FREQUENCY = 2
dt = 1 / FREQUENCY
data_columns_vehicle = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration', 'heading'], ['x', 'y']])
data_columns_vehicle = data_columns_vehicle.append(pd.MultiIndex.from_tuples([('heading', '°'), ('heading', 'd°')]))
data_columns_vehicle = data_columns_vehicle.append(pd.MultiIndex.from_product([['velocity', 'acceleration'], ['norm']]))

data_columns_pedestrian = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration'], ['x', 'y']])

curv_0_2 = 0
curv_0_1 = 0
total = 0

standardization = {
    'PEDESTRIAN': {
        'position': {
            'x': {'mean': 0, 'std': 1},
            'y': {'mean': 0, 'std': 1}
        },
        'velocity': {
            'x': {'mean': 0, 'std': 2},
            'y': {'mean': 0, 'std': 2}
        },
        'acceleration': {
            'x': {'mean': 0, 'std': 1},
            'y': {'mean': 0, 'std': 1}
        }
    },
    'VEHICLE': {
        'position': {
            'x': {'mean': 0, 'std': 80},
            'y': {'mean': 0, 'std': 80}
        },
        'velocity': {
            'x': {'mean': 0, 'std': 15},
            'y': {'mean': 0, 'std': 15},
            'norm': {'mean': 0, 'std': 15}
        },
        'acceleration': {
            'x': {'mean': 0, 'std': 4},
            'y': {'mean': 0, 'std': 4},
            'norm': {'mean': 0, 'std': 4}
        },
        'heading': {
            'x': {'mean': 0, 'std': 1},
            'y': {'mean': 0, 'std': 1},
            '°': {'mean': 0, 'std': np.pi},
            'd°': {'mean': 0, 'std': 1}
        }
    }
}

# Features of other agents.
state_features = {
    'state/id':
        tf.io.FixedLenFeature([128], tf.float32, default_value=None),
    'state/type':
        tf.io.FixedLenFeature([128], tf.float32, default_value=None),
    'state/is_sdc':
        tf.io.FixedLenFeature([128], tf.int64, default_value=None),
    'state/tracks_to_predict':
        tf.io.FixedLenFeature([128], tf.int64, default_value=None),
    'state/current/bbox_yaw':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/height':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/length':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/timestamp_micros':
        tf.io.FixedLenFeature([128, 1], tf.int64, default_value=None),
    'state/current/valid':
        tf.io.FixedLenFeature([128, 1], tf.int64, default_value=None),
    'state/current/vel_yaw':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/velocity_x':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/velocity_y':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/width':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/x':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/y':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/z':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/future/bbox_yaw':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/height':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/length':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/timestamp_micros':
        tf.io.FixedLenFeature([128, 80], tf.int64, default_value=None),
    'state/future/valid':
        tf.io.FixedLenFeature([128, 80], tf.int64, default_value=None),
    'state/future/vel_yaw':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/velocity_x':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/velocity_y':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/width':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/x':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/y':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/z':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/past/bbox_yaw':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/height':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/length':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/timestamp_micros':
        tf.io.FixedLenFeature([128, 10], tf.int64, default_value=None),
    'state/past/valid':
        tf.io.FixedLenFeature([128, 10], tf.int64, default_value=None),
    'state/past/vel_yaw':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/velocity_x':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/velocity_y':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/width':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/x':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/y':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/z':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
}


def trajectory_curvature(t):
    path_distance = np.linalg.norm(t[-1] - t[0])

    lengths = np.sqrt(np.sum(np.diff(t, axis=0) ** 2, axis=1))  # Length between points
    path_length = np.sum(lengths)
    if np.isclose(path_distance, 0.):
        return 0, 0, 0
    return (path_length / path_distance) - 1, path_length, path_distance


def get_viewport(all_states, all_states_mask):
    """Gets the region containing the data.

    Args:
        all_states: states of agents as an array of shape [num_agents, num_steps, 2].
        all_states_mask: binary mask of shape [num_agents, num_steps] for `all_states`.

    Returns:
        center_y: float. y coordinate for center of data.
        center_x: float. x coordinate for center of data.
        width: float. Width of data.
    """
    valid_states = all_states[all_states_mask]
    all_y = valid_states[..., 1]
    all_x = valid_states[..., 0]

    center_y = (np.max(all_y) + np.min(all_y)) / 2
    center_x = (np.max(all_x) + np.min(all_x)) / 2

    range_y = np.ptp(all_y)
    range_x = np.ptp(all_x)

    width = max(range_y, range_x)

    return center_y, center_x, width


def parse_data(data, features_description):
    parsed = tf.io.parse_single_example(data, features_description)

    ###
    classes = parsed['state/type'].numpy()
    all_ids = parsed['state/id'].numpy()
    all_lengths = np.concatenate([parsed['state/past/length'].numpy(), parsed['state/current/length'].numpy(),
                                  parsed['state/future/length'].numpy()], 1)
    all_heights = np.concatenate([parsed['state/past/height'].numpy(), parsed['state/current/height'].numpy(),
                                  parsed['state/future/height'].numpy()], 1)
    all_widths = np.concatenate([parsed['state/past/width'].numpy(), parsed['state/current/width'].numpy(),
                                 parsed['state/future/width'].numpy()], 1)
    all_yaws = np.concatenate([parsed['state/past/bbox_yaw'].numpy(), parsed['state/current/bbox_yaw'].numpy(),
                               parsed['state/future/bbox_yaw'].numpy()], 1)

    # [num_agents, num_past_steps, 2] float32.
    past_states = tf.stack(
        [parsed['state/past/x'], parsed['state/past/y']],
        -1).numpy()
    past_states_mask = parsed['state/past/valid'].numpy() > 0.0

    # [num_agents, 1, 2] float32.
    current_states = tf.stack(
        [parsed['state/current/x'], parsed['state/current/y']],
        -1).numpy()
    current_states_mask = parsed['state/current/valid'].numpy() > 0.0

    # [num_agents, num_future_steps, 2] float32.
    future_states = tf.stack(
        [parsed['state/future/x'], parsed['state/future/y']],
        -1).numpy()
    future_states_mask = parsed['state/future/valid'].numpy() > 0.0

    # [num_agens, num_past_steps + 1 + num_future_steps, depth] float32.
    all_states = np.concatenate([past_states, current_states, future_states], 1)

    # [num_agens, num_past_steps + 1 + num_future_steps] float32.
    all_states_mask = np.concatenate(
        [past_states_mask, current_states_mask, future_states_mask], 1)

    dt_in = 0.1
    dt_out = 0.5
    step = int(dt_out / dt_in)
    parsed_data = {
        'class': classes,
        'state': all_states[:, ::step],
        'mask': all_states_mask[:, ::step],
        'length': all_lengths[:, ::step],
        'height': all_heights[:, ::step],
        'width': all_widths[:, ::step],
        'yaw': all_yaws[:, ::step],
        'id': all_ids
    }
    return parsed_data


def process_scene(scene_id, parsed_data, env, center_y, center_x, width):
    data = pd.DataFrame(columns=['frame_id',
                                 'type',
                                 'node_id',
                                 'robot',
                                 'x', 'y', 'z',
                                 'length',
                                 'width',
                                 'height',
                                 'heading'])

    class_id2category = {1.: env.NodeType.VEHICLE,
                         2.: env.NodeType.PEDESTRIAN}
    num_steps = parsed_data['state'].shape[1]
    for frame_id, (s, m, l, h, w, y) in enumerate(zip(np.split(parsed_data['state'], num_steps, 1),
                                                      np.split(parsed_data['mask'], num_steps, 1),
                                                      np.split(parsed_data['length'], num_steps, 1),
                                                      np.split(parsed_data['height'], num_steps, 1),
                                                      np.split(parsed_data['width'], num_steps, 1),
                                                      np.split(parsed_data['yaw'], num_steps, 1))):
        states = s[:, 0]
        masks = m[:, 0]
        lengths = l[:, 0]
        heights = h[:, 0]
        widths = w[:, 0]
        yaws = y[:, 0]

        masked_x = states[:, 0][masks]
        masked_y = states[:, 1][masks]
        masked_l = lengths[masks]
        masked_h = heights[masks]
        masked_w = widths[masks]
        masked_yaw = yaws[masks]
        masked_id = parsed_data['id'][masks]
        masked_classes = parsed_data['class'][masks]
        for class_id in [1., 2.]:
            for j, agent_id in enumerate(masked_id[masked_classes == class_id]):
                node_id = str(int(agent_id))
                our_category = class_id2category[class_id]
                x_ = masked_x[masked_classes == class_id][j]
                y_ = masked_y[masked_classes == class_id][j]
                l_ = masked_l[masked_classes == class_id][j]
                w_ = masked_w[masked_classes == class_id][j]
                h_ = masked_h[masked_classes == class_id][j]
                yaw_ = masked_yaw[masked_classes == class_id][j]
                data_point = pd.Series({'frame_id': frame_id,
                                        'type': our_category,
                                        'node_id': node_id,
                                        'robot': False,
                                        'x': x_,
                                        'y': y_,
                                        'z': 0.0,
                                        'length': l_,
                                        'width': w_,
                                        'height': h_,
                                        'heading': yaw_})
                data = data.append(data_point, ignore_index=True)
    if len(data.index) == 0:
        return None

    data.sort_values('frame_id', inplace=True)
    max_timesteps = data['frame_id'].max()
    x_min = np.round(data['x'].min() - 50)
    x_max = np.round(data['x'].max() + 50)
    y_min = np.round(data['y'].min() - 50)
    y_max = np.round(data['y'].max() + 50)

    data['x'] = data['x'] - x_min
    data['y'] = data['y'] - y_min

    scene = Scene(timesteps=max_timesteps + 1, dt=dt, name=str(scene_id))
    ###
    scene.x_min = x_min
    scene.y_min = y_min
    scene.center_x = center_x
    scene.center_y = center_y
    scene.width = width
    ###
    for node_id in pd.unique(data['node_id']):
        node_frequency_multiplier = 1
        node_df = data[data['node_id'] == node_id]
        if node_df['x'].shape[0] < 2:
            continue

        if not np.all(np.diff(node_df['frame_id']) == 1):
            # print('Occlusion')
            continue  # TODO Make better

        node_values = node_df[['x', 'y']].values
        x = node_values[:, 0]
        y = node_values[:, 1]
        heading = node_df['heading'].values
        if node_df.iloc[0]['type'] == env.NodeType.VEHICLE:
            # Kalman filter Agent
            vx = derivative_of(x, scene.dt)
            vy = derivative_of(y, scene.dt)
            velocity = np.linalg.norm(np.stack((vx, vy), axis=-1), axis=-1)

            filter_veh = NonlinearKinematicBicycle(dt=scene.dt, sMeasurement=1.0)
            P_matrix = None
            for i in range(len(x)):
                if i == 0:  # initalize KF
                    # initial P_matrix
                    P_matrix = np.identity(4)
                elif i < len(x):
                    # assign new est values
                    x[i] = x_vec_est_new[0][0]
                    y[i] = x_vec_est_new[1][0]
                    heading[i] = x_vec_est_new[2][0]
                    velocity[i] = x_vec_est_new[3][0]

                if i < len(x) - 1:  # no action on last data
                    # filtering
                    x_vec_est = np.array([[x[i]],
                                          [y[i]],
                                          [heading[i]],
                                          [velocity[i]]])
                    z_new = np.array([[x[i + 1]],
                                      [y[i + 1]],
                                      [heading[i + 1]],
                                      [velocity[i + 1]]])
                    x_vec_est_new, P_matrix_new = filter_veh.predict_and_update(
                        x_vec_est=x_vec_est,
                        u_vec=np.array([[0.], [0.]]),
                        P_matrix=P_matrix,
                        z_new=z_new
                    )
                    P_matrix = P_matrix_new

            curvature, pl, _ = trajectory_curvature(np.stack((x, y), axis=-1))
            if pl < 1.0:  # vehicle is "not" moving
                x = x[0].repeat(max_timesteps + 1)
                y = y[0].repeat(max_timesteps + 1)
                heading = heading[0].repeat(max_timesteps + 1)
            global total
            global curv_0_2
            global curv_0_1
            total += 1
            if pl > 1.0:
                if curvature > .2:
                    curv_0_2 += 1
                    node_frequency_multiplier = 3 * int(np.floor(total / curv_0_2))
                elif curvature > .1:
                    curv_0_1 += 1
                    node_frequency_multiplier = 3 * int(np.floor(total / curv_0_1))
        vx = derivative_of(x, scene.dt)
        vy = derivative_of(y, scene.dt)
        ax = derivative_of(vx, scene.dt)
        ay = derivative_of(vy, scene.dt)

        if node_df.iloc[0]['type'] == env.NodeType.VEHICLE:
            v = np.stack((vx, vy), axis=-1)
            v_norm = np.linalg.norm(np.stack((vx, vy), axis=-1), axis=-1, keepdims=True)
            heading_v = np.divide(v, v_norm, out=np.zeros_like(v), where=(v_norm > 1.))
            heading_x = heading_v[:, 0]
            heading_y = heading_v[:, 1]

            data_dict = {('position', 'x'): x,
                         ('position', 'y'): y,
                         ('velocity', 'x'): vx,
                         ('velocity', 'y'): vy,
                         ('velocity', 'norm'): np.linalg.norm(np.stack((vx, vy), axis=-1), axis=-1),
                         ('acceleration', 'x'): ax,
                         ('acceleration', 'y'): ay,
                         ('acceleration', 'norm'): np.linalg.norm(np.stack((ax, ay), axis=-1), axis=-1),
                         ('heading', 'x'): heading_x,
                         ('heading', 'y'): heading_y,
                         ('heading', '°'): heading,
                         ('heading', 'd°'): derivative_of(heading, dt, radian=True)}
            node_data = pd.DataFrame(data_dict, columns=data_columns_vehicle)
        else:
            data_dict = {('position', 'x'): x,
                         ('position', 'y'): y,
                         ('velocity', 'x'): vx,
                         ('velocity', 'y'): vy,
                         ('acceleration', 'x'): ax,
                         ('acceleration', 'y'): ay}
            node_data = pd.DataFrame(data_dict, columns=data_columns_pedestrian)

        node = Node(node_type=node_df.iloc[0]['type'], node_id=node_id, data=node_data,
                    frequency_multiplier=node_frequency_multiplier)
        node.first_timestep = node_df['frame_id'].iloc[0]
        scene.nodes.append(node)

    return scene


def process_data(data_path, version, output_path):
    os.makedirs(output_path, exist_ok=True)

    dataset = tf.data.TFRecordDataset(data_path, compression_type='')

    features_description = {}
    features_description.update(state_features)

    env = Environment(node_type_list=['VEHICLE', 'PEDESTRIAN'], standardization=standardization)
    attention_radius = dict()
    attention_radius[(env.NodeType.PEDESTRIAN, env.NodeType.PEDESTRIAN)] = 10.0
    attention_radius[(env.NodeType.PEDESTRIAN, env.NodeType.VEHICLE)] = 20.0
    attention_radius[(env.NodeType.VEHICLE, env.NodeType.PEDESTRIAN)] = 20.0
    attention_radius[(env.NodeType.VEHICLE, env.NodeType.VEHICLE)] = 30.0

    env.attention_radius = attention_radius

    scenes = []
    for scene_id, data in tqdm(enumerate(dataset.as_numpy_iterator())):
        parsed_data = parse_data(data, features_description)
        center_y, center_x, width = get_viewport(parsed_data['state'], parsed_data['mask'])
        scene = process_scene(scene_id, parsed_data, env, center_y, center_x, width)
        scenes.append(scene)

    print(f'Processed {len(scenes):.2f} scenes')

    env.scenes = scenes
    if len(scenes) > 0:
        data_dict_path = os.path.join(output_path, f'waymo_{version}.pkl')
        with open(data_dict_path, 'wb') as f:
            dill.dump(env, f, protocol=dill.HIGHEST_PROTOCOL)
        print('Saved Environment!')

    global total
    global curv_0_2
    global curv_0_1
    print(f"Total Nodes: {total}")
    print(f"Curvature > 0.1 Nodes: {curv_0_1}")
    print(f"Curvature > 0.2 Nodes: {curv_0_2}")
    total = 0
    curv_0_1 = 0
    curv_0_2 = 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--data', type=str, required=True)
    #parser.add_argument('--version', type=str, required=True)
    #parser.add_argument('--output_path', type=str, required=True)
    #args = parser.parse_args()
    # process_data(args.data, args.version, args.output_path)
    process_data('/media/cds-k/Data_2/waymo_motion/validation/validation_tfexample.tfrecord-00000-of-00150',
                 'val',
                 '../../processed_waymo')
