import random
import sys
import os
import dill
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import itertools

from kalman_filter import NonlinearKinematicBicycle

# sys.path.append("/home/adeshkin/Desktop/shifts/sdc")
from ysdc_dataset_api.utils import get_file_paths, scenes_generator

sys.path.append("../../../trajectron")
from environment import Environment, Scene, Node
from environment import derivative_of as derivative_of


FREQUENCY = 5
dt = 1. / FREQUENCY


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


def trajectory_curvature(t):
    path_distance = np.linalg.norm(t[-1] - t[0])

    lengths = np.sqrt(np.sum(np.diff(t, axis=0) ** 2, axis=1))  # Length between points
    path_length = np.sum(lengths)
    if np.isclose(path_distance, 0.):
        return 0, 0, 0
    return (path_length / path_distance) - 1, path_length, path_distance


def get_viewport(x, y):
    """Gets the region containing the data.
    Returns:
        center_y: float. y coordinate for center of data.
        center_x: float. x coordinate for center of data.
        width: float. Width of data.
    """
    all_y = y
    all_x = x

    center_y = (np.max(all_y) + np.min(all_y)) / 2
    center_x = (np.max(all_x) + np.min(all_x)) / 2

    height = np.ptp(all_y)
    width = np.ptp(all_x)

    return center_y, center_x, width, height


def process_scene(sdc_scene, env):
    #dt_in = 0.2
    #dt_out = dt
    #step = int(dt_out / dt_in)

    data = pd.DataFrame(columns=['frame_id',
                                 'type',
                                 'node_id',
                                 'robot',
                                 'x', 'y', 'z',
                                 'length',
                                 'width',
                                 'height',
                                 'heading'])
    scene_id = str(sdc_scene.id)
    scene_tracks = {
        'past_vehicle': sdc_scene.past_vehicle_tracks,#[::3],
        'future_vehicle': sdc_scene.future_vehicle_tracks,#[2::3],
        'past_pedestrian': sdc_scene.past_pedestrian_tracks,#[::3],
        'future_pedestrian': sdc_scene.future_pedestrian_tracks,#[2::3]
    }

    for scene_track_name in scene_tracks:
        scene_track = scene_tracks[scene_track_name]

        if 'vehicle' in scene_track_name:
            category = env.NodeType.VEHICLE
        elif 'pedestrian' in scene_track_name:
            category = env.NodeType.PEDESTRIAN

        if 'past' in scene_track_name:
            frame_id = 0
        elif 'future' in scene_track_name:
            frame_id = 25

        for agent_tracks in scene_track:
            for agent_track in agent_tracks.tracks:
                agent_id = str(agent_track.track_id)
                x = agent_track.position.x
                y = agent_track.position.y
                z = agent_track.position.z
                size_x = agent_track.dimensions.x
                size_y = agent_track.dimensions.y
                size_z = agent_track.dimensions.z

                if 'vehicle' in scene_track_name:
                    yaw = agent_track.yaw
                elif 'pedestrian' in scene_track_name:
                    yaw = 0.0

                data_point = pd.Series({'frame_id': frame_id,
                                        'type': category,
                                        'node_id': agent_id,
                                        'robot': False,
                                        'x': x,
                                        'y': y,
                                        'z': z,
                                        'length': size_x,
                                        'width': size_y,
                                        'height': size_z,
                                        'heading': yaw})
                data = data.append(data_point, ignore_index=True)

            frame_id += 1

    if len(data.index) == 0:
        return None

    data.sort_values('frame_id', inplace=True)
    max_timesteps = data['frame_id'].max()
    
    # center_y, center_x, width, height = get_viewport(data['x'].to_numpy(), data['y'].to_numpy())
    
    x_min = np.round(data['x'].min())
    x_max = np.round(data['x'].max())
    y_min = np.round(data['y'].min())
    y_max = np.round(data['y'].max())

    data['x'] = data['x'] - x_min
    data['y'] = data['y'] - y_min

    scene = Scene(timesteps=max_timesteps + 1, dt=dt, name=str(scene_id))
    ###
    # prediction_request_agent_ids = set(str(pr.track_id) for pr in sdc_scene.prediction_requests)
    # scene.prediction_request_agent_ids = prediction_request_agent_ids
    #scene.x_min = x_min
    #scene.y_min = y_min
    #scene.center_x = center_x
    #scene.center_y = center_y
    #scene.width = width
    #scene.height = height
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
            # Kalman filter Agenti
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
    os.makedirs(f'{output_path}/{version}', exist_ok=True)
    
    dataset_path = f'{data_path}/{version}_pb/'
    filepaths = get_file_paths(dataset_path)

    part2filepaths = dict()
    for filepath in filepaths:
        part = filepath.split('/')[-2]
        if part not in part2filepaths:
            part2filepaths[part] = [filepath]
        else:
            part2filepaths[part].append(filepath)

    for part in part2filepaths:
        env = Environment(node_type_list=['VEHICLE', 'PEDESTRIAN'], standardization=standardization)
        attention_radius = dict()
        attention_radius[(env.NodeType.PEDESTRIAN, env.NodeType.PEDESTRIAN)] = 10.0
        attention_radius[(env.NodeType.PEDESTRIAN, env.NodeType.VEHICLE)] = 20.0
        attention_radius[(env.NodeType.VEHICLE, env.NodeType.PEDESTRIAN)] = 20.0
        attention_radius[(env.NodeType.VEHICLE, env.NodeType.VEHICLE)] = 30.0

        env.attention_radius = attention_radius

        scenes = []
        sdc_scenes = scenes_generator(part2filepaths[part])
        for sdc_scene in tqdm(sdc_scenes):
            scene = process_scene(sdc_scene, env)
            scenes.append(scene)

        print(f'Processed {len(scenes):.2f} scenes')

        env.scenes = scenes
        if len(scenes) > 0:
            data_dict_path = os.path.join(output_path, f'{version}/sdc_{version}_{part}.pkl')
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
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--version', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    args = parser.parse_args()
    
    process_data(args.data, args.version, args.output_path)
    # process_data('/media/cds-k/Data_2/canonical-trn-dev-data/data', 'validation', '/media/cds-k/data/nuScenes/traj++_processed_data/processed_sdc')