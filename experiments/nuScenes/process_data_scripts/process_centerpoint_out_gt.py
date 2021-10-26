import sys
import os
import dill
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Dict
from pyquaternion import Quaternion
import yaml

from kalman_filter import NonlinearKinematicBicycle
import json
nu_path = '../devkit/python-sdk/'
sys.path.append(nu_path)
sys.path.append("../../../trajectron")
from environment import Environment, Scene, Node
from environment import derivative_of_new as derivative_of

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


def process_scene(scene, env):
    scene_id = int(scene['sample_id'])
    data = pd.DataFrame(columns=['frame_id',
                                 'type',
                                 'node_id',
                                 'robot',
                                 'x', 'y', 'z',
                                 'length',
                                 'width',
                                 'height',
                                 'heading'])

    label2category = {'пешеход': env.NodeType.PEDESTRIAN,
                      'человек на самокате': env.NodeType.PEDESTRIAN,
                      'велосипедист': env.NodeType.PEDESTRIAN}

    for agent in scene['objects'].values():
        agent_id = agent['object_id']
        label = agent['object_label']

        if label not in label2category:
            print(label)
            continue
        our_category = label2category[label]

        width, length, height = agent['object_dimensions']

        rotation = agent['object_rotation']
        yaws = [rot[-1] for rot in rotation]

        trajs = agent['object_trajectory']

        frame_id = 6 - len(yaws)
        for traj, yaw in zip(trajs, yaws):
            data_point = pd.Series({'frame_id': frame_id,
                                    'type': our_category,
                                    'node_id': str(agent_id),
                                    'robot': False,
                                    'x': traj[0],
                                    'y': traj[1],
                                    'z': 0.0,
                                    'length': length,
                                    'width': width,
                                    'height': height,
                                    'heading': yaw})
            data = data.append(data_point, ignore_index=True)
            frame_id += 1

        if agent_id == scene['target_id']:
            target_trajs = scene['target']
            frame_id = 6
            for traj in target_trajs:
                data_point = pd.Series({'frame_id': frame_id,
                                        'type': our_category,
                                        'node_id': str(agent_id),
                                        'robot': False,
                                        'x': traj[0],
                                        'y': traj[1],
                                        'z': 0.0,
                                        'length': length,
                                        'width': width,
                                        'height': height,
                                        'heading': None})
                data = data.append(data_point, ignore_index=True)
                frame_id += 1

    if len(data.index) == 0:
        return None

    data.sort_values('frame_id', inplace=True)
    max_timesteps = data['frame_id'].max()

    center_y, center_x, width, height = get_viewport(data['x'].to_numpy(), data['y'].to_numpy())

    x_min = np.round(data['x'].min())
    x_max = np.round(data['x'].max())
    y_min = np.round(data['y'].min())
    y_max = np.round(data['y'].max())

    data['x'] = data['x'] - x_min
    data['y'] = data['y'] - y_min

    scene = Scene(timesteps=max_timesteps + 1, dt=dt, name=str(scene_id))
    ###
    scene.x_min = x_min
    scene.y_min = y_min

    scene.center_x = center_x
    scene.center_y = center_y
    scene.width = width
    scene.height = height
    ###

    for node_id in tqdm(pd.unique(data['node_id'])):
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


def process_data(data_dir, mode, output_path):
    os.makedirs(output_path, exist_ok=True)
    env = Environment(node_type_list=['VEHICLE', 'PEDESTRIAN'], standardization=standardization)
    attention_radius = dict()
    attention_radius[(env.NodeType.PEDESTRIAN, env.NodeType.PEDESTRIAN)] = 10.0
    attention_radius[(env.NodeType.PEDESTRIAN, env.NodeType.VEHICLE)] = 20.0
    attention_radius[(env.NodeType.VEHICLE, env.NodeType.PEDESTRIAN)] = 20.0
    attention_radius[(env.NodeType.VEHICLE, env.NodeType.VEHICLE)] = 30.0

    env.attention_radius = attention_radius

    filenames = [x for x in os.listdir(f'{data_dir}/{mode}') if '.json' in x]

    traj_scenes = []
    for filename in filenames:
        with open(f'{data_dir}/{mode}/{filename}', 'r') as f:
            data = json.load(f)
        for key in data:
            traj_scene = process_scene(data[key], env)
            traj_scenes.append(traj_scene)

    print(f'Processed {len(traj_scenes):.2f} scenes')

    env.scenes = traj_scenes

    if len(traj_scenes) > 0:
        data_dict_path = os.path.join(output_path, f'gt_{mode}.pkl')
        with open(data_dict_path, 'wb') as f:
            dill.dump(env, f, protocol=dill.HIGHEST_PROTOCOL)
        print('Saved Environment!')


if __name__ == '__main__':
    #parser = argparse.ArgumentParser()
    #parser.add_argument('--data_dir', type=str, required=True)
    #parser.add_argument('--mode', type=str, required=True)
    #parser.add_argument('--output_path', type=str, required=True)
    #args = parser.parse_args()
    #process_data(args.data_dir, args.mode, args.output_path)
    process_data('/home/cds-k/Desktop/centerpoint_out/motion_prediciton_validation_2.5Hz',
                 'validation',
                 '/home/cds-k/Desktop/centerpoint_out/motion_prediciton_validation_2.5Hz')
