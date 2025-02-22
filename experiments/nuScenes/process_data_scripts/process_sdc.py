import random
import sys
import os
import dill
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import itertools
import torch

#from .kalman_filter import NonlinearKinematicBicycle

# sys.path.append("/home/adeshkin/Desktop/shifts/sdc")
from ysdc_dataset_api.utils import get_file_paths, scenes_generator, get_latest_track_state_by_id, get_to_track_frame_transform
from ysdc_dataset_api.features import FeatureRenderer

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

renderer_config = {
    # parameters of feature maps to render
    'feature_map_params': {
        'rows': 400,
        'cols': 400,
        'resolution': 0.25,  # number of meters in one pixel
    },
    'renderers_groups': [
        # Having several feature map groups
        # allows to independently render feature maps with different history length.
        # This could be useful to render static features (road graph, etc.) once.
        {
            # start: int, first timestamp into the past to render, 0 – prediction time
            # stop: int, last timestamp to render inclusively, 24 – farthest known point into the past
            # step: int, grid step size,
            #            step=1 renders all points between start and stop,
            #            step=2 renders every second point, etc.
            'time_grid_params': {
                'start': 0,
                'stop': 0,
                'step': 1,
            },
            'renderers': [
                # each value is rendered at its own channel
                # occupancy -- 1 channel
                # velocity -- 2 channels (x, y)
                # acceleration -- 2 channels (x, y)
                # yaw -- 1 channel
                {'vehicles': ['occupancy']},
                # only occupancy and velocity are available for pedestrians
                {'pedestrians': ['occupancy']},
            ]
        },
        {
            'time_grid_params': {
                'start': 0,
                'stop': 0,
                'step': 1,
            },
            'renderers': [
                {
                    'road_graph': [
                        'crosswalk_occupancy',
                        'crosswalk_availability',
                        'lane_availability',
                        'lane_direction',
                        'lane_occupancy',
                        'lane_priority',
                        'lane_speed_limit',
                        'road_polygons',
                    ]
                }
            ]
        }
    ]
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
    # dt_in = 0.2
    # dt_out = dt
    # step = int(dt_out / dt_in)

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
        'past_vehicle': sdc_scene.past_vehicle_tracks,  # [::3],
        'future_vehicle': sdc_scene.future_vehicle_tracks,  # [2::3],
        'past_pedestrian': sdc_scene.past_pedestrian_tracks,  # [::3],
        'future_pedestrian': sdc_scene.future_pedestrian_tracks,  # [2::3]
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
    
    center_y, center_x, width, height = get_viewport(data['x'].to_numpy(), data['y'].to_numpy())
    
    x_min = np.round(data['x'].min())
    x_max = np.round(data['x'].max())
    y_min = np.round(data['y'].min())
    y_max = np.round(data['y'].max())

    data['x'] = data['x'] - x_min
    data['y'] = data['y'] - y_min

    scene = Scene(timesteps=max_timesteps + 1, dt=dt, name=str(scene_id))
    ###
    prediction_request_agent_ids = set(str(pr.track_id) for pr in sdc_scene.prediction_requests)
    scene.prediction_request_agent_ids = prediction_request_agent_ids
    scene.x_min = x_min
    scene.y_min = y_min
    scene.center_x = center_x
    scene.center_y = center_y
    scene.width = width
    scene.height = height

    prediction_request_agent_ids = {str(pr.track_id): pr for pr in sdc_scene.prediction_requests}
    map_ = []
    for request in prediction_request_agent_ids.values():
        track = get_latest_track_state_by_id(sdc_scene, request.track_id)
        to_track_frame_tf = get_to_track_frame_transform(track)
        renderer = FeatureRenderer(renderer_config)

        map_.append(renderer.produce_features(sdc_scene, to_track_frame_tf)['feature_maps'])

    tensor_map = np.stack(map_, axis=0)
    scene.map = torch.from_numpy(tensor_map)
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
    random.shuffle(filepaths)

    env = Environment(node_type_list=['VEHICLE', 'PEDESTRIAN'], standardization=standardization)
    attention_radius = dict()
    attention_radius[(env.NodeType.PEDESTRIAN, env.NodeType.PEDESTRIAN)] = 10.0
    attention_radius[(env.NodeType.PEDESTRIAN, env.NodeType.VEHICLE)] = 20.0
    attention_radius[(env.NodeType.VEHICLE, env.NodeType.PEDESTRIAN)] = 20.0
    attention_radius[(env.NodeType.VEHICLE, env.NodeType.VEHICLE)] = 30.0
    env.attention_radius = attention_radius

    if version == 'train':
        num_scenes = 5
    elif version == 'validation':
        num_scenes = 5

    scenes = []
    sdc_scenes = itertools.islice(scenes_generator(filepaths), num_scenes)
    for sdc_scene in tqdm(sdc_scenes):
        scene = process_scene(sdc_scene, env)
        scenes.append(scene)

    print(f'Processed {len(scenes):.2f} scenes')

    env.scenes = scenes
    if len(scenes) > 0:
        data_dict_path = os.path.join(output_path, f'{version}/sdc_{version}.pkl')
        with open(data_dict_path, 'wb') as f:
            dill.dump(env, f, protocol=dill.HIGHEST_PROTOCOL)
        print('Saved Environment!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--version', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    args = parser.parse_args()
    
    process_data(args.data, args.version, args.output_path)
    #process_data('/media/cds-k/Data_2/canonical-trn-dev-data/data', 'validation', '/media/cds-k/data/nuScenes/traj++_processed_data/processed_sdc')