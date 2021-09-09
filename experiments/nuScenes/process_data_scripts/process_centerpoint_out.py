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


def load_msgs_as_list(yaml_file_path: str) -> list:
    """
    Load ROSBAG data written in a .yaml file as a list of dicts.
    Every dict in a list corresponds to a ROSBAG time step.
    :param yaml_file_path: path to .yaml file.
    :return: list of time steps as a list of dicts.
    """

    with open(yaml_file_path) as f:
        try:
            msgs_list = list(yaml.load_all(f, Loader=yaml.FullLoader))
        except yaml.YAMLError as e:
            print(e)
    return msgs_list


def downsample_scenes(scenes_list: list,
                      input_dt: float = 0.1,
                      output_dt: float = 0.5) -> list:

    """
    10 Hz data frequency mey be redundent for the motion prediction task.
    Function used to downsampling with frequency `1 / output_dt`.
    Input frequency supposed to be constant, may be estimated as average input frequency.
    :param scenes_list: list of scenes, every scene represented as dict.
    :param input_dt: estimated mean input inverse frequency (supposed to be constant).
    :param output_dt: inverse frequency of output data.
    :return: downsampled list of scenes.
    """

    assert output_dt >= input_dt

    step_size = int(output_dt / input_dt)

    for scene_id, scene in enumerate(scenes_list):

        assert scene_id == scene['scene_id']

        for agent_id, agent in scene['agents'].items():
            traj_ds = agent['trajectory'][::step_size]
            yaws_ds = agent['yaws'][::step_size]

            scenes_list[scene_id]['agents'][agent_id]['trajectory'] = traj_ds
            scenes_list[scene_id]['agents'][agent_id]['yaws'] = yaws_ds

    return scenes_list


class MessagesToScenes:
    """
    Used to create scenes from ROSBAG messages.
    Messages supposed to be represented as list of dicts.
    Scene - chunk of data with duration `scene_duration_sec`.
    Scene used for creation of training and validation samples:
    it can be splited in two parts, first part of scene represent input, second part - target.
    """

    def __init__(self,
                 time_step: float = 0.1,  # time step of raw data
                 scene_duration_sec: int = 20,
                 score_threshold: float = 0.5  # agents with score below will be ignored
                 ):
        self.scene_duration_sec = scene_duration_sec
        self.n_ts_in_scene = int(scene_duration_sec / time_step)
        self.score_threshold = score_threshold
        self.scenes = []

    def get_new_scene(self, scene_id: int, first_timestamp: Dict) -> Dict:
        blank_scene = {
            'scene_id': scene_id,
            'first_timestamp': first_timestamp,
            'last_timestamp': None,
            'agents': {}
        }
        return blank_scene

    def get_yaw(self, orientation: Dict) -> float:
        q = Quaternion(w=orientation['w'],
                       x=orientation['x'],
                       y=orientation['y'],
                       z=orientation['z'])
        yaw = q.yaw_pitch_roll[0]
        return yaw

    def add_agent_to_scene(self, scene: Dict, obj: Dict) -> Dict:

        """
        Adding new agent to scene.
        Scene timeline supposed to be in a grid defined by input data frequency and scene duration;
        e.g. 5s scene with 10Hz input frequency leads to 50 timesteps in a scene.
        Absence of trajectory data for particular time step depicts as zero value.
        """

        agent = {'id': obj['id'],
                 'label': obj['label'],
                 'score': obj['score'],
                 'size': obj['size'],
                 'trajectory': np.full((self.n_ts_in_scene, 2), np.nan),
                 'yaws': np.full(self.n_ts_in_scene, np.nan)}

        agent['trajectory'][0, 0] = obj['pose']['position']['x']
        agent['trajectory'][0, 1] = obj['pose']['position']['y']
        agent['yaws'][0] = self.get_yaw(obj['pose']['orientation'])

        scene['agents'][obj['id']] = agent

        return scene

    def update_agent(self, scene: Dict, obj: Dict, n_ts: int) -> Dict:

        """ Update existing agent on scene. """

        scene['agents'][obj['id']]['trajectory'][n_ts, 0] = obj['pose']['position']['x']
        scene['agents'][obj['id']]['trajectory'][n_ts, 1] = obj['pose']['position']['y']
        scene['agents'][obj['id']]['yaws'][n_ts] = self.get_yaw(obj['pose']['orientation'])

        return scene

    def fill_scene(self, scene: Dict, lidar_objects: list, n_ts: int) -> Dict:

        """ Fill scene with data from particular time step. """

        for obj in lidar_objects:
            agent_id = obj['id']

            if obj['score'] >= self.score_threshold:
                if agent_id in scene['agents'].keys():
                    scene = self.update_agent(scene, obj, n_ts)
                else:
                    scene = self.add_agent_to_scene(scene, obj)
            else:
                continue
        return scene

    def create_scenes(self, lidar_msgs_list: list) -> list:

        """ Main function that create list of scenes. """

        n_ts = 0  # number of time step within scene
        scene_id = 0

        for msg in tqdm(lidar_msgs_list):

            if isinstance(msg, Dict):

                n_ts += 1

                ts = msg['header']['stamp']
                lidar_objects = msg['objects']

                if n_ts == 1:
                    scene = self.get_new_scene(scene_id, ts)
                    scene = self.fill_scene(scene, lidar_objects, n_ts - 1)

                elif n_ts == self.n_ts_in_scene:
                    scene['last_timestamp'] = ts
                    scene = self.fill_scene(scene, lidar_objects, n_ts - 1)
                    self.scenes.append(scene)
                    scene_id += 1
                    n_ts = 0

                else:
                    scene = self.fill_scene(scene, lidar_objects, n_ts - 1)

        # Add last incomplete scene
        if n_ts != 0:
            self.scenes.append(scene)

        return self.scenes


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

    range_y = np.ptp(all_y)
    range_x = np.ptp(all_x)

    width = max(range_y, range_x)

    return center_y, center_x, width


def process_scene(scene, env):
    scene_id = int(scene['scene_id'])
    data = pd.DataFrame(columns=['frame_id',
                                 'type',
                                 'node_id',
                                 'robot',
                                 'x', 'y', 'z',
                                 'length',
                                 'width',
                                 'height',
                                 'heading'])
    label2category = {0: env.NodeType.VEHICLE,
                      1: env.NodeType.PEDESTRIAN}

    for agent in scene['agents'].values():
        agent_id = agent['id']
        label = agent['label']

        if label not in label2category:
            continue

        size = agent['size']
        traj = agent['trajectory']
        yaws = agent['yaws']
        node_id = str(agent_id + 1)
        our_category = label2category[label]

        not_nan_idxs = [i for i, elem in enumerate(np.isnan(yaws)[:-1]) if elem == False and np.isnan(yaws)[i + 1] == False]

        if len(not_nan_idxs) == 0:
            continue

        for frame_id in not_nan_idxs:
            data_point = pd.Series({'frame_id': frame_id,
                                    'type': our_category,
                                    'node_id': node_id,
                                    'robot': False,
                                    'x': traj[frame_id][0],
                                    'y': traj[frame_id][1],
                                    'z': 0.0,
                                    'length': size['x'],
                                    'width': size['y'],
                                    'height': size['z'],
                                    'heading': yaws[frame_id]})
            data_point_next = pd.Series({'frame_id': frame_id+1,
                                    'type': our_category,
                                    'node_id': node_id,
                                    'robot': False,
                                    'x': traj[frame_id+1][0],
                                    'y': traj[frame_id+1][1],
                                    'z': 0.0,
                                    'length': size['x'],
                                    'width': size['y'],
                                    'height': size['z'],
                                    'heading': yaws[frame_id+1]})
            data = data.append(data_point, ignore_index=True)
            data = data.append(data_point_next, ignore_index=True)

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
    center_y, center_x, width = get_viewport(data['x'].to_numpy(), data['y'].to_numpy())
    scene.center_x = center_x
    scene.center_y = center_y
    scene.width = width
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
        '''
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
        '''
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


def process_data(data_dir, filename, output_path):
    os.makedirs(output_path, exist_ok=True)

    cfg = {
        'dataloader': {
            'in_time_res_sec': 0.1,
            'out_time_res_sec': 0.5,
            'score_threshold': 0.05,
            'scene_duration_sec': 20,
        }
    }

    msgs = load_msgs_as_list(f'{data_dir}/{filename}.yaml')
    scenes = MessagesToScenes(time_step=cfg['dataloader']['in_time_res_sec'],
                              scene_duration_sec=cfg['dataloader']['scene_duration_sec'],
                              score_threshold=cfg['dataloader']['score_threshold'])

    scenes_list = scenes.create_scenes(msgs)
    ds_scenes_list = downsample_scenes(scenes_list,
                                       input_dt=cfg['dataloader']['in_time_res_sec'],
                                       output_dt=cfg['dataloader']['out_time_res_sec'])

    env = Environment(node_type_list=['VEHICLE', 'PEDESTRIAN'], standardization=standardization)
    attention_radius = dict()
    attention_radius[(env.NodeType.PEDESTRIAN, env.NodeType.PEDESTRIAN)] = 10.0
    attention_radius[(env.NodeType.PEDESTRIAN, env.NodeType.VEHICLE)] = 20.0
    attention_radius[(env.NodeType.VEHICLE, env.NodeType.PEDESTRIAN)] = 20.0
    attention_radius[(env.NodeType.VEHICLE, env.NodeType.VEHICLE)] = 30.0

    env.attention_radius = attention_radius

    traj_scenes = []
    for scene in ds_scenes_list:
        traj_scene = process_scene(scene, env)
        traj_scenes.append(traj_scene)

    print(f'Processed {len(traj_scenes):.2f} scenes')

    env.scenes = traj_scenes

    if len(traj_scenes) > 0:
        data_dict_path = os.path.join(output_path, f'{filename}.pkl')
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
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--filename', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    args = parser.parse_args()
    process_data(args.data_dir, args.filename, args.output_path)
