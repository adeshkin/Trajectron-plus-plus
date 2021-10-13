import dill
import yaml
import os
import json
import numpy as np
import matplotlib.patheffects as pe
from scipy.ndimage import rotate
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.pyplot as plt
import torch
import gc
import sys
from torch import nn, optim, utils

from ysdc_dataset_api.dataset import MotionPredictionDatasetTest
from ysdc_dataset_api.features import FeatureRenderer

sys.path.append("../../trajectron")

from model.model_registrar import ModelRegistrar
from model import Trajectron
from utils import prediction_output_to_trajectories
from environment import Environment
from model.dataset import collate

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


def load_model(model_dir, env, checkpoint_name=None):
    model_registrar = ModelRegistrar(model_dir, 'cuda')
    model_registrar.load_models(checkpoint_name)
    with open(os.path.join(model_dir, 'config.json'), 'r') as config_json:
        hyperparams = json.load(config_json)

    hyperparams['map_enc_dropout'] = 0.0
    if 'incl_robot_node' not in hyperparams:
        hyperparams['incl_robot_node'] = False

    stg = Trajectron(model_registrar, hyperparams, None, 'cuda')

    stg.set_environment(env)

    stg.set_annealing_params()

    return stg, hyperparams


def create_fig(my_patch):
    fig = plt.figure(figsize=(10, 10))
    local_aspect_ratio = 1
    ax = fig.add_axes([0, 0, 1, 1 / local_aspect_ratio])
    x_margin = 0
    y_margin = 0
    ax.set_xlim(my_patch[0] - x_margin, my_patch[2] + x_margin)
    ax.set_ylim(my_patch[1] - y_margin, my_patch[3] + y_margin)

    return fig, ax


def main(params):
    preprocessed_data_fn = f"/media/cds-k/data/nuScenes/traj++_processed_data/{params['preprocessed_data_dir']}/{params['dataset_name']}.pkl"
    scene_idx = params['scene_idx']
    ph = params['ph']
    max_h = params['max_h']
    num_samples = params['num_samples']
    z_mode = True
    gmm_mode = True

    CAR_IMAGES = (plt.imread('icons/Car TOP_VIEW 375397.png'),  # blue
                  plt.imread('icons/Car TOP_VIEW F05F78.png'),  # red
                  plt.imread('icons/Car TOP_VIEW 80CBE5.png'),  # cyan
                  plt.imread('icons/Car TOP_VIEW ABCB51.png'),  # green
                  plt.imread('icons/Car TOP_VIEW C8B0B0.png'))  # gray
    COLORS = ('blue', 'red', 'cyan', 'green', 'gray')
    COLORS_ped = ('tab:blue',
                  'tab:orange',
                  'tab:green',
                  'tab:red',
                  'tab:purple',
                  'tab:brown',
                  'tab:pink',
                  'tab:gray',
                  'tab:olive',
                  'tab:cyan')
    NCOLORS = len(COLORS)
    NCOLORS_ped = len(COLORS_ped)

    # Load dataset
    env = Environment(node_type_list=['VEHICLE', 'PEDESTRIAN'], standardization=standardization)
    attention_radius = dict()
    attention_radius[(env.NodeType.PEDESTRIAN, env.NodeType.PEDESTRIAN)] = 10.0
    attention_radius[(env.NodeType.PEDESTRIAN, env.NodeType.VEHICLE)] = 20.0
    attention_radius[(env.NodeType.VEHICLE, env.NodeType.PEDESTRIAN)] = 20.0
    attention_radius[(env.NodeType.VEHICLE, env.NodeType.VEHICLE)] = 30.0
    env.attention_radius = attention_radius

    train_dataset_path = '/media/cds-k/Data_2/canonical-trn-dev-data/data/train_pb/'
    train_scene_tags_fpath = '/media/cds-k/Data_2/canonical-trn-dev-data/data/train_tags.txt'

    renderer = FeatureRenderer(renderer_config)

    hyperparams = dict()
    hyperparams['minimum_history_length'] = 24
    hyperparams['prediction_horizon'] = 25

    hyperparams['state'] = {"PEDESTRIAN": {"position": ["x", "y"], "velocity": ["x", "y"], "acceleration": ["x", "y"]},
                            "VEHICLE": {"position": ["x", "y"], "velocity": ["x", "y"], "acceleration": ["x", "y"],
                                        "heading": ["\u00b0", "d\u00b0"]}}
    hyperparams['pred_state'] = {"VEHICLE": {"position": ["x", "y"]}, "PEDESTRIAN": {"position": ["x", "y"]}}
    hyperparams['maximum_history_length'] = 24
    node_type_data_set = MotionPredictionDatasetTest(
        dataset_path=train_dataset_path,
        scene_tags_fpath=train_scene_tags_fpath,
        feature_producer=renderer,
        hyperparams=hyperparams,
        node_type='VEHICLE'
    )
    node_type_dataloader = utils.data.DataLoader(node_type_data_set,
                                                 collate_fn=collate,
                                                 pin_memory=True,
                                                 batch_size=32,
                                                 num_workers=6)
    train_data_loader = node_type_dataloader

    # Load model
    log_dir = './models'
    model_name = params['model_name']
    model_dir = os.path.join(log_dir, model_name)
    eval_stg, hyp = load_model(model_dir, env, checkpoint_name=params['checkpoint_name'])

    #
    save_dir = f"{params['save_dir']}/{params['preprocessed_data_dir'].split('processed_')[-1]}/{params['dataset_name']}/model_{model_name}_scene_{scene_idx}_ns_{num_samples}_ph_{ph}_max_h_{max_h}"
    if z_mode:
        save_dir += '_Z'
    if gmm_mode:
        save_dir += '_GMM'
    os.makedirs(f"{save_dir}/bev_maps", exist_ok=True)

    scene = eval_scenes[scene_idx]

    minpos = np.array([scene.x_min, scene.y_min])
    center = (scene.center_x, scene.center_y)
    width = scene.width + 50
    height = scene.height + 50

    my_patch = (center[0] - width / 2, center[1] - height / 2,
                center[0] + width / 2, center[1] + height / 2)

    vis_alpha = 0.1
    if num_samples == 1:
        vis_alpha = 0.5

    node_id2agent_id = dict()
    for t in range(1, scene.timesteps - 1):
        timesteps = np.array([t])
        with torch.no_grad():
            predictions = eval_stg.predict(scene,
                                           timesteps,
                                           ph,
                                           min_future_timesteps=ph,
                                           min_history_timesteps=params['min_h'],
                                           num_samples=num_samples,
                                           z_mode=z_mode,
                                           gmm_mode=gmm_mode,
                                           full_dist=False)

        prediction_dict, histories_dict, futures_dict = \
            prediction_output_to_trajectories(predictions, dt=scene.dt, max_h=max_h, ph=ph, map=None)

        if len(prediction_dict) == 0:
            continue

        v_nodes = list(filter(lambda k: 'VEHICLE' in repr(k), predictions[t].keys()))
        p_nodes = list(filter(lambda k: 'PEDESTRIAN' in repr(k), predictions[t].keys()))

        fig, ax = create_fig(my_patch)

        edge_width = 2
        circle_edge_width = 0.5
        node_circle_size = 0.3
        for idx, node in enumerate(p_nodes):
            player_future = futures_dict[t][node]
            player_past = histories_dict[t][node]
            player_predict = prediction_dict[t][node]
            player_future += minpos
            player_past += minpos
            player_predict += minpos

            node_id = repr(node).split('/')[1]
            if 'nuscenes' in params['save_dir']:
                if node_id not in node_id2agent_id:
                    node_id2agent_id[node_id] = len(node_id2agent_id) + 1
                    agent_id = node_id2agent_id[node_id]
                else:
                    agent_id = node_id2agent_id[node_id]
            else:
                agent_id = int(node_id)

            agent_color = COLORS_ped[agent_id % NCOLORS_ped]

            # Current Node Position
            circle = plt.Circle((player_past[-1, 0], player_past[-1, 1]),
                                node_circle_size,
                                facecolor=agent_color,
                                edgecolor='k',
                                lw=circle_edge_width,
                                zorder=3)
            ax.add_artist(circle)

            for i, row in enumerate(player_predict[0]):
                ax.plot(row[:, 0], row[:, 1],
                        marker='o', color=agent_color,
                        linewidth=1, alpha=vis_alpha, markersize=4)

            ax.plot(player_future[:, 0],
                    player_future[:, 1],
                    'k--',
                    zorder=650,
                    path_effects=[pe.Stroke(linewidth=edge_width, foreground='k'), pe.Normal()])

            ax.plot(player_past[:, 0],
                    player_past[:, 1],
                    'k--',
                    zorder=650,
                    path_effects=[pe.Stroke(linewidth=edge_width, foreground='k'), pe.Normal()])

        for idx, node in enumerate(v_nodes):
            player_future = futures_dict[t][node]
            player_past = histories_dict[t][node]
            player_predict = prediction_dict[t][node]
            player_future += minpos
            player_predict += minpos
            player_past += minpos

            node_id = repr(node).split('/')[1]
            if 'nuscenes' in params['save_dir']:
                if node_id not in node_id2agent_id:
                    node_id2agent_id[node_id] = len(node_id2agent_id) + 1
                    agent_id = node_id2agent_id[node_id]
                else:
                    agent_id = node_id2agent_id[node_id]
            else:
                agent_id = int(node_id)
            agent_car = CAR_IMAGES[agent_id % NCOLORS]
            agent_color = COLORS[agent_id % NCOLORS]

            heading = node.get(np.array([t]), {'heading': ['°']})[0, 0] * 180 / np.pi

            r_img = rotate(agent_car, heading, reshape=True)
            oi = OffsetImage(r_img, zoom=0.0053, zorder=700)
            veh_box = AnnotationBbox(oi, (player_past[-1, 0], player_past[-1, 1]), frameon=False)
            veh_box.zorder = 700
            ax.add_artist(veh_box)

            for i, row in enumerate(player_predict[0]):
                ax.plot(row[:, 0], row[:, 1],
                        marker='o', color=agent_color,
                        linewidth=1, alpha=vis_alpha, markersize=4)

            ax.plot(player_future[:, 0], player_future[:, 1],
                    marker='s', color=agent_color, alpha=0.6,
                    linewidth=1, markersize=8, markerfacecolor='none')

            ax.plot(player_past[:, 0], player_past[:, 1],
                    marker='d', color=agent_color, alpha=0.3,
                    linewidth=1, markersize=8, markerfacecolor='none')

        plt.grid()
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        plt.title(f"dataset: {params['dataset_name']}\nmodel: {model_name}\nscene: {scene_idx}\nph:{ph}\nmax_h:{max_h}")
        plt.savefig(f"{save_dir}/bev_maps/{t:02d}.png", bbox_inches='tight')
        fig.clf()
        plt.close()
        del player_future, player_past, player_predict, predictions
        gc.collect()

        print(f"\nScene: {scene_idx}, time step: {t}\n")


if __name__ == "__main__":
    config_filename = 'sdc'
    with open(f'inference_configs/{config_filename}.yaml', 'r') as file:
        params = yaml.load(file, yaml.Loader)
    main(params)
