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

sys.path.append("../../trajectron")

from model.model_registrar import ModelRegistrar
from model import Trajectron
from utils import prediction_output_to_trajectories


def load_model(model_dir, env, ts=3999):
    model_registrar = ModelRegistrar(model_dir, 'cuda')
    model_registrar.load_models(ts)
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

    CAR_IMAGES = (plt.imread('icons/Car TOP_VIEW 375397.png'),  # blue
                  plt.imread('icons/Car TOP_VIEW F05F78.png'),  # red
                  plt.imread('icons/Car TOP_VIEW 80CBE5.png'),  # cyan
                  plt.imread('icons/Car TOP_VIEW ABCB51.png'),  # green
                  plt.imread('icons/Car TOP_VIEW C8B0B0.png'))  # gray
    COLORS = ('blue', 'red', 'cyan', 'green', 'gray')
    NCOLORS = len(COLORS)

    # Load dataset
    with open(preprocessed_data_fn, 'rb') as f:
        eval_env = dill.load(f, encoding='latin1')
    eval_scenes = eval_env.scenes

    # Load model
    log_dir = './models'
    model_name = params['model_name']
    model_dir = os.path.join(log_dir, model_name)
    eval_stg, hyp = load_model(model_dir, eval_env, ts=12)

    #
    save_dir = f"{params['save_dir']}/{params['preprocessed_data_dir'].split('processed_')[-1]}/{params['dataset_name']}/model_{model_name}_scene_{scene_idx}_ns_{num_samples}_ph_{ph}_max_h_{max_h}"
    os.makedirs(f"{save_dir}/bev_maps", exist_ok=True)

    scene = eval_scenes[scene_idx]

    minpos = np.array([scene.x_min, scene.y_min])
    center = (scene.center_x, scene.center_y)
    width = scene.width + 50
    height = scene.height + 50

    my_patch = (center[0] - width / 2, center[1] - height / 2,
                center[0] + width / 2, center[1] + height / 2)
    node_id2agent_id = dict()
    for t in range(1, scene.timesteps - 1):
        timesteps = np.array([t])
        with torch.no_grad():
            predictions = eval_stg.predict(scene,
                                           timesteps, ph, num_samples=num_samples,
                                           z_mode=False,
                                           gmm_mode=False,
                                           full_dist=True,
                                           all_z_sep=False)

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

            agent_color = COLORS[agent_id % NCOLORS]

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
                        linewidth=1, alpha=0.1, markersize=4)

            ax.plot(player_future[:, 0],
                    player_future[:, 1],
                    'w--',
                    zorder=650,
                    path_effects=[pe.Stroke(linewidth=edge_width, foreground='k'), pe.Normal()])

            ax.plot(player_past[:, 0],
                    player_past[:, 1],
                    'g--',
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
                        linewidth=1, alpha=0.1, markersize=4)

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
    config_filename = 'centerpoint_out'
    with open(f'inference_configs/{config_filename}.yaml', 'r') as file:
        params = yaml.load(file, yaml.Loader)
    main(params)
