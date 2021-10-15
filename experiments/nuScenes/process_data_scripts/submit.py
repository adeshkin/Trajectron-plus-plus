import json
from torch import utils
import tqdm
import os
import sys
import numpy as np
import torch

from ysdc_dataset_api.evaluation import Submission, ObjectPrediction, trajectory_array_to_proto, WeightedTrajectory, \
    save_submission_proto
from ysdc_dataset_api.dataset import MotionPredictionDatasetTrain, MotionPredictionDatasetTest
from ysdc_dataset_api.features import FeatureRenderer

sys.path.append("/home/cds-k/Desktop/motion_prediction/Trajectron-plus-plus/trajectron")
from environment import Environment
from model.dataset import collate_sdc_test
from model.model_registrar import ModelRegistrar
from model import Trajectron

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


def filter_moscow_no_precipitation_data(scene_tags_dict):
    if (scene_tags_dict['track'] == 'Moscow' and
            scene_tags_dict['precipitation'] == 'kNoPrecipitation'):
        return True
    else:
        return False


def filter_ood_validation_data(scene_tags_dict):
    if (scene_tags_dict['track'] in ['Skolkovo', 'Modiin', 'Innopolis'] and
            scene_tags_dict[
                'precipitation'] in ['kNoPrecipitation', 'kRain', 'kSnow']):
        return True
    else:
        return False


def load_model(model_dir, env, checkpoint=None):
    model_registrar = ModelRegistrar(model_dir, 'cuda')
    model_registrar.load_models(checkpoint)
    with open(os.path.join(model_dir, 'config.json'), 'r') as config_json:
        hyperparams = json.load(config_json)

    hyperparams['map_enc_dropout'] = 0.0
    if 'incl_robot_node' not in hyperparams:
        hyperparams['incl_robot_node'] = False

    stg = Trajectron(model_registrar, hyperparams, None, 'cuda')

    stg.set_environment(env)

    stg.set_annealing_params()
    return stg, hyperparams


def main():
    submission = Submission()

    validation_dataset_path = '/media/cds-k/Data_2/canonical-trn-dev-data/data/validation_pb/'
    prerendered_dataset_path = None
    scene_tags_fpath = '/media/cds-k/Data_2/canonical-trn-dev-data/data/validation_tags.txt'
    model_dir = '../models/models_13_Oct_2021_15_36_40_int_ee_sdc_ph_25_maxhl_24_min_hl_24_map_bs_32'
    checkpoint = 'ep_1_step_16000'
    ph = 25

    with open('../train_configs/config_ph_25_maxhl_24_minhl_24_map.json', 'r', encoding='utf-8') as conf_json:
        hyperparams = json.load(conf_json)
    # hyperparams['use_map_encoding'] =
    env = Environment(node_type_list=['VEHICLE', 'PEDESTRIAN'], standardization=standardization)
    attention_radius = dict()
    attention_radius[(env.NodeType.PEDESTRIAN, env.NodeType.PEDESTRIAN)] = 10.0
    attention_radius[(env.NodeType.PEDESTRIAN, env.NodeType.VEHICLE)] = 20.0
    attention_radius[(env.NodeType.VEHICLE, env.NodeType.PEDESTRIAN)] = 20.0
    attention_radius[(env.NodeType.VEHICLE, env.NodeType.VEHICLE)] = 30.0
    env.attention_radius = attention_radius

    node_type = env.NodeType[0]
    node_types = [node_type]

    renderer = FeatureRenderer(renderer_config)

    moscow_validation_dataset = MotionPredictionDatasetTest(
        dataset_path=validation_dataset_path,
        prerendered_dataset_path=prerendered_dataset_path,
        scene_tags_fpath=scene_tags_fpath,
        feature_producer=renderer,
        scene_tags_filter=filter_moscow_no_precipitation_data,
        hyperparams=hyperparams,
        node_type=node_type
    )
    # dataset_iter = iter(moscow_validation_dataset)

    ood_validation_dataset = MotionPredictionDatasetTest(
        dataset_path=validation_dataset_path,
        prerendered_dataset_path=prerendered_dataset_path,
        scene_tags_fpath=scene_tags_fpath,
        feature_producer=renderer,
        scene_tags_filter=filter_ood_validation_data,
        hyperparams=hyperparams,
        node_type=node_type
    )

    moscow_validation_dataloader = utils.data.DataLoader(moscow_validation_dataset,
                                                         collate_fn=collate_sdc_test,
                                                         pin_memory=True,
                                                         batch_size=32,
                                                         num_workers=1)
    #for batch in moscow_validation_dataloader:
    #    data_item = batch
    #    break
    ood_validation_dataloader = utils.data.DataLoader(ood_validation_dataset,
                                                      collate_fn=collate_sdc_test,
                                                      pin_memory=True,
                                                      batch_size=32,
                                                      num_workers=1)

    eval_stg, hyp = load_model(model_dir, env, checkpoint=checkpoint)

    for dataset_key, is_ood, dataloader in zip(
            ['ood_validation', 'moscow_validation'],
            [True, False],
            [ood_validation_dataloader, moscow_validation_dataloader]):
        for batch_id, batch in enumerate(tqdm.tqdm(dataloader)):
            with torch.no_grad():
                predictions = eval_stg.predict_batch(batch,
                                                     ph,
                                                     node_types,
                                                     num_samples=1,
                                                     gmm_mode=True,
                                                     full_dist=False,
                                                     all_z_sep=True)

            for result in predictions:
                d1 = {'scene_id': 'sdgffsdg', 'track_id': 123, 'trajs': np.ones((24, 25, 2)),
                      'probs': np.random.uniform(0, 1, 24)}
                d2 = {'scene_id': 'fsdg', 'track_id': 13, 'trajs': np.ones((24, 25, 2)),
                      'probs': np.random.uniform(0, 1, 24)}
                #np.save('d.npy', d)
                #dl = np.load('d.npy', allow_pickle=True)

            for result in predictions:
                pred = ObjectPrediction()
                pred.track_id = int(result['track_id'])
                pred.scene_id = result['scene_id']
                for k, traj in enumerate(result['trajs']):
                    weight = result['weights'][k]
                    pred.weighted_trajectories.append(WeightedTrajectory(
                        trajectory=trajectory_array_to_proto(traj),
                        weight=weight,
                    ))
                pred.uncertainty_measure = result['U']
                pred.is_ood = is_ood

                submission.predictions.append(pred)

    save_submission_proto('../submissions/dev_moscow_and_ood_submission_map_1_16000.pb', submission=submission)


if __name__ == '__main__':
    main()
