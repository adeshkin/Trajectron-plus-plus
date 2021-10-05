import json
from torch import utils
import tqdm
import os
import sys
import numpy as np
import torch


from ysdc_dataset_api.evaluation import Submission, object_prediction_from_model_output, save_submission_proto
from ysdc_dataset_api.dataset import MotionPredictionDataset

sys.path.append("/home/cds-k/Desktop/motion_prediction/Trajectron-plus-plus/trajectron")
from environment import Environment
from model.dataset import collate
from model.model_registrar import ModelRegistrar
from model import Trajectron
from utils import prediction_output_to_trajectories

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


def main():
    submission = Submission()

    validation_dataset_path = '/media/cds-k/Data_2/canonical-trn-dev-data/data/validation_pb/'
    prerendered_dataset_path = None
    scene_tags_fpath = '/media/cds-k/Data_2/canonical-trn-dev-data/data/validation_tags.txt'
    model_dir = '../models/models_05_Oct_2021_10_30_29_int_ee_sdc_ph_25_maxhl_24_min_hl_24_bs_64'
    model_epoch = 1000

    with open('../train_configs/config_ph_25_maxhl_24_minhl_24.json', 'r', encoding='utf-8') as conf_json:
        hyperparams = json.load(conf_json)

    env = Environment(node_type_list=['VEHICLE', 'PEDESTRIAN'], standardization=standardization)
    attention_radius = dict()
    attention_radius[(env.NodeType.PEDESTRIAN, env.NodeType.PEDESTRIAN)] = 10.0
    attention_radius[(env.NodeType.PEDESTRIAN, env.NodeType.VEHICLE)] = 20.0
    attention_radius[(env.NodeType.VEHICLE, env.NodeType.PEDESTRIAN)] = 20.0
    attention_radius[(env.NodeType.VEHICLE, env.NodeType.VEHICLE)] = 30.0
    env.attention_radius = attention_radius

    node_type = env.NodeType[0]

    moscow_validation_dataset = MotionPredictionDataset(
        dataset_path=validation_dataset_path,
        prerendered_dataset_path=prerendered_dataset_path,
        scene_tags_fpath=scene_tags_fpath,
        scene_tags_filter=filter_moscow_no_precipitation_data,
        hyperparams=hyperparams,
        node_type=node_type
    )

    ood_validation_dataset = MotionPredictionDataset(
        dataset_path=validation_dataset_path,
        prerendered_dataset_path=prerendered_dataset_path,
        scene_tags_fpath=scene_tags_fpath,
        scene_tags_filter=filter_ood_validation_data,
        hyperparams=hyperparams,
        node_type=node_type
    )

    moscow_validation_dataloader = utils.data.DataLoader(moscow_validation_dataset,
                                                         collate_fn=collate,
                                                         pin_memory=True,
                                                         batch_size=64,
                                                         num_workers=10)

    ood_validation_dataloader = utils.data.DataLoader(ood_validation_dataset,
                                                      collate_fn=collate,
                                                      pin_memory=True,
                                                      batch_size=64,
                                                      num_workers=1)

    eval_stg, hyp = load_model(model_dir, env, ts=model_epoch)

    for dataset_key, is_ood, dataloader in zip(
            ['ood_validation', 'moscow_validation'],
            [True, False],
            [ood_validation_dataloader, moscow_validation_dataloader]):
        for batch_id, batch in enumerate(tqdm.tqdm(dataloader)):
            timesteps = np.array([24])
            with torch.no_grad():
                predictions = eval_stg.predict(batch,
                                               timesteps,
                                               25,
                                               min_future_timesteps=25,
                                               min_history_timesteps=24,
                                               num_samples=1,
                                               z_mode=True,
                                               gmm_mode=True,
                                               full_dist=False)

            for result in predictions:
                traj1 = {'trajectory': result['traj'].tolist(),
                         'weight': 1.0}

                proto = object_prediction_from_model_output(
                    track_id=result['track_id'],
                    scene_id=result['scene_id'],
                    weighted_trajectories=[traj1],
                    uncertainty_measure=100,
                    is_ood=is_ood)

                submission.predictions.append(proto)

            save_submission_proto('dev_moscow_and_ood_submission_1000.pb', submission=submission)


if __name__ == '__main__':
    main()
