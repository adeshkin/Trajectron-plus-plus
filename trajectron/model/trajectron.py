import torch
import numpy as np
from model.mgcvae import MultimodalGenerativeCVAE
from model.dataset import get_timesteps_data, restore


class Trajectron(object):
    def __init__(self, model_registrar,
                 hyperparams, log_writer,
                 device):
        super(Trajectron, self).__init__()
        self.hyperparams = hyperparams
        self.log_writer = log_writer
        self.device = device
        self.curr_iter = 0

        self.model_registrar = model_registrar
        self.node_models_dict = dict()
        self.nodes = set()

        self.env = None

        self.min_ht = self.hyperparams['minimum_history_length']
        self.max_ht = self.hyperparams['maximum_history_length']
        self.ph = self.hyperparams['prediction_horizon']
        self.state = self.hyperparams['state']
        self.state_length = dict()
        for state_type in self.state.keys():
            self.state_length[state_type] = int(
                np.sum([len(entity_dims) for entity_dims in self.state[state_type].values()])
            )
        self.pred_state = self.hyperparams['pred_state']

    def set_environment(self, env):
        self.env = env

        self.node_models_dict.clear()
        edge_types = env.get_edge_types()
        for node_type in env.NodeType:
            # Only add a Model for NodeTypes we want to predict
            if node_type in self.pred_state.keys():
                self.node_models_dict[node_type] = MultimodalGenerativeCVAE(env,
                                                                            node_type,
                                                                            self.model_registrar,
                                                                            self.hyperparams,
                                                                            self.device,
                                                                            edge_types,
                                                                            log_writer=self.log_writer)

    def set_curr_iter(self, curr_iter):
        self.curr_iter = curr_iter
        for node_str, model in self.node_models_dict.items():
            model.set_curr_iter(curr_iter)

    def set_annealing_params(self):
        for node_str, model in self.node_models_dict.items():
            model.set_annealing_params()

    def step_annealers(self, node_type=None):
        if node_type is None:
            for node_type in self.node_models_dict:
                self.node_models_dict[node_type].step_annealers()
        else:
            self.node_models_dict[node_type].step_annealers()

    def train_loss(self, batch, node_type):
        (first_history_index,
         x_t, y_t, x_st_t, y_st_t,
         neighbors_data_st,
         neighbors_edge_value,
         robot_traj_st_t,
         _), map = batch

        x = x_t.to(self.device)
        y = y_t.to(self.device)
        x_st_t = x_st_t.to(self.device)
        y_st_t = y_st_t.to(self.device)
        if robot_traj_st_t is not None:
            robot_traj_st_t = robot_traj_st_t.to(self.device)
        if type(map) == torch.Tensor:
            map = map.to(self.device)

        # Run forward pass
        model = self.node_models_dict[node_type]
        loss = model.train_loss(inputs=x,
                                inputs_st=x_st_t,
                                first_history_indices=first_history_index,
                                labels=y,
                                labels_st=y_st_t,
                                neighbors=restore(neighbors_data_st),
                                neighbors_edge_value=restore(neighbors_edge_value),
                                robot=robot_traj_st_t,
                                map=map,
                                prediction_horizon=self.ph)

        return loss

    def eval_loss(self, batch, node_type):
        (first_history_index,
         x_t, y_t, x_st_t, y_st_t,
         neighbors_data_st,
         neighbors_edge_value,
         robot_traj_st_t,
         map) = batch

        x = x_t.to(self.device)
        y = y_t.to(self.device)
        x_st_t = x_st_t.to(self.device)
        y_st_t = y_st_t.to(self.device)
        if robot_traj_st_t is not None:
            robot_traj_st_t = robot_traj_st_t.to(self.device)
        if type(map) == torch.Tensor:
            map = map.to(self.device)

        # Run forward pass
        model = self.node_models_dict[node_type]
        nll = model.eval_loss(inputs=x,
                              inputs_st=x_st_t,
                              first_history_indices=first_history_index,
                              labels=y,
                              labels_st=y_st_t,
                              neighbors=restore(neighbors_data_st),
                              neighbors_edge_value=restore(neighbors_edge_value),
                              robot=robot_traj_st_t,
                              map=map,
                              prediction_horizon=self.ph)

        return nll.cpu().detach().numpy()

    def predict_batch(self,
                      batch,
                      ph,
                      node_types,
                      num_samples=1,
                      z_mode=False,
                      gmm_mode=False,
                      full_dist=True,
                      all_z_sep=False):
        results = []
        for node_type in node_types:
            if node_type not in self.pred_state:
                continue

            model = self.node_models_dict[node_type]

            (first_history_index,
             x_t, y_t, x_st_t, y_st_t,
             neighbors_data_st,
             neighbors_edge_value,
             robot_traj_st_t,
             _), map, node_id, scene_id, x_min, y_min = batch

            x = x_t.to(self.device)
            x_st_t = x_st_t.to(self.device)
            if robot_traj_st_t is not None:
                robot_traj_st_t = robot_traj_st_t.to(self.device)
            if type(map) == torch.Tensor:
                map = map.to(self.device)

            # Run forward pass
            probs, predictions = model.predict(inputs=x,
                                               inputs_st=x_st_t,
                                               first_history_indices=first_history_index,
                                               neighbors=restore(neighbors_data_st),
                                               neighbors_edge_value=restore(neighbors_edge_value),
                                               robot=robot_traj_st_t,
                                               map=map,
                                               prediction_horizon=ph,
                                               num_samples=num_samples,
                                               z_mode=z_mode,
                                               gmm_mode=gmm_mode,
                                               full_dist=full_dist,
                                               all_z_sep=all_z_sep)

            num_preds = 3
            B = predictions.shape[1]

            probs = probs.squeeze(dim=1)

            topk_probs = torch.topk(probs, k=num_preds, dim=1).values
            topk_indices = torch.topk(probs, k=num_preds, dim=1).indices

            log_topk_probs = torch.log(topk_probs)
            uncertainty = - torch.mean(log_topk_probs, dim=1)

            best_plans = [predictions[topk_indices[b], b, :, :] for b in range(B)]
            best_probs = torch.nn.functional.softmax(log_topk_probs, dim=1).cpu().detach().numpy()

            print()
            # for k, plan in enumerate(topk_plans):
            #    result = {'trajs': plan.cpu().detach().numpy() + np.array([x_min[k], y_min[k]]),
            #              'track_id': node_id[k],
            #              'scene_id': scene_id[k],
            #              'weights': norm_topk_probs[k],
            #              'U': uncertainty[k]}
            #    results.append(result)

        return results

    def predict(self,
                scene,
                timesteps,
                ph,
                num_samples=1,
                min_future_timesteps=0,
                min_history_timesteps=1,
                z_mode=False,
                gmm_mode=False,
                full_dist=True,
                all_z_sep=False):

        predictions_dict = {}
        for node_type in self.env.NodeType:
            if node_type not in self.pred_state:
                continue

            model = self.node_models_dict[node_type]

            # Get Input data for node type and given timesteps
            batch = get_timesteps_data(env=self.env, scene=scene, t=timesteps, node_type=node_type, state=self.state,
                                       pred_state=self.pred_state, edge_types=model.edge_types,
                                       min_ht=min_history_timesteps, max_ht=self.max_ht, min_ft=min_future_timesteps,
                                       max_ft=min_future_timesteps, hyperparams=self.hyperparams)
            # There are no nodes of type present for timestep
            if batch is None:
                continue
            (first_history_index,
             x_t, y_t, x_st_t, y_st_t,
             neighbors_data_st,
             neighbors_edge_value,
             robot_traj_st_t,
             map), nodes, timesteps_o = batch
            #map = torch.randn((1, 10, 400, 400))
            x = x_t.to(self.device)
            x_st_t = x_st_t.to(self.device)
            if robot_traj_st_t is not None:
                robot_traj_st_t = robot_traj_st_t.to(self.device)
            if type(map) == torch.Tensor:
                map = map.to(self.device)

            # Run forward pass
            probs, predictions = model.predict(inputs=x,
                                               inputs_st=x_st_t,
                                               first_history_indices=first_history_index,
                                               neighbors=neighbors_data_st,
                                               neighbors_edge_value=neighbors_edge_value,
                                               robot=robot_traj_st_t,
                                               map=map,
                                               prediction_horizon=ph,
                                               num_samples=num_samples,
                                               z_mode=z_mode,
                                               gmm_mode=gmm_mode,
                                               full_dist=full_dist,
                                               all_z_sep=all_z_sep)

            predictions_np = predictions.cpu().detach().numpy()
            if np.isnan(predictions_np).any():
                print(predictions_np)
            # Shifts prediction requests
            '''
            if scene.prediction_request_agent_ids:
                mask_agents = np.array(
                    [repr(node).split('/')[1] in scene.prediction_request_agent_ids for node in nodes])
                if True not in mask_agents:
                    continue
                predictions_np = predictions_np[:, mask_agents]
                timesteps_o = np.array(timesteps_o)[mask_agents]
                nodes = np.array(nodes)[mask_agents]
            '''
            # Assign predictions to node
            for i, ts in enumerate(timesteps_o):
                if ts not in predictions_dict.keys():
                    predictions_dict[ts] = dict()
                predictions_dict[ts][nodes[i]] = np.transpose(predictions_np[:, [i]], (1, 0, 2, 3))

        return predictions_dict
