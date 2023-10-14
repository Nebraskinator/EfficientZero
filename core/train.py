import os
import ray
import time
import torch

import numpy as np
import torch.optim as optim
import torch.nn.functional as F

import copy
import gc

from torch.nn import L1Loss
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler
from core.log import _log
from core.test import _test
from core.replay_buffer import ReplayBuffer
from ray.util.queue import Queue
from core.storage import SharedStorage#, QueueStorage
from core.selfplay_worker import DataWorkerSpawner
from core.reanalyze_worker import BatchWorker_GPU, BatchWorker_CPU

def consist_loss_func(f1, f2):
    """Consistency loss function: similarity loss
    Parameters
    """
    f1 = F.normalize(f1, p=2., dim=-1, eps=1e-5)
    f2 = F.normalize(f2, p=2., dim=-1, eps=1e-5)
    return -(f1 * f2).sum(dim=1)


def adjust_lr(config, optimizer, step_count):
    # adjust learning rate, step lr every lr_decay_steps
    if step_count < config.lr_warm_step:
        lr = config.lr_init * step_count / config.lr_warm_step
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        lr = config.lr_init * config.lr_decay_rate ** ((step_count - config.lr_warm_step) // config.lr_decay_steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return lr

def check_tensor(tensor, name):
    if torch.isnan(tensor).any():
        print(f"{name} contains nan")
    if torch.isinf(tensor).any():
        print(f"{name} contains inf")
                

def update_weights(model, batch, optimizer, replay_buffer, config, scaler, vis_result=False):
    """update models given a batch data
    Parameters
    ----------
    model: Any
        EfficientZero models
    batch: Any
        a batch data inlcudes [inputs_batch, targets_batch]
    replay_buffer: Any
        replay buffer
    scaler: Any
        scaler for torch amp
    vis_result: bool
        True -> log some visualization data in tensorboard (some distributions, values, etc)
    """
    
    curr_model, inputs_batch, targets_batch = batch
    obs_batch_ori, action_batch, mask_batch, indices, weights_lst, make_time = inputs_batch
    target_value_prefix, target_value, target_policy = targets_batch

    # no augmentation needed when using cross-attention
    #if config.use_augmentation and config.augmentation[0] == 'tft': 
    #    obs_batch_ori = config.tft_augmentation(obs_batch_ori)
    # [:, 0: config.stacked_observations * 3,:,:]
    # obs_batch_ori is the original observations in a batch
    # obs_batch is the observation for hat s_t (predicted hidden states from dynamics function)
    # obs_target_batch is the observations for s_t (hidden states from representation function)
    # to save GPU memory usage, obs_batch_ori contains (stack + unroll steps) frames
    obs_batch_ori = torch.from_numpy(obs_batch_ori).to(config.device).float()
    #obs_batch = obs_batch_ori[:, 0: config.stacked_observations * config.image_channel, :, :]
    #obs_target_batch = obs_batch_ori[:, config.image_channel:, :, :]
    obs_batch = torch.reshape(obs_batch_ori[:, 0:config.stacked_observations, :, :, :, :], (obs_batch_ori.shape[0],config.stacked_observations*obs_batch_ori.shape[2], *obs_batch_ori.shape[3:]))
    obs_target_batch = obs_batch_ori[:, 1:, :, :, :, :]

    # do augmentations
    #if config.use_augmentation and config.augmentation[0] != 'tft':
    #    obs_batch = config.transform(obs_batch)
    #    obs_target_batch = config.transform(obs_target_batch)

    # use GPU tensor
    action_batch = torch.from_numpy(action_batch).to(config.device).unsqueeze(-1).long()
    mask_batch = torch.from_numpy(mask_batch).to(config.device).float()
    target_value_prefix = torch.from_numpy(target_value_prefix).to(config.device).float()
    target_value = torch.from_numpy(target_value).to(config.device).float()
    target_policy = torch.from_numpy(target_policy).to(config.device).float()
    weights = torch.from_numpy(weights_lst).to(config.device).float()

    batch_size = obs_batch.size(0)
    assert batch_size == config.batch_size == target_value_prefix.size(0)
    metric_loss = torch.nn.L1Loss()

    # some logs preparation
    other_log = {}
    other_dist = {}

    other_loss = {
        'l1': -1,
        'l1_1': -1,
        'l1_-1': -1,
        'l1_0': -1,
    }
    for i in range(config.num_unroll_steps):
        key = 'unroll_' + str(i + 1) + '_l1'
        other_loss[key] = -1
        other_loss[key + '_1'] = -1
        other_loss[key + '_-1'] = -1
        other_loss[key + '_0'] = -1

    # transform targets to categorical representation
    transformed_target_value_prefix = config.scalar_transform(target_value_prefix)
    target_value_prefix_phi = config.reward_phi(transformed_target_value_prefix)

    transformed_target_value = config.scalar_transform(target_value)
    target_value_phi = config.value_phi(transformed_target_value)

    if config.amp_type == 'torch_amp':
        with autocast():
            value, _, policy_logits, hidden_state, reward_hidden, _, _ = model.initial_inference(obs_batch)
    else:
        value, _, policy_logits, hidden_state, reward_hidden, _, _ = model.initial_inference(obs_batch)
    scaled_value = config.inverse_value_transform(value)

    if vis_result:
        state_lst = hidden_state.detach().cpu().numpy()

    # Check for nan or inf in inputs
    check_tensor(value, "value initial")
    check_tensor(policy_logits, "policy initial")
    check_tensor(hidden_state, "hidden state initial")
    #check_tensor(reward_hidden, "reward hidden initial")

    predicted_value_prefixs = []
    # Note: Following line is just for logging.
    if vis_result:
        predicted_values, predicted_policies = scaled_value.detach().cpu(), torch.softmax(policy_logits, dim=1).detach().cpu()

    # calculate the new priorities for each transition
    value_priority = L1Loss(reduction='none')(scaled_value.squeeze(-1), target_value[:, 0])
    value_priority = value_priority.data.cpu().numpy() + config.prioritized_replay_eps

    # loss of the first step
    value_loss = config.scalar_value_loss(value, target_value_phi[:, 0])
    #config.scalar_value_loss(value, target_value_phi[:, 0])
    policy_loss = -(torch.log_softmax(policy_logits, dim=1) * target_policy[:, 0]).sum(1)
    #-(torch.log_softmax(policy_logits, dim=1) * target_policy[:, 0]).sum(1)
    value_prefix_loss = torch.zeros(batch_size, device=config.device)
    consistency_loss = torch.zeros(batch_size, device=config.device)
    commitment_loss = torch.zeros(batch_size, device=config.device)
    chance_loss = torch.zeros(batch_size, device=config.device)

    target_value_prefix_cpu = target_value_prefix.detach().cpu()
    gradient_scale = 1 / config.num_unroll_steps
    # loss of the unrolled steps
    if config.amp_type == 'torch_amp':
        # use torch amp
        with autocast():
            for step_i in range(config.num_unroll_steps):
                # unroll with the dynamics functions
                
                # predict the afterstate
                afterstate_value, afterstate_value_prefix, chance_token_logits, hidden_afterstate, reward_hidden, _, _ = model.recurrent_afterstate_inference(hidden_state, reward_hidden, action_batch[:, step_i])

                #beg_index = config.image_channel * step_i
                #end_index = config.image_channel * (step_i + config.stacked_observations)

                # obtain the oracle hidden states and chance outcomes from representation function
                _, _, _, presentation_state, _, chance_token_onehot, chance_token_softmax = model.initial_inference(torch.reshape(obs_target_batch[:, i:i+config.stacked_observations, :, :, :, :], (obs_batch_ori.shape[0],config.stacked_observations*obs_batch_ori.shape[2], *obs_batch_ori.shape[3:])))
                
                #chance_token_onehot, chance_token_softmax = model.encoder_network(torch.reshape(obs_target_batch[:, i:i+config.stacked_observations, :, :, :, :], (obs_batch_ori.shape[0],config.stacked_observations*obs_batch_ori.shape[2], *obs_batch_ori.shape[3:])))

                # predict the state
                state_value, state_value_prefix, policy_logits, hidden_state, reward_hidden, _, _ = model.recurrent_state_inference(hidden_afterstate, reward_hidden, chance_token_onehot)              
                              
                # no grad for the presentation_state branch
                dynamic_proj = model.project(hidden_state, with_grad=True)
                observation_proj = model.project(presentation_state, with_grad=False)
                temp_loss = consist_loss_func(dynamic_proj, observation_proj) * mask_batch[:, step_i]

                other_loss['consist_' + str(step_i + 1)] = temp_loss.mean().item()
                consistency_loss += temp_loss
                
                policy_loss += -(torch.log_softmax(policy_logits, dim=1) * target_policy[:, step_i + 1]).sum(1) * mask_batch[:, step_i]
                #
                
                chance_logit_softmax = torch.nn.Softmax(dim=-1)(chance_token_logits)
                chance_loss += ((chance_token_onehot + 1e-5)*(torch.log(chance_token_onehot + 1e-5)-torch.log(chance_logit_softmax + 1e-5))).sum(1) * mask_batch[:, step_i]
                #-(torch.log_softmax(chance_token_logits, dim=1) * (chance_token_onehot)).sum(1)
                value_loss += config.scalar_value_loss(afterstate_value, target_value_phi[:, step_i + 1]) * mask_batch[:, step_i]
                #
                value_loss += config.scalar_value_loss(state_value, target_value_phi[:, step_i + 1]) * mask_batch[:, step_i]
                #
                value_prefix_loss += config.scalar_reward_loss(afterstate_value_prefix, target_value_prefix_phi[:, step_i]) * mask_batch[:, step_i]
                #
                value_prefix_loss += config.scalar_reward_loss(state_value_prefix, target_value_prefix_phi[:, step_i]) * mask_batch[:, step_i]
                #
                
                commitment_loss += ((chance_token_onehot + 1e-5)*(torch.log(chance_token_onehot + 1e-5)-torch.log(chance_token_softmax + 1e-5))).sum(1) * mask_batch[:, step_i]
                #

                #((chance_token_onehot)*(torch.log(chance_token_onehot)-torch.log(chance_token_softmax))).sum(1)
                # Follow MuZero, set half gradient
                hidden_state.register_hook(lambda grad: grad * 0.5)

                # reset hidden states
                if (step_i + 1) % config.lstm_horizon_len == 0:
                    reward_hidden = (torch.zeros(1, config.batch_size, config.lstm_hidden_size).to(config.device),
                                     torch.zeros(1, config.batch_size, config.lstm_hidden_size).to(config.device))

                if vis_result:
                    scaled_value_prefixs = config.inverse_reward_transform(state_value_prefix.detach())
                    scaled_value_prefixs_cpu = scaled_value_prefixs.detach().cpu()

                    predicted_values = torch.cat((predicted_values, config.inverse_value_transform(value).detach().cpu()))
                    predicted_value_prefixs.append(scaled_value_prefixs_cpu)
                    predicted_policies = torch.cat((predicted_policies, torch.softmax(policy_logits, dim=1).detach().cpu()))
                    state_lst = np.concatenate((state_lst, hidden_state.detach().cpu().numpy()))

                    key = 'unroll_' + str(step_i + 1) + '_l1'

                    value_prefix_indices_0 = (target_value_prefix_cpu[:, step_i].unsqueeze(-1) == 0)
                    value_prefix_indices_n1 = (target_value_prefix_cpu[:, step_i].unsqueeze(-1) == -1)
                    value_prefix_indices_1 = (target_value_prefix_cpu[:, step_i].unsqueeze(-1) == 1)

                    target_value_prefix_base = target_value_prefix_cpu[:, step_i].reshape(-1).unsqueeze(-1)

                    other_loss[key] = metric_loss(scaled_value_prefixs_cpu, target_value_prefix_base)
                    if value_prefix_indices_1.any():
                        other_loss[key + '_1'] = metric_loss(scaled_value_prefixs_cpu[value_prefix_indices_1], target_value_prefix_base[value_prefix_indices_1])
                    if value_prefix_indices_n1.any():
                        other_loss[key + '_-1'] = metric_loss(scaled_value_prefixs_cpu[value_prefix_indices_n1], target_value_prefix_base[value_prefix_indices_n1])
                    if value_prefix_indices_0.any():
                        other_loss[key + '_0'] = metric_loss(scaled_value_prefixs_cpu[value_prefix_indices_0], target_value_prefix_base[value_prefix_indices_0])
    

    else:
        for step_i in range(config.num_unroll_steps):
            # unroll with the dynamics function
            value, value_prefix, policy_logits, hidden_state, reward_hidden = model.recurrent_inference(hidden_state, reward_hidden, action_batch[:, step_i])

            #beg_index = config.image_channel * step_i
            #end_index = config.image_channel * (step_i + config.stacked_observations)

            # consistency loss
            if config.consistency_coeff > 0:
                # obtain the oracle hidden states from representation function
                _, _, _, presentation_state, _, chance_token_onehot, chance_token_softmax = model.initial_inference(obs_target_batch[:, i, :, :, :])
                # no grad for the presentation_state branch
                dynamic_proj = model.project(hidden_state, with_grad=True)
                observation_proj = model.project(presentation_state, with_grad=False)
                temp_loss = consist_loss_func(dynamic_proj, observation_proj) * mask_batch[:, step_i]

                other_loss['consist_' + str(step_i + 1)] = temp_loss.mean().item()
                consistency_loss += temp_loss

            policy_loss += -(torch.log_softmax(policy_logits, dim=1) * target_policy[:, step_i + 1]).sum(1) * mask_batch[:, step_i]
            #-(torch.log_softmax(policy_logits, dim=1) * target_policy[:, step_i + 1]).sum(1) * mask_batch[:, step_i]
            value_loss += config.scalar_value_loss(value, target_value_phi[:, step_i + 1]) * mask_batch[:, step_i]
            #config.scalar_value_loss(value, target_value_phi[:, step_i + 1]) * mask_batch[:, step_i]
            value_prefix_loss += config.scalar_reward_loss(value_prefix, target_value_prefix_phi[:, step_i]) * mask_batch[:, step_i]
            #config.scalar_reward_loss(value_prefix, target_value_prefix_phi[:, step_i]) * mask_batch[:, step_i]
            # Follow MuZero, set half gradient
            hidden_state.register_hook(lambda grad: grad * 0.5)

            # reset hidden states
            if (step_i + 1) % config.lstm_horizon_len == 0:
                reward_hidden = (torch.zeros(1, config.batch_size, config.lstm_hidden_size).to(config.device),
                                 torch.zeros(1, config.batch_size, config.lstm_hidden_size).to(config.device))

            if vis_result:
                scaled_value_prefixs = config.inverse_reward_transform(value_prefix.detach())
                scaled_value_prefixs_cpu = scaled_value_prefixs.detach().cpu()

                predicted_values = torch.cat((predicted_values, config.inverse_value_transform(value).detach().cpu()))
                predicted_value_prefixs.append(scaled_value_prefixs_cpu)
                predicted_policies = torch.cat((predicted_policies, torch.softmax(policy_logits, dim=1).detach().cpu()))
                state_lst = np.concatenate((state_lst, hidden_state.detach().cpu().numpy()))

                key = 'unroll_' + str(step_i + 1) + '_l1'

                value_prefix_indices_0 = (target_value_prefix_cpu[:, step_i].unsqueeze(-1) == 0)
                value_prefix_indices_n1 = (target_value_prefix_cpu[:, step_i].unsqueeze(-1) == -1)
                value_prefix_indices_1 = (target_value_prefix_cpu[:, step_i].unsqueeze(-1) == 1)

                target_value_prefix_base = target_value_prefix_cpu[:, step_i].reshape(-1).unsqueeze(-1)

                other_loss[key] = metric_loss(scaled_value_prefixs_cpu, target_value_prefix_base)
                if value_prefix_indices_1.any():
                    other_loss[key + '_1'] = metric_loss(scaled_value_prefixs_cpu[value_prefix_indices_1], target_value_prefix_base[value_prefix_indices_1])
                if value_prefix_indices_n1.any():
                    other_loss[key + '_-1'] = metric_loss(scaled_value_prefixs_cpu[value_prefix_indices_n1], target_value_prefix_base[value_prefix_indices_n1])
                if value_prefix_indices_0.any():
                    other_loss[key + '_0'] = metric_loss(scaled_value_prefixs_cpu[value_prefix_indices_0], target_value_prefix_base[value_prefix_indices_0])
    # ----------------------------------------------------------------------------------
   
    #print((policy_loss.mean(), chance_loss.mean(), value_loss.mean(), value_prefix_loss.mean(), consistency_loss.mean(), commitment_loss.mean()))

    # weighted loss with masks (some invalid states which are out of trajectory.)
    loss = (config.consistency_coeff * consistency_loss + config.policy_loss_coeff * policy_loss +
            config.value_loss_coeff * value_loss + config.reward_loss_coeff * value_prefix_loss + 
            config.commitment_loss_coeff * commitment_loss + config.chance_loss_coeff * chance_loss)
    weighted_loss = (weights * loss).mean()

    # backward
    parameters = model.parameters()
    if config.amp_type == 'torch_amp':
        with autocast():
            total_loss = weighted_loss
            total_loss.register_hook(lambda grad: grad * gradient_scale)
    else:
        total_loss = weighted_loss
        total_loss.register_hook(lambda grad: grad * gradient_scale)
    optimizer.zero_grad()

    if config.amp_type == 'none':
        total_loss.backward()
    elif config.amp_type == 'torch_amp':
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)

    torch.nn.utils.clip_grad_norm_(parameters, config.max_grad_norm)
    if config.amp_type == 'torch_amp':
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    # ----------------------------------------------------------------------------------
    # update priority
    new_priority = ray.put(value_priority)
    replay_buffer.update_priorities.remote(indices, new_priority, make_time)

    # packing data for logging
    loss_data = (total_loss.item(), weighted_loss.item(), loss.mean().item(), 0, policy_loss.mean().item(),
                 value_prefix_loss.mean().item(), value_loss.mean().item(), consistency_loss.mean(), 
                 chance_loss.mean(), commitment_loss.mean())
    if vis_result:
        #reward_w_dist, representation_mean, dynamic_mean, reward_mean = model.get_params_mean()
        #other_dist['reward_weights_dist'] = reward_w_dist
        #other_log['representation_weight'] = representation_mean
        #other_log['dynamic_weight'] = dynamic_mean
        #other_log['reward_weight'] = reward_mean

        # reward l1 loss
        value_prefix_indices_0 = (target_value_prefix_cpu[:, :config.num_unroll_steps].reshape(-1).unsqueeze(-1) == 0)
        value_prefix_indices_n1 = (target_value_prefix_cpu[:, :config.num_unroll_steps].reshape(-1).unsqueeze(-1) == -1)
        value_prefix_indices_1 = (target_value_prefix_cpu[:, :config.num_unroll_steps].reshape(-1).unsqueeze(-1) == 1)

        target_value_prefix_base = target_value_prefix_cpu[:, :config.num_unroll_steps].reshape(-1).unsqueeze(-1)

        predicted_value_prefixs = torch.stack(predicted_value_prefixs).transpose(1, 0).squeeze(-1)
        predicted_value_prefixs = predicted_value_prefixs.reshape(-1).unsqueeze(-1)
        other_loss['l1'] = metric_loss(predicted_value_prefixs, target_value_prefix_base)
        if value_prefix_indices_1.any():
            other_loss['l1_1'] = metric_loss(predicted_value_prefixs[value_prefix_indices_1], target_value_prefix_base[value_prefix_indices_1])
        if value_prefix_indices_n1.any():
            other_loss['l1_-1'] = metric_loss(predicted_value_prefixs[value_prefix_indices_n1], target_value_prefix_base[value_prefix_indices_n1])
        if value_prefix_indices_0.any():
            other_loss['l1_0'] = metric_loss(predicted_value_prefixs[value_prefix_indices_0], target_value_prefix_base[value_prefix_indices_0])

        td_data = (value_priority, target_value_prefix.detach().cpu().numpy(), target_value.detach().cpu().numpy(),
                   transformed_target_value_prefix.detach().cpu().numpy(), transformed_target_value.detach().cpu().numpy(),
                   target_value_prefix_phi.detach().cpu().numpy(), target_value_phi.detach().cpu().numpy(),
                   predicted_value_prefixs.detach().cpu().numpy(), predicted_values.detach().cpu().numpy(),
                   target_policy.detach().cpu().numpy(), predicted_policies.detach().cpu().numpy(), state_lst,
                   other_loss, other_log, other_dist)
        priority_data = (weights, indices)
    else:
        td_data, priority_data = None, None
    del value_priority
    del new_priority
    gc.collect()
    return loss_data, td_data, priority_data, scaler


def _train(models, target_models, replay_buffers, shared_storage, mcts_storage, batch_storage, config, summary_writer, counter_init=0):
    """training loop
    Parameters
    ----------
    model: Any
        EfficientZero models
    target_model: Any
        EfficientZero models for reanalyzing
    replay_buffer: Any
        replay buffer
    shared_storage: Any
        model storage
    batch_storage: Any
        batch storage (queue)
    summary_writer: Any
        logging for tensorboard
    """
    # ----------------------------------------------------------------------------------
    models = [model.to(config.device) for model in models]
    target_models = [target_model.to(config.device) for target_model in target_models]
    lrs = [config.lr_init for _ in models]
    #optimizers = [optim.SGD(model.parameters(), lr=config.lr_init, momentum=config.momentum,
    #                      weight_decay=config.weight_decay) for model in models]
    optimizers = [optim.AdamW(model.parameters(), lr=config.lr_init,
                          weight_decay=config.weight_decay) for model in models]
    
    scaler = GradScaler()

    [model.train() for model in models]
    [target_model.eval() for target_model in target_models]
    # ----------------------------------------------------------------------------------
    # set augmentation tools
    if config.use_augmentation and config.augmentation[0] != 'tft':
        config.set_transforms()
    # wait until collecting enough data to start
    while not (all([ray.get(replay_buffer.get_total_len.remote()) >= config.start_transitions for replay_buffer in replay_buffers if replay_buffer])):
        time.sleep(5)
        pass
    print('Begin training...')
    # set signals for other workers

    step_counts = [counter_init for _ in models]
    # Note: the interval of the current model and the target model is between x and 2x. (x = target_model_interval)
    # recent_weights is the param of the target model
    recent_weights = [model.get_weights() for model in models]
    
    training_models = [i for i in range(config.num_models) if i not in config.freeze_models]
    curr_model = training_models[0]
    # while loop
    while any([step_counts[m] < config.training_steps + config.last_steps for m in range(config.num_models)]):
        # remove data if the replay buffer is full. (more data settings)
        if step_counts[curr_model] == counter_init:
            shared_storage.set_start_signal.remote(curr_model)
        #if step_counts[curr_model] == 3:
        #    torch.autograd.set_detect_anomaly(True)
        if step_counts[curr_model] % 40 == 0:          
            [replay_buffer.remove_to_fit.remote() for replay_buffer in replay_buffers if replay_buffer]

        # obtain a batch
        if batch_storage.qsize() > 0:
            batch = batch_storage.get()
        else:
            time.sleep(0.3)
            continue
        if batch[0] != curr_model:
            del batch
            gc.collect()
            continue
        
        batch = copy.deepcopy(batch)
        shared_storage.incr_counter.remote(curr_model)
        lrs[curr_model] = adjust_lr(config, optimizers[curr_model], step_counts[curr_model])

        # update model for self-play
        if step_counts[curr_model] % config.checkpoint_interval == 0:
            updated_weights = models[curr_model].get_weights()
            shared_storage.set_weights.remote(updated_weights, curr_model)


        # update model for reanalyzing
        if step_counts[curr_model] % config.target_model_interval == 0:
            shared_storage.set_target_weights.remote(recent_weights[curr_model], curr_model)
            recent_weights[curr_model] = models[curr_model].get_weights()

        if step_counts[curr_model] % config.vis_interval == 0:
            vis_result = True
        else:
            vis_result = False

        if config.amp_type == 'torch_amp':
            log_data = update_weights(models[curr_model], batch, optimizers[curr_model], replay_buffers[curr_model], config, scaler, vis_result)
            scaler = log_data[3]
        else:
            log_data = update_weights(models[curr_model], batch, optimizers[curr_model], replay_buffers[curr_model], config, scaler, vis_result)

        if step_counts[curr_model] % config.log_interval == 0:
            _log(config, step_counts[curr_model], log_data[0:3], models[curr_model], replay_buffers[curr_model], lrs[curr_model], shared_storage, summary_writer, vis_result)

        # The queue is empty.
        if step_counts[curr_model] >= 100 and step_counts[curr_model] % 50 == 0 and batch_storage.qsize() == 0:
            print('Warning: Batch Queue is empty (Require more batch actors Or batch actor fails).')

        step_counts[curr_model] += 1
        #print("training ongoing, step : " + step_count)
        # save models
        if step_counts[curr_model] % config.save_ckpt_interval == 0:
            model_path = os.path.join(config.model_dir, 'model_{}_{}.p'.format(step_counts[curr_model], curr_model))
            torch.save(models[curr_model].state_dict(), model_path)
            
        if step_counts[curr_model] % config.prev_model_update_interval == 0:
            shared_storage.update_previous_models.remote(curr_model)
            
        if step_counts[curr_model] % config.model_switch_interval == 0:
            idx = training_models.index(curr_model)
            idx += 1
            if idx >= len(training_models):
                idx = 0
            curr_model = training_models[idx]
            print('training model '+str(curr_model))
            shared_storage.set_current_model.remote(curr_model)
            
        

    shared_storage.set_weights.remote(models[curr_model].get_weights(), curr_model)
    time.sleep(30)
    return models[curr_model].get_weights()


def train(config, summary_writer, model_path=None):
    """training process
    Parameters
    ----------
    summary_writer: Any
        logging for tensorboard
    model_path: str
        model path for resuming
        default: train from scratch
    """
    models = [config.get_uniform_network() for _ in range(config.num_models)]
    target_models = [config.get_uniform_network() for _ in range(config.num_models)]
    prev_models = [config.get_uniform_network() for _ in range(config.num_prev_models)]
    counter_init = 0
    if model_path:
        for i, path in enumerate(model_path):
            print('resume model from path: ', path)
            weights = torch.load(path)
    
            models[i].load_state_dict(weights)
            target_models[i].load_state_dict(weights)
            for m in prev_models:
                m.load_state_dict(weights)
            try:
                counter_init = int(path.split('_')[-2])
            except:
                continue

    storage = SharedStorage.remote(models, target_models, prev_models, counter_init=counter_init)
    # prepare the batch and mctc context storage
    batch_storage = Queue(maxsize=config.batch_queue_size, actor_options={"num_cpus": 3})
    mcts_storage = Queue(maxsize=config.batch_queue_size, actor_options={"num_cpus": 3})
    #batch_storage = QueueStorage(int(config.batch_queue_size - 1), config.batch_queue_size)
    #mcts_storage = QueueStorage(int(config.mcts_queue_size - 1), config.mcts_queue_size)
    replay_buffers = [ReplayBuffer.remote(config=config) if i not in config.freeze_models else None for i in range(config.num_models)]

    # other workers
    workers = []

    # reanalyze workers
    cpu_workers = [BatchWorker_CPU.remote(idx, replay_buffers, storage, batch_storage, mcts_storage, config) for idx in range(config.cpu_actor)]
    workers += [cpu_worker.run.remote() for cpu_worker in cpu_workers]
    gpu_workers = [BatchWorker_GPU.remote(idx, replay_buffers, storage, batch_storage, mcts_storage, config) for idx in range(config.gpu_actor)]
    workers += [gpu_worker.run.remote() for gpu_worker in gpu_workers]

    # self-play workers
    data_workers = [DataWorkerSpawner.remote(rank, replay_buffers, storage, config, log=rank==0) for rank in range(config.num_actors)]
    for data_worker in data_workers:
        workers += [data_worker.run.remote()]
        time.sleep(30)
    # test workers
    #workers += [_test.remote(config, storage)]

    # training loop
    final_weights = _train(models, target_models, replay_buffers, storage, mcts_storage, batch_storage, config, summary_writer, counter_init=counter_init)

    ray.wait(workers)
    print('Training over...')

    return models, final_weights
