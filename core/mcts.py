import torch

import numpy as np
import core.ctree.cytree as tree

from torch.cuda.amp import autocast as autocast


class MCTS(object):
    def __init__(self, config):
        self.config = config

    def search(self, roots, model, hidden_state_roots, device=None, training_start=True):
        """Do MCTS for the roots (a batch of root nodes in parallel). Parallel in model inference
        Parameters
        ----------
        roots: Any
            a batch of expanded root nodes
        hidden_state_roots: list
            the hidden states of the roots
        reward_hidden_roots: list
            the value prefix hidden states in LSTM of the roots
        """
        with torch.no_grad():
            model.eval()

            # preparation
            num = roots.num
            if device==None:
                device = self.config.device
            pb_c_base, pb_c_init, discount = self.config.pb_c_base, self.config.pb_c_init, self.config.discount
            # the data storage of hidden states: storing the states of all the tree nodes
            # hidden_state_roots is of shape (batch, *hidden_shape).
            # thus the final list indexing is [idx, batch_idx]
            hidden_state_pool = [hidden_state_roots]
            # 1 x batch x 64
            # the data storage of value prefix hidden states in LSTM
            # rewardhidden_roots is a tuple (h, c) where the shape of h and c
            # is (1, batch, hidden_size).
            # thus the final list indexing is [idx, 0, batch_idx] for each hidden
            # state.
            #reward_hidden_c_pool = [reward_hidden_roots[0]]
            #reward_hidden_h_pool = [reward_hidden_roots[1]]
            # the index of each layer in the tree
            hidden_state_index_x = 0
            # minimax value storage
            min_max_stats_lst = tree.MinMaxStatsList(num)
            min_max_stats_lst.set_delta(self.config.value_delta_max)
            horizons = self.config.lstm_horizon_len
            if training_start:
                iters = self.config.num_simulations
                max_moves_considered = self.config.max_moves_considered
            else:
                iters = 1
                max_moves_considered = 1
            for index_simulation in range(iters):
                hs_to_append = [None for _ in range(num)]
                hscr_to_append = [None for _ in range(num)]
                hshr_to_append = [None for _ in range(num)]
                value_prefix_lst = [None for _ in range(num)]
                value_lst = [None for _ in range(num)]
                policy_lst = [None for _ in range(num)]
                hidden_afterstates = []
                #hidden_afterstates_c_reward = []
                #hidden_afterstates_h_reward = []
                hidden_states = []
                #hidden_states_c_reward = []
                #hidden_states_h_reward = []

                # prepare a result wrapper to transport results between python and c++ parts
                results = tree.ResultsWrapper(num)
                
                
                # Stochastic MCTS:
                # Split the predictions into two groups: afterstate and state
                # Predict for each group
                # Collate the predictions so they are in the original order
                # Backpropagate.
                
                # hidden_state_index_x_lst: the iter index of leaf node states in hidden_state_pool
                # hidden_state_index_y_lst: the batch index of leaf node states in hidden_state_pool
                # the hidden state of the leaf node is hidden_state_pool[x, y]; value prefix states are the same
                hidden_state_index_x_lst, hidden_state_index_y_lst, policy_node_y_lst, chance_node_y_lst, last_actions = tree.batch_traverse(roots, iters, max_moves_considered, discount, min_max_stats_lst, results)
                
                states_to_predict = []
                states_to_predict_c_reward = []
                states_to_predict_h_reward = []
                actions_to_predict = []
                
                # the chance nodes expand using chance logits predictions from an afterstate
                for ic in chance_node_y_lst:
                    ix = hidden_state_index_x_lst[ic]
                    iy = hidden_state_index_y_lst[ic]
                    states_to_predict.append(hidden_state_pool[ix][iy])
                    #states_to_predict_c_reward.append(reward_hidden_c_pool[ix][0][iy])
                    #states_to_predict_h_reward.append(reward_hidden_h_pool[ix][0][iy])
                    actions_to_predict.append(last_actions[ic])

                if states_to_predict:
                    states_to_predict = torch.from_numpy(np.asarray(states_to_predict)).to(device).float()
                    #states_to_predict_c_reward = torch.from_numpy(np.asarray(states_to_predict_c_reward)).to(device).unsqueeze(0)
                    #states_to_predict_h_reward = torch.from_numpy(np.asarray(states_to_predict_h_reward)).to(device).unsqueeze(0)

                    actions_to_predict = torch.from_numpy(np.asarray(actions_to_predict)).to(device).unsqueeze(1).long()
                    
                    # evaluation for leaf nodes
                    if self.config.amp_type == 'torch_amp':
                        with autocast():
                            network_output = model.recurrent_afterstate_inference(states_to_predict, (states_to_predict_c_reward, states_to_predict_h_reward), actions_to_predict)
                    else:
                        network_output = model.recurrent_afterstate_inference(states_to_predict, (states_to_predict_c_reward, states_to_predict_h_reward), actions_to_predict)

                    hidden_state_nodes = network_output.hidden_state
                    value_prefix_pool = network_output.value_prefix.reshape(-1).tolist()
                    value_pool = network_output.value.reshape(-1).tolist()
                    policy_logits_pool = network_output.policy_logits.tolist()
                    #reward_hidden_nodes = network_output.reward_hidden
                                       
                    for i, y in enumerate(chance_node_y_lst):
                        hs_to_append[y] = hidden_state_nodes[i]
                        value_prefix_lst[y] = value_prefix_pool[i]
                        value_lst[y] = value_pool[i]
                        policy_lst[y] = policy_logits_pool[i]
                        #hscr_to_append[y] = reward_hidden_nodes[0][:, [i], :]
                        #hshr_to_append[y] = reward_hidden_nodes[1][:, [i], :]
                
                states_to_predict = []
                states_to_predict_c_reward = []
                states_to_predict_h_reward = []
                actions_to_predict = []
                
                # the policy nodes expand using policy logits predictions from a state
                for ic in policy_node_y_lst:
                    ix = hidden_state_index_x_lst[ic]
                    iy = hidden_state_index_y_lst[ic]
                    states_to_predict.append(hidden_state_pool[ix][iy])
                    #states_to_predict_c_reward.append(reward_hidden_c_pool[ix][0][iy])
                    #states_to_predict_h_reward.append(reward_hidden_h_pool[ix][0][iy])
                    actions_to_predict.append(last_actions[ic])
                               
                if states_to_predict:
                    states_to_predict = torch.from_numpy(np.asarray(states_to_predict)).to(device).float()
                    #states_to_predict_c_reward = torch.from_numpy(np.asarray(states_to_predict_c_reward)).to(device).unsqueeze(0)
                    #states_to_predict_h_reward = torch.from_numpy(np.asarray(states_to_predict_h_reward)).to(device).unsqueeze(0)

                    actions_to_predict = torch.from_numpy(np.asarray(actions_to_predict)).to(device).long()
                    actions_to_predict = torch.nn.functional.one_hot(actions_to_predict, self.config.num_chance_tokens)

                    # evaluation for leaf nodes
                    if self.config.amp_type == 'torch_amp':
                        with autocast():
                            network_output = model.recurrent_state_inference(states_to_predict, (states_to_predict_c_reward, states_to_predict_h_reward), actions_to_predict)
                    else:
                        network_output = model.recurrent_state_inference(states_to_predict, (states_to_predict_c_reward, states_to_predict_h_reward), actions_to_predict)

                    hidden_state_nodes = network_output.hidden_state
                    value_prefix_pool = network_output.value_prefix.reshape(-1).tolist()
                    value_pool = network_output.value.reshape(-1).tolist()
                    policy_logits_pool = network_output.policy_logits.tolist()
                    #reward_hidden_nodes = network_output.reward_hidden
                                        
                    for i, y in enumerate(policy_node_y_lst):
                        hs_to_append[y] = hidden_state_nodes[i]
                        value_prefix_lst[y] = value_prefix_pool[i]
                        value_lst[y] = value_pool[i]
                        policy_lst[y] = policy_logits_pool[i]
                        #hscr_to_append[y] = reward_hidden_nodes[0][:, [i], :]
                        #hshr_to_append[y] = reward_hidden_nodes[1][:, [i], :]
                
                hidden_state_pool.append(hs_to_append)
                #hscr_to_append = np.concatenate(hscr_to_append, axis=-2)
                #hshr_to_append = np.concatenate(hshr_to_append, axis=-2)
                
                # reset 0
                # reset the hidden states in LSTM every horizon steps in search
                # only need to predict the value prefix in a range (eg: s0 -> s5)
                assert horizons > 0
                search_lens = results.get_search_len()
                reset_idx = (np.array(search_lens) % horizons == 0)
                assert len(reset_idx) == num
                #hscr_to_append[:, reset_idx, :] = 0
                #hshr_to_append[:, reset_idx, :] = 0
                is_reset_lst = reset_idx.astype(np.int32).tolist()

                #reward_hidden_c_pool.append(hscr_to_append)
                #reward_hidden_h_pool.append(hshr_to_append)
                
                hidden_state_index_x += 1

                #print((hidden_state_index_x, discount, value_prefix_lst, value_lst, policy_lst))

                # backpropagation along the search path to update the attributes
                # the hidden_state_index_y is inferred by the order
                tree.batch_back_propagate(hidden_state_index_x, discount,
                                          value_prefix_lst, value_lst, policy_lst,
                                          min_max_stats_lst, results, is_reset_lst)
