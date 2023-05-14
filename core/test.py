import os
import ray
import time
import torch

import numpy as np
import core.ctree.cytree as cytree

from tqdm.auto import tqdm
from torch.cuda.amp import autocast as autocast
from core.mcts import MCTS
from core.game import GameHistory
from core.utils import select_action, prepare_observation_lst


@ray.remote(num_gpus=0.125)
def _test(config, shared_storage):
    test_model = config.get_uniform_network()
    best_test_score = float('-inf')
    episodes = 0
    while True:
        counter = ray.get(shared_storage.get_counter.remote())
        if counter >= config.training_steps + config.last_steps:
            time.sleep(30)
            break
        if counter >= config.test_interval * episodes:
            episodes += 1
            test_model.set_weights(ray.get(shared_storage.get_weights.remote()))
            test_model.eval()

            test_score, eval_steps, _ = test(config, test_model, counter, config.test_episodes, config.device, False, save_video=False)
            mean_score = test_score.mean()
            std_score = test_score.std()
            print('Start evaluation at step {}.'.format(counter))
            if mean_score >= best_test_score:
                best_test_score = mean_score
                torch.save(test_model.state_dict(), config.model_path)

            test_log = {
                'mean_score': mean_score,
                'std_score': std_score,
                'max_score': test_score.max(),
                'min_score': test_score.min(),
            }

            shared_storage.add_test_log.remote(counter, test_log)
            print('Training step {}, test scores: \n{} of {} eval steps.'.format(counter, test_score, eval_steps))

        time.sleep(30)


def test(config, model, counter, test_episodes, device, render, save_video=False, final_test=False, use_pb=False):
    """evaluation test
    Parameters
    ----------
    model: any
        models for evaluation
    counter: int
        current training step counter
    test_episodes: int
        number of test episodes
    device: str
        'cuda' or 'cpu'
    render: bool
        True -> render the image during evaluation
    save_video: bool
        True -> save the videos during evaluation
    final_test: bool
        True -> this test is the final test, and the max moves would be 108k/skip
    use_pb: bool
        True -> use tqdm bars
    """
    model.to(device)
    model.eval()
    save_path = os.path.join(config.exp_path, 'recordings', 'step_{}'.format(counter))
    print("testing...")
    with torch.no_grad():
        # new games
        
        env = config.new_game(seed=0, save_video=save_video, save_path=save_path, test=True, final_test=final_test,
                              video_callable=lambda episode_id: True, uid=0)
        
        player = env.live_agents[0]
        
        max_episode_steps = config.max_moves
        
        if use_pb:
            pb = tqdm(np.arange(max_episode_steps), leave=True)
        ep_ori_rewards = np.zeros(test_episodes)
        ep_clip_rewards = np.zeros(test_episodes)
        for i in range(test_episodes):
            print("test match "+str(i+1) +" of "+str(test_episodes))
            # initializations
            obs, taking_actions, action_masks = env.reset()
    
            game_history = GameHistory(env.action_space_size(), max_length=max_episode_steps, config=config)    
    
            game_history.init([obs[player] for _ in range(config.stacked_observations)])
    
            step = 0
            done = False
            # loop
            while not done:
                if config.image_based:
                    stack_obs = []
                    stack_obs.append(game_history.step_obs())
                    stack_obs = prepare_observation_lst(stack_obs)
                    stack_obs = torch.from_numpy(stack_obs).to(device).float()
                else:
                    stack_obs = [game_history.step_obs()]
                    stack_obs = torch.from_numpy(np.array(stack_obs)).to(device)
                with autocast():
                    network_output = model.initial_inference(stack_obs.float())
                hidden_state_roots = network_output.hidden_state
                reward_hidden_roots = network_output.reward_hidden
                value_prefix_pool = network_output.value_prefix
                policy_logits_pool = network_output.policy_logits.tolist()
    
                roots = cytree.Roots(1, config.action_space_size, config.num_simulations)
                roots.prepare_no_noise(value_prefix_pool, policy_logits_pool)
                # do MCTS for a policy (argmax in testing)
                MCTS(config).search(roots, model, hidden_state_roots, reward_hidden_roots)
                roots_distributions = roots.get_distributions()
                roots_values = roots.get_values()
                

                distributions, value = roots_distributions[0], roots_values[0]
                # select the argmax, not sampling
                actions = {}
                actions[player], _ = select_action(distributions, temperature=1, deterministic=True)
                for p in env.live_agents:
                    if p != player:
                        actions[p] = np.random.randint(0, env.action_space_size())

                obs, ori_reward, taking_actions, done_dict, action_masks = env.step(actions)

                done = done_dict[player]
                if done:
                    obs = np.zeros((48, 10, 128)).astype(int)
                else:
                    obs = obs[player]

                if config.clip_reward:
                    clip_reward = np.sign(ori_reward[player])
                else:
                    clip_reward = ori_reward[player]
                game_history.store_search_stats(distributions, value)
                game_history.append(actions[player], obs, clip_reward)
                
                ep_ori_rewards[i] += ori_reward[player]
                ep_clip_rewards[i] += clip_reward
    
                step += 1
                if use_pb:
                    pb.set_description('{} In step {}, scores: {}(max: {}, min: {}) currently.'
                                       ''.format(config.env_name, counter,
                                                 ep_ori_rewards.mean(), ep_ori_rewards.max(), ep_ori_rewards.min()))
                    pb.update(1)
    
            #env.close()
    print("testing complete")
    return ep_ori_rewards, step, save_path
