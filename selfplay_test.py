# -*- coding: utf-8 -*-
"""
Created on Tue May  2 20:55:57 2023

@author: ruhe
"""

from config.tft import game_config
from core.selfplay_worker import DataWorker
from core.replay_buffer import ReplayBuffer
from core.storage import SharedStorage, QueueStorage
from core.test import _test
import argparse
import torch
import os
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EfficientZero')
    parser.add_argument('--env', required=True, help='Name of the environment')
    parser.add_argument('--result_dir', default=os.path.join(os.getcwd(), 'results'),
                        help="Directory Path to store results (default: %(default)s)")
    parser.add_argument('--case', required=True, choices=['atari', 'tft'],
                        help="It's used for switching between different domains(default: %(default)s)")
    parser.add_argument('--opr', required=True, choices=['train', 'test'])
    parser.add_argument('--amp_type', required=True, choices=['torch_amp', 'none'],
                        help='choose automated mixed precision type')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='no cuda usage (default: %(default)s)')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='If enabled, logs additional values  '
                             '(gradients, target value, reward distribution, etc.) (default: %(default)s)')
    parser.add_argument('--render', action='store_true', default=False,
                        help='Renders the environment (default: %(default)s)')
    parser.add_argument('--save_video', action='store_true', default=True, help='save video in test.')
    parser.add_argument('--force', action='store_true', default=False,
                        help='Overrides past results (default: %(default)s)')
    parser.add_argument('--cpu_actor', type=int, default=14, help='batch cpu actor')
    # test works uses 0.125 GPU
    parser.add_argument('--gpu_actor', type=int, default=6, help='batch gpu actor (0.125)')
    parser.add_argument('--selfplay_actor', type=int, default=5, help='selfplay gpu actor (0.125)')
    parser.add_argument('--p_mcts_num', type=int, default=8, help='number of parallel mcts')
    parser.add_argument('--seed', type=int, default=0, help='seed (default: %(default)s)')
    parser.add_argument('--num_gpus', type=int, default=1, help='gpus available')
    parser.add_argument('--num_cpus', type=int, default=28, help='cpus available')
    parser.add_argument('--revisit_policy_search_rate', type=float, default=0.99,
                        help='Rate at which target policy is re-estimated (default: %(default)s)')
    parser.add_argument('--use_root_value', action='store_true', default=False,
                        help='choose to use root value in reanalyzing')
    parser.add_argument('--use_priority', action='store_true', default=True,
                        help='Uses priority for data sampling in replay buffer. '
                             'Also, priority for new data is calculated based on loss (default: False)')
    parser.add_argument('--use_max_priority', action='store_true', default=True, help='max priority')
    parser.add_argument('--test_episodes', type=int, default=10, help='Evaluation episode count (default: %(default)s)')
    parser.add_argument('--use_augmentation', action='store_true', default=False, help='use augmentation')
    parser.add_argument('--augmentation', type=str, default=['shift', 'intensity'], nargs='+',
                        choices=['none', 'rrc', 'affine', 'crop', 'blur', 'shift', 'intensity'],
                        help='Style of augmentation')
    parser.add_argument('--info', type=str, default='none', help='debug string')
    parser.add_argument('--load_model', action='store_true', default=False, help='choose to load model')
    parser.add_argument('--model_path', type=str, default='./results/test_model.p', help='load model path')
    parser.add_argument('--object_store_memory', type=int, default=40 * 1024 * 1024 * 1024, help='object store memory')
    
    # Process arguments
    args = parser.parse_args()
    args.device = 'cuda' if (not args.no_cuda) and torch.cuda.is_available() else 'cpu'
    assert args.revisit_policy_search_rate is None or 0 <= args.revisit_policy_search_rate <= 1, \
        ' Revisit policy search rate should be in [0,1]'
    
    game_config.set_game('a')
    exp_path = game_config.set_config(args)
    
   
    models = [game_config.get_uniform_network() for _ in range(game_config.num_models)]
    target_models = [game_config.get_uniform_network() for _ in range(game_config.num_models)]
    #print(models[0])
    if args.load_model:
        model_path = [args.model_path + "_{}.p".format(i) for i in range(game_config.num_models)]
    if not args.load_model or not all([os.path.exists(path) for path in model_path]):
        print("model path not found, proceeding without previous weights")
        model_path = None
    counter_init = 0
    if model_path:
        for i, path in enumerate(model_path):
            print('resume model from path: ', path)
            weights = torch.load(path)
    
            models[i].load_state_dict(weights)
            target_models[i].load_state_dict(weights)

            try:
                counter_init = int(path.split('_')[-2])
            except:
                continue
    
    storage = SharedStorage.remote(models, target_models, [], counter_init=counter_init)
    replay_buffers = [ReplayBuffer.remote(config=game_config) for _ in range(game_config.num_models)]
    
    #_test(game_config, storage)
    
    #data_workers = [DataWorker.remote(rank, replay_buffers, storage, game_config, log=rank==0) for rank in range(game_config.num_actors)]
    #workers = [worker.run.remote() for worker in data_workers]
    
    a = DataWorker(1, replay_buffers, storage, game_config)
    a.run()
    
    while True:
        time.sleep(60)
        [r.remove_to_fit.remote() for r in replay_buffers]
    