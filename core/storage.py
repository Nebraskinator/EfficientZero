import ray
import gc
import copy
from ray.util.queue import Queue


class QueueStorage(object):
    def __init__(self, threshold=15, size=20):
        """Queue storage
        Parameters
        ----------
        threshold: int
            if the current size if larger than threshold, the data won't be collected
        size: int
            the size of the queue
        """
        self.threshold = threshold
        self.size = size
        self.queue = Queue(maxsize=self.size, actor_options={"num_cpus": 3})
        self.current_model = 0


    def push(self, batch, curr_model):
        if self.queue.qsize() <= self.threshold:
            self.queue.put(batch)


    def pop(self):
        if self.queue.qsize() > 0:
            return self.queue.get()
        else:
            return None

    def get_len(self):
        return self.queue.qsize()


@ray.remote
class SharedStorage(object):
    def __init__(self, models, target_models, prev_models, counter_init=0):
        """Shared storage for models and others
        Parameters
        ----------
        model: any
            models for self-play (update every checkpoint_interval)
        target_model: any
            models for reanalyzing (update every target_model_interval)
        """
        self.step_counters = [counter_init for _ in range(len(models))]
        self.test_counters = [0 for _ in range(len(models))]
        self.models = models
        self.target_models = target_models
        self.prev_models = prev_models
        self.current_model = 0
        self.ori_reward_log = []
        self.reward_log = []
        self.reward_max_log = []
        self.test_dict_log = {}
        self.eps_lengths = []
        self.eps_lengths_max = []
        self.temperature_log = []
        self.visit_entropies_log = []
        self.priority_self_play_log = []
        self.distributions_log = {}
        self.starts = [False for _ in range(len(models))]

    def set_start_signal(self, curr_model):
        self.starts[curr_model] = True

    def get_start_signal(self, curr_model):
        return self.starts[curr_model]

    def get_weights(self, curr_model):
        return self.models[curr_model].get_weights()

    def set_weights(self, weights, curr_model):
        weights_copy = copy.deepcopy(weights)
        self.models[curr_model].set_weights(weights_copy)
        del weights
        del weights_copy
        gc.collect()

    def get_target_weights(self, curr_model):
        return self.target_models[curr_model].get_weights()

    def set_target_weights(self, weights, curr_model):
        weights_copy = copy.deepcopy(weights)
        self.target_models[curr_model].set_weights(weights_copy)
        del weights
        del weights_copy
        gc.collect()

    def update_previous_models(self, curr_model):
        if self.prev_models:
            self.prev_models[0].set_weights(self.models[curr_model].get_weights())
            self.prev_models = self.prev_models[1:] + [self.prev_models[0]]
    
    def get_previous_models_weights(self):
        return [m.get_weights() for m in self.prev_models]
    
    def set_counter(self, count, curr_model):
        self.step_counters[curr_model] = count

    def incr_counter(self, curr_model):
        self.step_counters[curr_model] += 1

    def get_counter(self, curr_model):
        return self.step_counters[curr_model]
    
    def get_current_model(self):
        return self.current_model
    
    def set_current_model(self, curr_model):
        self.current_model = curr_model

    def set_data_worker_logs(self, eps_len, eps_len_max, eps_ori_reward, eps_reward, eps_reward_max, temperature, visit_entropy, priority_self_play, distributions):
        self.eps_lengths.append(eps_len)
        self.eps_lengths_max.append(eps_len_max)
        self.ori_reward_log.append(eps_ori_reward)
        self.reward_log.append(eps_reward)
        self.reward_max_log.append(eps_reward_max)
        self.temperature_log.append(temperature)
        self.visit_entropies_log.append(visit_entropy)
        self.priority_self_play_log.append(priority_self_play)

        for key, val in distributions.items():
            if key not in self.distributions_log.keys():
                self.distributions_log[key] = []
            self.distributions_log[key] += val

    def add_test_log(self, test_counter, test_dict):
        self.test_counter = test_counter
        for key, val in test_dict.items():
            if key not in self.test_dict_log.keys():
                self.test_dict_log[key] = []
            self.test_dict_log[key].append(val)

    def get_worker_logs(self):
        if len(self.reward_log) > 0:
            ori_reward = sum(self.ori_reward_log) / len(self.ori_reward_log)
            reward = sum(self.reward_log) / len(self.reward_log)
            reward_max = sum(self.reward_max_log) / len(self.reward_max_log)
            eps_lengths = sum(self.eps_lengths) / len(self.eps_lengths)
            eps_lengths_max = sum(self.eps_lengths_max) / len(self.eps_lengths_max)
            temperature = sum(self.temperature_log) / len(self.temperature_log)
            visit_entropy = sum(self.visit_entropies_log) / len(self.visit_entropies_log)
            priority_self_play = sum(self.priority_self_play_log) / len(self.priority_self_play_log)
            distributions = self.distributions_log

            self.ori_reward_log = []
            self.reward_log = []
            self.reward_max_log = []
            self.eps_lengths = []
            self.eps_lengths_max = []
            self.temperature_log = []
            self.visit_entropies_log = []
            self.priority_self_play_log = []
            self.distributions_log = {}

        else:
            ori_reward = None
            reward = None
            reward_max = None
            eps_lengths = None
            eps_lengths_max = None
            temperature = None
            visit_entropy = None
            priority_self_play = None
            distributions = None

        if len(self.test_dict_log) > 0:
            test_dict = self.test_dict_log

            self.test_dict_log = {}
            test_counter = self.test_counter
        else:
            test_dict = None
            test_counter = None

        return ori_reward, reward, reward_max, eps_lengths, eps_lengths_max, test_counter, test_dict, temperature, visit_entropy, priority_self_play, distributions
