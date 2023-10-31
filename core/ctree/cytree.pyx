# distutils: language=c++
import ctypes
cimport cython
from ctree cimport CMinMaxStatsList, CNode, CRoots, CSearchResults, cbatch_back_propagate, cbatch_traverse
from libcpp.vector cimport vector
from libc.stdlib cimport malloc, free
from libcpp.list cimport list as cpplist

import numpy as np
cimport numpy as np

ctypedef np.npy_float FLOAT
ctypedef np.npy_intp INTP


cdef class MinMaxStatsList:
    cdef CMinMaxStatsList *cmin_max_stats_lst

    def __cinit__(self, int num):
        self.cmin_max_stats_lst = new CMinMaxStatsList(num)

    def set_delta(self, float value_delta_max):
        self.cmin_max_stats_lst[0].set_delta(value_delta_max)

    def __dealloc__(self):
        del self.cmin_max_stats_lst


cdef class ResultsWrapper:
    cdef CSearchResults cresults

    def __cinit__(self, int num):
        self.cresults = CSearchResults(num)

    def get_search_len(self):
        return self.cresults.search_lens


cdef class Roots:
    cdef int root_num
    cdef int pool_size
    cdef CRoots *roots

    def __cinit__(self, int root_num, int pool_size):
        self.root_num = root_num
        self.pool_size = pool_size
        self.roots = new CRoots(root_num, self.pool_size)

    def prepare(self, float root_exploration_fraction, list noises, list value_prefix_pool, list value_pool, list policy_logits_pool, list action_mappings):
        self.roots[0].prepare(root_exploration_fraction, noises, value_prefix_pool, value_pool, policy_logits_pool, action_mappings)

    def prepare_no_noise(self, list value_prefix_pool, list value_pool, list policy_logits_pool, list action_mappings):
        self.roots[0].prepare_no_noise(value_prefix_pool, value_pool, policy_logits_pool, action_mappings)

    def get_trajectories(self):
        return self.roots[0].get_trajectories()

    def get_distributions(self):
        return self.roots[0].get_distributions()

    def get_values(self):
        return self.roots[0].get_values()
    
    def get_children_values(self, float discount):
        return self.roots[0].get_children_values(discount)
    
    def get_policies(self, float discount):
        return self.roots[0].get_policies(discount)    

    def clear(self):
        self.roots[0].clear()

    def __dealloc__(self):
        del self.roots

    @property
    def num(self):
        return self.root_num


cdef class Node:
    cdef CNode cnode

    def __cinit__(self):
        pass

    def __cinit__(self, float prior, int action_num):
        # self.cnode = CNode(prior, action_num)
        pass

    def expand_as_root(self, int to_play, int hidden_state_index_x, int hidden_state_index_y, float value_prefix, float value, list policy_logits, list action_mapping):
        cdef vector[float] cpolicy = policy_logits
        cdef vector[int] caction_mapping = action_mapping
        self.cnode.expand_as_root(to_play, hidden_state_index_x, hidden_state_index_y, value_prefix, value, cpolicy, caction_mapping)
    #def __dealloc__(self):
    #    del self.cnode

def batch_back_propagate(int hidden_state_index_x, float discount, list value_prefixs, list values, list policies, MinMaxStatsList min_max_stats_lst, ResultsWrapper results, list is_reset_lst):
    cdef int i
    cdef vector[float] cvalue_prefixs = value_prefixs
    cdef vector[float] cvalues = values
    cdef vector[vector[float]] cpolicies = policies

    cbatch_back_propagate(hidden_state_index_x, discount, cvalue_prefixs, cvalues, cpolicies,
                          min_max_stats_lst.cmin_max_stats_lst, results.cresults, is_reset_lst)


def batch_traverse(Roots roots, int num_simulations, int max_num_considered_actions, float discount, MinMaxStatsList min_max_stats_lst, ResultsWrapper results):

    cbatch_traverse(roots.roots, num_simulations, max_num_considered_actions, discount, min_max_stats_lst.cmin_max_stats_lst, results.cresults)

    return results.cresults.hidden_state_index_x_lst, results.cresults.hidden_state_index_y_lst, results.cresults.policy_node_y_lst, results.cresults.chance_node_y_lst, results.cresults.last_actions
