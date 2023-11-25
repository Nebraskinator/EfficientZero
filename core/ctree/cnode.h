#ifndef CNODE_H
#define CNODE_H

#include "cminimax.h"
#include <math.h>
#include <vector>
#include <stack>
#include <stdlib.h>
#include <time.h>
#include <cmath>
#include <sys/timeb.h>
//#include <sys/time.h>


const int DEBUG_MODE = 0;

namespace tree {

    class CNode {
        public:
            int visit_count, to_play, action, action_num, hidden_state_index_x, hidden_state_index_y, best_action, is_reset, select_child_using_chance;
            float value_prefix, raw_value, prior, value_sum, gumbel_scale, gumbel_rng;
            std::vector<int> children_index;
            std::vector<CNode>* ptr_node_pool;
            std::vector<float> gumbel;

            CNode();
            CNode(int action, float prior, int select_child_using_chance, std::vector<CNode> *ptr_node_pool);
            ~CNode();

            void expand(int to_play, int hidden_state_index_x, int hidden_state_index_y, float value_prefix, float value, const std::vector<float> &policy_logits);
            void expand_as_root(int to_play, int hidden_state_index_x, int hidden_state_index_y, float value_prefix, float value, const std::vector<float> &policy_logits, const std::vector<int> &action_mapping);
            void add_exploration_noise(float exploration_fraction, const std::vector<float> &noises);
            float get_mean_q(int isRoot, float parent_q, float discount);
            void print_out();

            int expanded();

            float value();
            std::vector<float> get_q(float discount_factor);

            std::vector<int> get_trajectory();
            std::vector<int> get_children_distribution();
            CNode* get_child(int action);
            std::vector<float> get_children_value(float discount_factor);
            std::vector<float> get_policy(float discount_factor);
    };

    class CRoots{
        public:
            int root_num, pool_size;
            std::vector<CNode> roots;
            std::vector<std::vector<CNode>> node_pools;

            CRoots();
            CRoots(int root_num, int pool_size);
            ~CRoots();

            void prepare(float root_exploration_fraction, const std::vector<std::vector<float>> &noises, const std::vector<float> &value_prefixs, const std::vector<float> &values, const std::vector<std::vector<float>> &policies, const std::vector<std::vector<int>> &action_mappings);
            void prepare_no_noise(const std::vector<float> &value_prefixs, const std::vector<float> &values, const std::vector<std::vector<float>> &policies, const std::vector<std::vector<int>> &action_mappings);
            void clear();
            std::vector<std::vector<int>> get_trajectories();
            std::vector<std::vector<int>> get_distributions();
            std::vector<float> get_values();
            std::vector<std::vector<float> > get_children_values(float discount_factor);
            std::vector<std::vector<float> > get_policies(float discount_factor);
    };

    class CSearchResults{
        public:
            int num;
            std::vector<int> hidden_state_index_x_lst, hidden_state_index_y_lst, policy_node_y_lst, chance_node_y_lst, last_actions, search_lens;
            std::vector<CNode*> nodes;
            std::vector<std::vector<CNode*>> search_paths;

            CSearchResults();
            CSearchResults(int num);
            ~CSearchResults();
            
            int CSearchResults::get_searched_node(int path, int node_idx);

    };


    //*********************************************************
    void update_tree_q(CNode* root, tools::CMinMaxStats &min_max_stats, float discount);
    void cback_propagate(std::vector<CNode*> &search_path, tools::CMinMaxStats &min_max_stats, int to_play, float value, float discount);
    void cbatch_back_propagate(int hidden_state_index_x, float discount, const std::vector<float> &value_prefixs, const std::vector<float> &values, const std::vector<std::vector<float>> &policies, tools::CMinMaxStatsList *min_max_stats_lst, CSearchResults &results, std::vector<int> is_reset_lst);
    int cselect_child(CNode* root, tools::CMinMaxStats &min_max_stats, int pb_c_base, float pb_c_init, float discount, float mean_q);
    float cucb_score(CNode *child, tools::CMinMaxStats &min_max_stats, float parent_mean_q, int is_reset, float total_children_visit_counts, float parent_value_prefix, float pb_c_base, float pb_c_init, float discount);
    void cbatch_traverse(CRoots *roots, int num_simulations, int max_num_considered_actions, float discount, tools::CMinMaxStatsList *min_max_stats_lst, CSearchResults &results);
    void cbatch_step(tools::CMinMaxStatsList *min_max_stats_lst, CSearchResults &results, const std::vector<int> &to_step, int hidden_state_index_x, float discount, const std::vector<float> &value_prefixs, const std::vector<float> &values, const std::vector<std::vector<float>> &policies);
    void csoftmax(std::vector<float> &input, int input_len);
    float compute_mixed_value(float raw_value, std::vector<float> q_values, std::vector<int> &child_visit, std::vector<float> &child_prior);
    void rescale_qvalues(std::vector<float> &value, float epsilon);
    std::vector<float> qtransform_completed_by_mix_value(CNode *root, std::vector<int> & child_visit, \
        std::vector<float> & child_prior, float discount= 0.99, float maxvisit_init = 50.0, float value_scale = 0.1, \
        bool rescale_values = true, float epsilon = 1e-8);
    std::vector<int> get_sequence_of_considered_visits(int max_num_considered_actions, int num_simulations);
    std::vector<std::vector<int> > get_table_of_considered_visits(int max_num_considered_actions, int num_simulations);
    std::vector<float> score_considered(int considered_visit, std::vector<float> gumbel, std::vector<float> logits, std::vector<float> normalized_qvalues, std::vector<int> visit_counts);
    std::vector<float> generate_gumbel(float gumbel_scale, float gumbel_rng, int shape);
    std::pair<int, int> cselect_root_child(CNode* root, float discount_factor, int num_simulations, int max_num_considered_actions);
    int cselect_interior_child(CNode* root, float discount_factor);
}

#endif