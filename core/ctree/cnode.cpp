#include <iostream>
#include <numeric>
#include <random>
#include "cnode.h"


namespace tree{

    CSearchResults::CSearchResults(){
        this->num = 0;
    }

    CSearchResults::CSearchResults(int num){
        this->num = num;
        for(int i = 0; i < num; ++i){
            this->search_paths.push_back(std::vector<CNode*>());
        }
    }

    CSearchResults::~CSearchResults(){}
    
    int CSearchResults::get_searched_node(int path, int node_idx){
        std::vector<CNode*> s_path = this->search_paths[path];
        int path_len = static_cast<int>(s_path.size());
        if(node_idx >= path_len){
        node_idx = path_len - 1;
        }
        if(node_idx < 0){
        node_idx = (::std::max)(0, path_len + node_idx);
        }
        CNode* node = s_path[node_idx];
        return node->action;
        }


    //*********************************************************

    CNode::CNode(){
        this->prior = 0.0;
        this->action = 0;
        this->action_num = 0;
        this->best_action = -1;

        this->is_reset = 0;
        this->visit_count = 0;
        this->value_sum = 0.0;
        this->to_play = 0;
        this->value_prefix = 0.0;
        this->raw_value = 0.0;
        this->ptr_node_pool = nullptr;
        
        this->gumbel_scale = 10.0;
        this->gumbel_rng=0.0;
    }

    CNode::CNode(int action, float prior, int select_child_using_chance, std::vector<CNode>* ptr_node_pool){
        this->prior = prior;
        this->action = action;
        this->action_num = 0;
        this->select_child_using_chance = select_child_using_chance;

        this->is_reset = 0;
        this->visit_count = 0;
        this->value_sum = 0.0;
        this->best_action = -1;
        this->to_play = 0;
        this->value_prefix = 0.0;
        this->raw_value = 0.0;
        this->ptr_node_pool = ptr_node_pool;
        this->hidden_state_index_x = -1;
        this->hidden_state_index_y = -1;
        
        this->gumbel_scale = 10.0;
        this->gumbel_rng=0.0;
              
    }

    CNode::~CNode(){}

    void CNode::expand(int to_play, int hidden_state_index_x, int hidden_state_index_y, float value_prefix, float value, const std::vector<float> &policy_logits){
        this->to_play = to_play;
        this->hidden_state_index_x = hidden_state_index_x;
        this->hidden_state_index_y = hidden_state_index_y;
        this->value_prefix = value_prefix;
        this->raw_value = value;

        this->action_num = static_cast<int>(policy_logits.size());
        if(this->select_child_using_chance){}
        else{
        this->gumbel = generate_gumbel(this->gumbel_scale, this->gumbel_rng, this->action_num);
        }
        
        int action_num = this->action_num;
        float temp_policy;
        float policy_sum = 0.0;
        //float* policy = new float[action_num];
        std::vector<float> policy(action_num);
        float policy_max = FLOAT_MIN;
        for(int a = 0; a < action_num; ++a){
            if(policy_max < policy_logits[a]){
                policy_max = policy_logits[a];
            }
        }

        for(int a = 0; a < action_num; ++a){
            temp_policy = exp(policy_logits[a] - policy_max);
            policy_sum += temp_policy;
            policy[a] = temp_policy;
        }

        float prior;
        int ch = 1 - this->select_child_using_chance;        
        std::vector<CNode>* ptr_node_pool = this->ptr_node_pool;
        for(int a = 0; a < action_num; ++a){
            prior = policy[a] / policy_sum;
            int index = ptr_node_pool->size();
            this->children_index.push_back(index);
            ptr_node_pool->push_back(CNode(a, prior, ch, ptr_node_pool));
        }
    }

    void CNode::expand_as_root(int to_play, int hidden_state_index_x, int hidden_state_index_y, float value_prefix, float value, const std::vector<float> &policy_logits, const std::vector<int> &action_mapping){
        this->to_play = to_play;
        this->hidden_state_index_x = hidden_state_index_x;
        this->hidden_state_index_y = hidden_state_index_y;
        this->value_prefix = value_prefix;
        this->raw_value = value;

        this->action_num = static_cast<int>(policy_logits.size());
        if(this->select_child_using_chance){}
        else{
        this->gumbel = generate_gumbel(this->gumbel_scale, this->gumbel_rng, this->action_num);
        }
        
        int action_num = this->action_num;
        float temp_policy;
        float policy_sum = 0.0;
        //float* policy = new float[action_num];
        std::vector<float> policy(action_num);
        float policy_max = FLOAT_MIN;
        for(int a = 0; a < action_num; ++a){
            if(policy_max < policy_logits[a]){
                policy_max = policy_logits[a];
            }
        }

        for(int a = 0; a < action_num; ++a){
            temp_policy = exp(policy_logits[a] - policy_max);
            policy_sum += temp_policy;
            policy[a] = temp_policy;
        }

        float prior;
        int ch = 1 - this->select_child_using_chance;        
        std::vector<CNode>* ptr_node_pool = this->ptr_node_pool;
        for(int a = 0; a < action_num; ++a){
            prior = policy[a] / policy_sum;
            int index = ptr_node_pool->size();
            this->children_index.push_back(index);
            ptr_node_pool->push_back(CNode(action_mapping[a], prior, ch, ptr_node_pool));
        }
    }

    void CNode::add_exploration_noise(float exploration_fraction, const std::vector<float> &noises){
        float noise, prior;
        for(int a = 0; a < this->action_num; ++a){
            noise = noises[a];
            CNode* child = this->get_child(a);

            prior = child->prior;
            child->prior = prior * (1 - exploration_fraction) + noise * exploration_fraction;
        }
    }

    float CNode::get_mean_q(int isRoot, float parent_q, float discount){
        float total_unsigned_q = 0.0;
        int total_visits = 0;
        float parent_value_prefix = this->value_prefix;
        for(int a = 0; a < this->action_num; ++a){
            CNode* child = this->get_child(a);
            if(child->visit_count > 0){
                float true_reward = child->value_prefix - parent_value_prefix;
                if(this->is_reset == 1){
                    true_reward = child->value_prefix;
                }
                float qsa = true_reward + discount * child->value();
                total_unsigned_q += qsa;
                total_visits += 1;
            }
        }

        float mean_q = 0.0;
        if(isRoot && total_visits > 0){
            mean_q = (total_unsigned_q) / (total_visits);
        }
        else{
            mean_q = (parent_q + total_unsigned_q) / (total_visits + 1);
        }
        return mean_q;
    }

    void CNode::print_out(){
        return;
    }

    int CNode::expanded(){
        int child_num = this->children_index.size();
        if(child_num > 0) {
            return 1;
        }
        else {
            return 0;
        }
    }

    float CNode::value(){
        float true_value = 0.0;
        if(this->visit_count == 0){
            return true_value;
        }
        else{
            true_value = this->value_sum / this->visit_count;
            return true_value;
        }
    }

    std::vector<int> CNode::get_trajectory(){
        std::vector<int> traj;

        CNode* node = this;
        int best_action = node->best_action;
        while(best_action >= 0){
            traj.push_back(best_action);

            node = node->get_child(best_action);
            best_action = node->best_action;
        }
        return traj;
    }

    std::vector<int> CNode::get_children_distribution(){
        std::vector<int> distribution;
        if(this->expanded()){
            for(int a = 0; a < this->action_num; ++a){
                CNode* child = this->get_child(a);
                distribution.push_back(child->visit_count);
            }
        }
        return distribution;
    }

    CNode* CNode::get_child(int action){
        int index = this->children_index[action];
        return &((*(this->ptr_node_pool))[index]);
    }
    
    std::vector<float> CNode::get_children_value(float discount_factor)
    {
        /*
        Overview:
            Get the completed value of child nodes.
        Outputs:
            - discount_factor: the discount_factor of reward.
            - action_space_size: the size of action space.
        */
        float infymin = -std::numeric_limits<float>::infinity();
        std::vector<int> child_visit_count;
        std::vector<float> child_prior;
        for(int a = 0; a < this->action_num; ++a){
            CNode* child = this->get_child(a);
            child_visit_count.push_back(child->visit_count);
            child_prior.push_back(child->prior);
        }
        assert(child_visit_count.size()==child_prior.size());
        // compute the completed value
        std::vector<float> completed_qvalues = qtransform_completed_by_mix_value(this, child_visit_count, child_prior, discount_factor);
        std::vector<float> values;
        for (int i=0;i<child_prior.size();i++){
            values.push_back(completed_qvalues[i]);
        }

        return values;
    }
    
    std::vector<float> CNode::get_policy(float discount_factor){
        /*
        Overview:
            Compute the improved policy of the current node.
        Arguments:
            - discount_factor: the discount_factor of reward.
            - action_space_size: the action space size of environment.
        */
        float infymin = -std::numeric_limits<float>::infinity();
        std::vector<int> child_visit_count;
        std::vector<float> child_prior;
        for(int a = 0; a < this->action_num; ++a){
            CNode* child = this->get_child(a);
            child_visit_count.push_back(child->visit_count);
            child_prior.push_back(child->prior);
        }
        assert(child_visit_count.size()==child_prior.size());
        // compute the completed value
        std::vector<float> completed_qvalues = qtransform_completed_by_mix_value(this, child_visit_count, child_prior, discount_factor);
        std::vector<float> probs;
        for (int i=0;i<child_prior.size();i++){
            probs.push_back(child_prior[i] + completed_qvalues[i]);
        }

        csoftmax(probs, probs.size());

        return probs;
    }
    
    std::vector<float> CNode::get_q(float discount_factor)
    {
        /*
        Overview:
            Compute the q value of the current node.
        Arguments:
            - discount_factor: the discount_factor of reward.
        */
        std::vector<float> child_value;
        for(int a = 0; a < this->action_num; ++a){
            CNode* child = this->get_child(a);
            float true_reward = child->value_prefix;
            float qsa = true_reward + discount_factor * child->value();
            child_value.push_back(qsa);
        }
        return child_value;
    }

    //*********************************************************

    CRoots::CRoots(){
        this->root_num = 0;
        this->pool_size = 0;
    }

    CRoots::CRoots(int root_num, int pool_size){
        this->root_num = root_num;
        this->pool_size = pool_size;

        this->node_pools.reserve(root_num);
        this->roots.reserve(root_num);

        for(int i = 0; i < root_num; ++i){
            this->node_pools.push_back(std::vector<CNode>());
            this->node_pools[i].reserve(pool_size);

            this->roots.push_back(CNode(0, 0, 0, &this->node_pools[i]));
        }
    }

    CRoots::~CRoots(){}

    void CRoots::prepare(float root_exploration_fraction, const std::vector<std::vector<float>> &noises, const std::vector<float> &value_prefixs, const std::vector<float> &values, const std::vector<std::vector<float>> &policies, const std::vector<std::vector<int>> &action_mappings){
        for(int i = 0; i < this->root_num; ++i){
            this->roots[i].expand_as_root(0, 0, i, value_prefixs[i], values[i], policies[i], action_mappings[i]);
            this->roots[i].add_exploration_noise(root_exploration_fraction, noises[i]);

            this->roots[i].visit_count += 1;
        }
    }

    void CRoots::prepare_no_noise(const std::vector<float> &value_prefixs, const std::vector<float> &values, const std::vector<std::vector<float>> &policies, const std::vector<std::vector<int>> &action_mappings){
        for(int i = 0; i < this->root_num; ++i){
            this->roots[i].expand_as_root(0, 0, i, value_prefixs[i], values[i], policies[i], action_mappings[i]);

            this->roots[i].visit_count += 1;
        }
    }

    void CRoots::clear(){
        this->node_pools.clear();
        this->roots.clear();
    }

    std::vector<std::vector<int>> CRoots::get_trajectories(){
        std::vector<std::vector<int>> trajs;
        trajs.reserve(this->root_num);

        for(int i = 0; i < this->root_num; ++i){
            trajs.push_back(this->roots[i].get_trajectory());
        }
        return trajs;
    }

    std::vector<std::vector<int>> CRoots::get_distributions(){
        std::vector<std::vector<int>> distributions;
        distributions.reserve(this->root_num);

        for(int i = 0; i < this->root_num; ++i){
            distributions.push_back(this->roots[i].get_children_distribution());
        }
        return distributions;
    }

    std::vector<float> CRoots::get_values(){
        std::vector<float> values;
        for(int i = 0; i < this->root_num; ++i){
            values.push_back(this->roots[i].value());
        }
        return values;
    }
    
    std::vector<std::vector<float> > CRoots::get_children_values(float discount_factor)
    {
        /*
        Overview:
            Compute the completed value of each root.
        Arguments:
            - discount_factor: the discount_factor of reward.
            - action_space_size: the action space size of environment.
        */
        std::vector<std::vector<float> > values;
        values.reserve(this->root_num);

        for (int i = 0; i < this->root_num; ++i)
        {
            values.push_back(this->roots[i].get_children_value(discount_factor));
        }
        return values;
        
    }
    
    std::vector<std::vector<float> > CRoots::get_policies(float discount_factor)
    {
        /*
        Overview:
            Compute the improved policy of each root.
        Arguments:
            - discount_factor: the discount_factor of reward.
            - action_space_size: the action space size of environment.
        */
        std::vector<std::vector<float> > probs;
        probs.reserve(this->root_num);

        for(int i = 0; i < this->root_num; ++i){
            probs.push_back(this->roots[i].get_policy(discount_factor));
        }
        return probs;
    }

    //*********************************************************

    void update_tree_q(CNode* root, tools::CMinMaxStats &min_max_stats, float discount){
        std::stack<CNode*> node_stack;
        node_stack.push(root);
        //float parent_value_prefix = 0.0;
        int is_reset = 0;
        while(node_stack.size() > 0){
            CNode* node = node_stack.top();
            node_stack.pop();

            if(node != root){
                float true_reward = node->value_prefix;
                if(is_reset == 1){
                    true_reward = node->value_prefix;
                }
                float qsa = true_reward + discount * node->value();
                min_max_stats.update(qsa);
            }

            for(int a = 0; a < node->action_num; ++a){
                CNode* child = node->get_child(a);
                if(child->expanded()){
                    node_stack.push(child);
                }
            }

            //parent_value_prefix = node->value_prefix;
            is_reset = node->is_reset;
        }
    }

    void cback_propagate(std::vector<CNode*> &search_path, tools::CMinMaxStats &min_max_stats, int to_play, float value, float discount){
        float bootstrap_value = value;
        int path_len = search_path.size();
        for(int i = path_len - 1; i >= 0; --i){
            CNode* node = search_path[i];
            
            node->value_sum += bootstrap_value;
            node->visit_count += 1;

//            float true_reward  = node->reward;
//            int is_reset = 0;
//            if(i >= 1){
//                CNode* parent = search_path[i - 1];
//                parent_value_prefix = parent->value_prefix;
//                is_reset = parent->is_reset;
//                float qsa = (node->value_prefix - parent_value_prefix) + discount * node->value();
//                min_max_stats.update(qsa);
//            }

            float true_reward = node->value_prefix;
//            if(is_reset == 1){
                // parent is reset
//                true_reward = node->value_prefix;
//            }
            
            min_max_stats.update(true_reward + discount * node->value());
            bootstrap_value = true_reward + discount * bootstrap_value;
        }
        //min_max_stats.clear();
        //CNode* root = search_path[0];
        
        //update_tree_q(root, min_max_stats, discount);
    }

    void cbatch_back_propagate(int hidden_state_index_x, float discount, const std::vector<float> &value_prefixs, const std::vector<float> &values, const std::vector<std::vector<float>> &policies, tools::CMinMaxStatsList *min_max_stats_lst, CSearchResults &results, std::vector<int> is_reset_lst){
        for(int i = 0; i < results.num; ++i){
        
            results.nodes[i]->expand(0, hidden_state_index_x, i, value_prefixs[i], values[i], policies[i]);
            // reset
            results.nodes[i]->is_reset = is_reset_lst[i];

            cback_propagate(results.search_paths[i], min_max_stats_lst->stats_lst[i], 0, values[i], discount);
        }
    }
    
    
    std::vector<float> score_considered(int considered_visit, std::vector<float> gumbel, std::vector<float> logits, std::vector<float> normalized_qvalues, std::vector<int> visit_counts)
    {
        /*
        Overview:
            Calculate the score of nodes to be considered according to the considered visit.
        Arguments:
            - considered_visit: the visit counts of node to be considered.
            - gumbel: the gumbel vector.
            - logits: the logits vector of child nodes.
            - normalized_qvalues: the normalized Q values of child nodes.
            - visit_counts: the visit counts of child nodes.
        Outputs:
            - the score of nodes to be considered.
        */
        float low_logit = -1e9;
        float max_logit = *max_element(logits.begin(), logits.end());
        for (unsigned int i=0;i < logits.size();i++){
            logits[i] -= max_logit;
        }
        std::vector<float> penalty;
        for (unsigned int i=0;i < visit_counts.size();i++){
            // Only consider the nodes with specific visit counts
            if (visit_counts[i]==considered_visit)
                penalty.push_back(0);
            else
                penalty.push_back(-std::numeric_limits<float>::infinity());
        }
        
        assert(gumbel.size()==logits.size()==normalized_qvalues.size()==penalty.size());
        std::vector<float> score;
        for (unsigned int i=0;i < visit_counts.size();i++){
            score.push_back((::std::max)(low_logit, gumbel[i] + logits[i] + normalized_qvalues[i]) + penalty[i]);
        }

        return score;
    }

    int cselect_child(CNode* root, tools::CMinMaxStats &min_max_stats, int pb_c_base, float pb_c_init, float discount, float mean_q){
        int action = 0;
        if (root->select_child_using_chance) {
        
            int res = 10000;
            int roll = rand() % res;
            float prior_sum = 0;
            for(int a = 0; a < root->action_num; ++a){
                CNode* child = root->get_child(a);
                prior_sum += child->prior;
                if (roll <= res * prior_sum) {
                    action = a;
                    break;
                }

            }
        
        }
        
        else {
            float max_score = FLOAT_MIN;
            const float epsilon = 0.000000001;
            std::vector<int> max_index_lst;
            for(int a = 0; a < root->action_num; ++a){
                CNode* child = root->get_child(a);
                float temp_score = cucb_score(child, min_max_stats, mean_q, root->is_reset, root->visit_count - 1, root->value_prefix, pb_c_base, pb_c_init, discount);

                if(max_score < temp_score){
                    max_score = temp_score;

                    max_index_lst.clear();
                    max_index_lst.push_back(a);
                }
                else if(temp_score >= max_score - epsilon){
                    max_index_lst.push_back(a);
                }
            }

            if(max_index_lst.size() > 0){
                int rand_index = rand() % max_index_lst.size();
                action = max_index_lst[rand_index];
            }
        }
        
        return action;
    }

    std::pair<int, int> cselect_root_child(CNode* root, float discount_factor, int num_simulations, int max_num_considered_actions)
    {
        /*
        Overview:
            Select the child node of the roots in gumbel muzero.
        Arguments:
            - root: the roots to select the child node.
            - disount_factor: the discount factor of reward.
            - num_simulations: the upper limit number of simulations.
            - max_num_considered_actions: the maximum number of considered actions.
        Outputs:
            - action: the action to select.
        */
        int max_child_num = 0;
        int max_action = 0;
        if (root->select_child_using_chance) {
        
            int res = 10000;
            int roll = rand() % res;
            float prior_sum = 0;
            for(int a = 0; a < root->action_num; ++a){
                CNode* child = root->get_child(a);
                prior_sum += child->prior;
                if (roll <= res * prior_sum) {
                    max_child_num = a;
                    max_action = child->action;
                    break;
                }

            }
        
        }
        
        else{        
            std::vector<int> child_visit_count;
            std::vector<float> child_prior;
            for(int a = 0; a < root->action_num; ++a){
                CNode* child = root->get_child(a);
                child_visit_count.push_back(child->visit_count);
                child_prior.push_back(child->prior);
            }
            assert(child_visit_count.size()==child_prior.size());
    
            std::vector<float> completed_qvalues = qtransform_completed_by_mix_value(root, child_visit_count, child_prior, discount_factor);
            std::vector<std::vector<int> > visit_table = get_table_of_considered_visits(max_num_considered_actions, num_simulations);
            
            int num_valid_actions = root->action_num;
            int num_considered = (::std::min)(max_num_considered_actions, num_simulations);
            int simulation_index = std::accumulate(child_visit_count.begin(), child_visit_count.end(), 0);
            int considered_visit = visit_table[num_considered][simulation_index];
    
            std::vector<float> score = score_considered(considered_visit, root->gumbel, child_prior, completed_qvalues, child_visit_count);
    
            float argmax = -std::numeric_limits<float>::infinity();

            int index = 0;
            for(int a = 0; a < root->action_num; ++a){
                if(score[index] > argmax){
                    argmax = score[index];
                    max_child_num = a;
                    CNode* child = root->get_child(a);
                    max_action = child->action;
                }
                index += 1;
            }
        }
        return std::pair<int, int>(max_child_num, max_action);    
    }

    int cselect_interior_child(CNode* root, float discount_factor)
        {
            /*
            Overview:
                Select the child node of the interior node in gumbel muzero.
            Arguments:
                - root: the roots to select the child node.
                - disount_factor: the discount factor of reward.
            Outputs:
                - action: the action to select.
            */
            
            int max_action = 0;
            if (root->select_child_using_chance) {
            
                int res = 10000;
                int roll = rand() % res;
                float prior_sum = 0;
                for(int a = 0; a < root->action_num; ++a){
                    CNode* child = root->get_child(a);
                    prior_sum += child->prior;
                    if (roll <= res * prior_sum) {
                        max_action = a;
                        break;
                    }

                }
            
            }
            
            else{
            
                std::vector<int> child_visit_count;
                std::vector<float> child_prior;
                for(int a = 0; a < root->action_num; ++a){
                    CNode* child = root->get_child(a);
                    child_visit_count.push_back(child->visit_count);
                    child_prior.push_back(child->prior);
                }
                assert(child_visit_count.size()==child_prior.size());
                std::vector<float> completed_qvalues = qtransform_completed_by_mix_value(root, child_visit_count, child_prior, discount_factor);
                std::vector<float> probs;
                for (int i=0;i<child_prior.size();i++){
                    probs.push_back(child_prior[i] + completed_qvalues[i]);
                }
                csoftmax(probs, probs.size());
                int visit_count_sum = std::accumulate(child_visit_count.begin(), child_visit_count.end(), 0);
                std::vector<float> to_argmax;
                for (int i=0;i<probs.size();i++){
                    to_argmax.push_back(probs[i] - (float)child_visit_count[i]/(float)(1+visit_count_sum));
                }
                
                float argmax = -std::numeric_limits<float>::infinity();
                int index = 0;
                for(int a = 0; a < root->action_num; ++a){
                    if(to_argmax[index] > argmax){
                        argmax = to_argmax[index];
                        max_action = a;
                    }
                    index += 1;
                }
            }
            return max_action;
        }

    float cucb_score(CNode *child, tools::CMinMaxStats &min_max_stats, float parent_mean_q, int is_reset, float total_children_visit_counts, float parent_value_prefix, float pb_c_base, float pb_c_init, float discount){
        float pb_c = 0.0, prior_score = 0.0, value_score = 0.0;
        pb_c = log((total_children_visit_counts + pb_c_base + 1) / pb_c_base) + pb_c_init;
        pb_c *= (sqrt(total_children_visit_counts) / (child->visit_count + 1));

        prior_score = pb_c * child->prior;
        if (child->visit_count == 0){
            value_score = parent_mean_q;
        }
        else {
            float true_reward = child->value_prefix - parent_value_prefix;
            if(is_reset == 1){
                true_reward = child->value_prefix;
            }
            value_score = true_reward + discount * child->value();
        }

        value_score = min_max_stats.normalize(value_score);

        if (value_score < 0) value_score = 0;
        if (value_score > 1) value_score = 1;

        float ucb_value = prior_score + value_score;
        return ucb_value;
    }

    void cbatch_traverse(CRoots *roots, int num_simulations, int max_num_considered_actions, float discount, tools::CMinMaxStatsList *min_max_stats_lst, CSearchResults &results){
        // set seed
        timeval t1;
        gettimeofday(&t1, NULL);
        srand(t1.tv_usec);

        int last_action = -1;
        float parent_q = 0.0;
        results.search_lens = std::vector<int>();
        
        
        for(int i = 0; i < results.num; ++i){
            CNode *node = &(roots->roots[i]);
            int is_root = 1;
            int search_len = 0;
            results.search_paths[i].push_back(node);

            while(node->expanded()){
                int action;
                int child_idx;
                if(is_root){                    
                    std::pair<int, int> result = cselect_root_child(node, discount, num_simulations, max_num_considered_actions);
                    child_idx = result.first;
                    action = result.second;
                    }
                else{
                    action = cselect_interior_child(node, discount);
                    child_idx = action;
                    }
                    
                //float mean_q = node->get_mean_q(is_root, parent_q, discount);
                is_root = 0;
                //parent_q = mean_q;

                //int action = cselect_child(node, min_max_stats_lst->stats_lst[i], pb_c_base, pb_c_init, discount, mean_q);
                node->best_action = child_idx;
                // next
                node = node->get_child(child_idx);
                               
                last_action = action;
                results.search_paths[i].push_back(node);
                search_len += 1;
            }

            // results need to

            CNode* parent = results.search_paths[i][results.search_paths[i].size() - 2];

            results.hidden_state_index_x_lst.push_back(parent->hidden_state_index_x);
            results.hidden_state_index_y_lst.push_back(parent->hidden_state_index_y);
            
            if (node->select_child_using_chance) {
                results.chance_node_y_lst.push_back(i);
            }
            
            else {
                results.policy_node_y_lst.push_back(i);
            }

            results.last_actions.push_back(last_action);
            results.search_lens.push_back(search_len);
            results.nodes.push_back(node);
        }
    }

    void cbatch_step(tools::CMinMaxStatsList *min_max_stats_lst, CSearchResults &results, const std::vector<int> &to_step, int hidden_state_index_x, float discount, const std::vector<float> &value_prefixs, const std::vector<float> &values, const std::vector<std::vector<float>> &policies){
        // set seed
        timeval t1;
        gettimeofday(&t1, NULL);
        srand(t1.tv_usec);        
        for (int i=0;i<to_step.size();i++){
            results.nodes[to_step[i]]->expand(0, hidden_state_index_x, i, value_prefixs[i], values[i], policies[i]);
            int action;
            int child_idx;
            action = cselect_interior_child(results.nodes[to_step[i]], discount);
            child_idx = action;
            results.nodes[to_step[i]]->best_action = child_idx;
            CNode* node = results.nodes[to_step[i]]->get_child(child_idx);
            results.search_paths[to_step[i]].push_back(node);
            CNode* parent = results.search_paths[to_step[i]][results.search_paths[to_step[i]].size() - 2];
            results.hidden_state_index_x_lst[to_step[i]] = parent->hidden_state_index_x;
            results.hidden_state_index_y_lst[to_step[i]] = parent->hidden_state_index_y;
            results.last_actions[to_step[i]] = action;
            results.search_lens[to_step[i]] += 1;
            results.nodes[to_step[i]] = node;
        }
    }

    
    std::vector<float> generate_gumbel(float gumbel_scale, float gumbel_rng, int shape){
        /*
        Overview:
            Generate gumbel vectors.
        Arguments:
            - gumbel_scale: the scale of gumbel.
            - gumbel_rng: the seed to generate gumbel.
            - shape: the shape of gumbel vectors to be generated
        Outputs:
            - gumbel vectors.
        */
        std::mt19937 gen(static_cast<unsigned int>(gumbel_rng));
        std::extreme_value_distribution<float> d(0, 1);

        std::vector<float> gumbel;
        for (int i = 0;i < shape;i++)
            gumbel.push_back(gumbel_scale * d(gen));
        return gumbel;
    }
    
    float compute_mixed_value(float raw_value, std::vector<float> q_values, std::vector<int> &child_visit, std::vector<float> &child_prior)
    {
        /*
        Overview:
            Compute the mixed Q value.
        Arguments:
            - raw_value: the approximated value of the current node from the value network.
            - q_value: the q value of the current node.
            - child_visit: the visit counts of the child nodes.
            - child_prior: the prior of the child nodes.
        Outputs:
            - mixed Q value.
        */
        float visit_count_sum = 0.0;
        float probs_sum = 0.0;
        float weighted_q_sum = 0.0;
        float min_num = -10e7;

        for(unsigned int i = 0;i < child_visit.size();i++)
            visit_count_sum += child_visit[i];

        for(unsigned int i = 0;i < child_prior.size();i++)
            // Ensuring non-nan prior
            child_prior[i] = (::std::max)(child_prior[i], min_num);
        
        for(unsigned int i = 0;i < child_prior.size();i++)
            if (child_visit[i] > 0)
                probs_sum += child_prior[i];
        
        for (unsigned int i = 0;i < child_prior.size();i++)
            if (child_visit[i] > 0){
                weighted_q_sum += child_prior[i] * q_values[i] / probs_sum;
            }

        return (raw_value + visit_count_sum * weighted_q_sum) / (visit_count_sum+1);
    }
    
    void rescale_qvalues(std::vector<float> &value, float epsilon){
        /*
        Overview:
            Rescale the q value with max-min normalization.
        Arguments:
            - value: the value vector to be rescaled.
            - epsilon: the lower limit of gap.
        */
        float max_value = *max_element(value.begin(), value.end());
        float min_value = *min_element(value.begin(), value.end());
        float gap = max_value - min_value;
        gap = (::std::max)(gap, epsilon);
        for (unsigned int i = 0;i < value.size();i++){
            value[i] = (value[i]-min_value)/gap;
        }
    }
    
    std::vector<float> qtransform_completed_by_mix_value(CNode *root, std::vector<int> & child_visit, \
        std::vector<float> & child_prior, float discount_factor, float maxvisit_init, float value_scale, \
        bool rescale_values, float epsilon)
        {
            /*
            Overview:
                Calculate the q value with mixed value.
            Arguments:
                - root: the roots that search from.
                - child_visit: the visit counts of the child nodes.
                - child_prior: the prior of the child nodes.
                - discount_factor: the discount factor of reward.
                - maxvisit_init: the init of the maximization of visit counts.
                - value_cale: the scale of value.
                - rescale_values: whether to rescale the values.
                - epsilon: the lower limit of gap in max-min normalization
            Outputs:
                - completed Q value.
            */
            assert (child_visit.size() == child_prior.size());
            std::vector<float> qvalues;
            std::vector<float> child_prior_tmp;
    
            child_prior_tmp.assign(child_prior.begin(), child_prior.end());
            qvalues = root->get_q(discount_factor);
            csoftmax(child_prior_tmp, child_prior_tmp.size());
            // TODO: should be raw_value here
            float value = compute_mixed_value(root->raw_value, qvalues, child_visit, child_prior_tmp);
            std::vector<float> completed_qvalue;
    
            for (unsigned int i = 0;i < child_prior_tmp.size();i++){
                if (child_visit[i] > 0){
                    completed_qvalue.push_back(qvalues[i]);
                }
                else{
                    completed_qvalue.push_back(value);
                }
            }
    
            if (rescale_values){
                rescale_qvalues(completed_qvalue, epsilon);
            }
    
            float max_visit = *max_element(child_visit.begin(), child_visit.end());
            float visit_scale = maxvisit_init + max_visit;
    
            for (unsigned int i=0;i < completed_qvalue.size();i++){
                completed_qvalue[i] = completed_qvalue[i] * visit_scale * value_scale;
            }
            return completed_qvalue;
            
        }
    
    std::vector<int> get_sequence_of_considered_visits(int max_num_considered_actions, int num_simulations)
        {
            /*
            Overview:
                Calculate the considered visit sequence.
            Arguments:
                - max_num_considered_actions: the maximum number of considered actions.
                - num_simulations: the upper limit number of simulations.
            Outputs:
                - the considered visit sequence.
            */
            std::vector<int> visit_seq;
            if(max_num_considered_actions <= 1){
                for (int i=0;i < num_simulations;i++)
                    visit_seq.push_back(i);
                return visit_seq;
            }
    
            int log2max = std::ceil(std::log2(max_num_considered_actions));
            std::vector<int> visits;
            for (int i = 0;i < max_num_considered_actions;i++)
                visits.push_back(0);
            int num_considered = max_num_considered_actions;
            while (visit_seq.size() < num_simulations){
                int num_extra_visits = (::std::max)(1, (int)(num_simulations / (log2max * num_considered)));
                for (int i = 0;i < num_extra_visits;i++){
                    visit_seq.insert(visit_seq.end(), visits.begin(), visits.begin() + num_considered);
                    for (int j = 0;j < num_considered;j++)
                        visits[j] += 1;
                }
                num_considered = (::std::max)(2, num_considered/2);
            }
            std::vector<int> visit_seq_slice;
            visit_seq_slice.assign(visit_seq.begin(), visit_seq.begin() + num_simulations);
            return visit_seq_slice;
        }

    std::vector<std::vector<int> > get_table_of_considered_visits(int max_num_considered_actions, int num_simulations)
        {
            /*
            Overview:
                Calculate the table of considered visits.
            Arguments:
                - max_num_considered_actions: the maximum number of considered actions.
                - num_simulations: the upper limit number of simulations.
            Outputs:
                - the table of considered visits.
            */
            std::vector<std::vector<int> > table;
            for (int m=0;m < max_num_considered_actions+1;m++){
                table.push_back(get_sequence_of_considered_visits(m, num_simulations));
            }
            return table;
        }
        
    void csoftmax(std::vector<float> &input, int input_len)
        {
            /*
            Overview:
                Softmax transformation.
            Arguments:
                - input: the vector to be transformed.
                - input_len: the length of input vector.
            */
            assert (input != NULL);
            assert (input_len != 0);
            int i;
            float m;
            // Find maximum value from input array
            m = input[0];
            for (i = 1; i < input_len; i++) {
                if (input[i] > m) {
                    m = input[i];
                }
            }
    
            float sum = 0;
            for (i = 0; i < input_len; i++) {
                sum += expf(input[i]-m);
            }
    
            for (i = 0; i < input_len; i++) {
                input[i] = expf(input[i] - m - log(sum));
            }    
        }

}