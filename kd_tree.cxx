#include "kd_tree.hxx"

#include <list>
#include <vector>
#include <algorithm>
#include <cfloat>
#include <iostream>

kd_node::kd_node()
{
    this->left = NULL;
    this->right = NULL;
    this->ki = -1;
}

feature::feature() 
{ 
    this->x = -1;
    this->y = -1;
    this->key = std::vector<float>();
}

feature::feature(float x, float y, int n)
{
    this->x = x;
    this->y = y;
    this->key = std::vector<float>(n);
}


mq_node::mq_node(kd_node *node, float dist)
{
    this->node = node;
    this->dist = dist;
}

kd_tree::kd_tree(std::vector<feature> feat_list, int nKey)
{
    this->nKey = nKey;
    if (feat_list.size() == 0) {
        std::cerr << "feat_list size must be larger than 0" << std::endl;
        exit(EXIT_FAILURE);
        return;
    }
    if (nKey == 0) {
        std::cerr << "nKey must be larger than 0" << std::endl;
        exit(EXIT_FAILURE);
    }

    this->feat_list = feat_list; // deep copy
    this->root = new kd_node();
    for (int i=0; i<feat_list.size(); i++)
        this->root->idx_list.push_front(i);
    divide_kd_node(this->root); 
}


void 
kd_tree::divide_kd_node(kd_node *node)
{
    if (node->idx_list.size() <= 1) {
        return;
    }

    set_partition(node);
    int ki = node->ki;
    float kv = node->kv;

    kd_node *left  = new kd_node();
    kd_node *right = new kd_node();
    for (std::list<int>::iterator it=node->idx_list.begin();
        it != node->idx_list.end(); ++it)
    {
        int idx = *it;
        if (feat_list[idx].key[ki] < kv)
            left->idx_list.push_front(idx);
        else
            right->idx_list.push_front(idx);
    }

    if (left->idx_list.size() == 0 || right->idx_list.size() == 0) {
        delete left;
        delete right;
        node->ki = -1;
        return;
    }

    node->idx_list.clear();
    node->left = left;
    node->right = right;
    
    divide_kd_node(left);
    divide_kd_node(right);
}

void 
kd_tree::set_partition(kd_node *node)
{
    std::vector< float > mean_list(nKey, 0.0);  // E[X]
    std::vector< float > mean2_list(nKey, 0.0); // E[X^2]
    std::vector< float > var_list(nKey, 0.0); // V(X)

    // Get E[X] and E[X^2]  value
    for (std::list<int>::iterator it=node->idx_list.begin();
        it != node->idx_list.end(); ++it)
    {
        int idx = *it;
        for (int j=0; j<nKey; j++) {
            float X = feat_list[idx].key[j];
            mean_list[j]  += X;
            mean2_list[j] += X*X;
        }
    }
    for (int j=0; j<nKey; j++) {
        mean_list[j] /= node->idx_list.size();
        mean2_list[j] /= node->idx_list.size();
        var_list[j] = mean2_list[j] - mean_list[j]*mean_list[j];
    }

    // ki : the index of the key with the largest variance
    node->ki = std::max_element(var_list.begin(), var_list.end()) - var_list.begin();

    // kv : median value
    std::vector<float> v(feat_list.size());
    for (int i=0; i<feat_list.size(); i++)
        v[i] = feat_list[i].key[node->ki];
    std::nth_element(v.begin(), v.begin()+v.size()/2, v.end());
    node->kv = v[v.size()/2];
}

int 
kd_tree::bbf_search(const feature &feat, const int MAX_EXPLORE)
{
    std::list<int> nhbrs;
    std::priority_queue<mq_node, 
        std::vector<mq_node>, 
        mq_node_comparator> mq;

    mq.push( mq_node(root, 0.0) );
    float minDist=FLT_MAX;
    int minNhbr;
    int t=0;
    while ( mq.size() > 0 && t < MAX_EXPLORE) {
        mq_node  tmp_mq_node = mq.top(); mq.pop();
        kd_node* tmp_kd_node = tmp_mq_node.node;
        int   tmp_ki = tmp_mq_node.node->ki;
        float tmp_kv = tmp_mq_node.node->kv;
        
        // if leaf node, search that node.
        if (tmp_ki == -1) {
            for (std::list<int>::iterator it=tmp_kd_node->idx_list.begin();
                it != tmp_kd_node->idx_list.end(); 
                ++it)
            {
                int idx = *it;
                const feature &_feat = feat_list[idx];
                float dist = 0.0;
                for (int i=0; i<nKey; i++) {
                    float sd = (_feat.key[i] - feat.key[i]);
                    dist += sd * sd;
                }
                if (minDist > dist) {
                    minDist = dist;
                    minNhbr = idx;
                }
            }
            ++t;
        } else {
            if (feat.key[tmp_ki] < tmp_kv) {
                mq.push( mq_node(tmp_kd_node->left, 0.0) );
                mq.push( mq_node(tmp_kd_node->right, tmp_kv - feat.key[tmp_ki]) );
            } else {
                mq.push( mq_node(tmp_kd_node->right, 0.0) );
                mq.push( mq_node(tmp_kd_node->left, feat.key[tmp_ki] - tmp_kv) );
            }
        }
    }

    return minNhbr;
}
