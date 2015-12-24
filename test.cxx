#include "kd_tree.hxx"
#include <iostream>
#include <iomanip>

int main()
{
    const int nKey = 5;

    std::vector<feature> feat_list;
    for (int i=0; i< 1<<nKey; i++) {
        feature feat(i,i,nKey);
        std::cout << std::setw(3) << i << " - ";
        for (int j=0; j<nKey; j++) {
            feat.key[j] = float(i >> j & 1) * 10.0;
            std::cout << std::setw(5) << feat.key[j];
        }
        std::cout << std::endl;
        feat_list.push_back(feat);
    }

    kd_tree tree(feat_list, nKey);

    feature feat(1<<nKey, 1<<nKey, nKey);
    std::cout << " EX - ";
    for (int i=0; i<nKey; i++) {
        feat.key[i] = 2.0;
    }
    feat.key[3] = 8.0;
    feat.key[4] = 8.0;
    for (int i=0; i<nKey; i++)
        std::cout << std::setw(5) << feat.key[i];
    std::cout << std::endl;

    for (int i=0; i<feat_list.size(); i++) {
        float dist=0;
        for (int j=0; j<nKey; j++) {
            float sd = feat_list[i].key[j]-feat.key[j];
            dist += sd * sd ;
        }
        std::cout << "idx : " << std::setw(5) << i 
                  << " / dist : " << std::setw(5) << dist 
                  << std::endl;
    }

    int idx = tree.bbf_search(feat);
    std::cout << "bbf result idx : " << idx << std::endl;
}
