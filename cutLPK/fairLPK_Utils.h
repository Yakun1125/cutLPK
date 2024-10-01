#include "cutLPK_Utils.h"

void constructFairLPK(LPK& lp, Eigen::MatrixXd& dis_matrix, int N, int K, std::vector<Eigen::Triplet<int>>& basic_triplets, std::vector<std::vector<bool>>& dataGroups, std::vector<int>& groupRatio, const parameters& params);

