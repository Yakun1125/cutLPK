#pragma once
#include "Utils_Struct.h"

void constructLPK(LPK& lp, Eigen::MatrixXd& dis_matrix, int N, int K);
void constructSpectralLPK(LPK& lp, Eigen::MatrixXd& L, int N, int K);
void addInitialCuts(const parameters& params, int N, Eigen::MatrixXd& Lloyd_Xsol, LPK& lp, std::vector<validInequality>& cuts);
void constructFairLPK(LPK& lp, Eigen::MatrixXd& dis_matrix, int N, int K, std::vector<std::vector<bool>>& dataGroups, std::vector<int>& groupRatio, std::vector<double> fairness_param, std::string fairness_type);