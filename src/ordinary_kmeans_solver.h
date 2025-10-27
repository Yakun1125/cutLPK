#pragma once

#include <vector>
#include <Eigen/Dense>

#include "Utils_Struct.h"

struct OrdinaryKMeansResult {
    double lloyd_objective = kInfinity;
    cutLPKSolveInfo cut_info{};
    ICPStatus icp_status = ICPStatus::ERROR;
    bool bnb_executed = false;
    BnBStatus bnb_status = BnBStatus::ERROR;
    std::vector<int> assignment;  // best integer assignment recovered from solution
};

OrdinaryKMeansResult solveOrdinaryKMeans(
    const std::vector<Eigen::VectorXd>& dataPoints,
    int K,
    const parameters& params
);
