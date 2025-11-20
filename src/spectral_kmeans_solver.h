#pragma once

#include <vector>
#include <Eigen/Dense>

#include "Utils_Struct.h"

struct SpectralKMeansResult {
    double spectral_objective = kInfinity;
    cutLPKSolveInfo cut_info{};
    ICPStatus icp_status = ICPStatus::ERROR;
    bool bnb_executed = false;
    BnBStatus bnb_status = BnBStatus::ERROR;
    Eigen::MatrixXd best_solution;  // best solution matrix
};

SpectralKMeansResult solveSpectralKMeans(
    const Eigen::MatrixXd& L,
    int K,
    const parameters& params
);
