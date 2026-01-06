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
    const VectorXdList& dataPoints,
    int K,
    const parameters& params
);

struct FairKMeansResult {
    double lloyd_objective = kInfinity;
    cutLPKSolveInfo cut_info{};
    ICPStatus icp_status = ICPStatus::ERROR;
    bool bnb_executed = false;
    BnBStatus bnb_status = BnBStatus::ERROR;
    std::vector<int> assignment;  // best integer assignment recovered from solution
};

FairKMeansResult solveFairKMeans(
    const VectorXdList& dataPoints,
    int K,
    const std::vector<std::vector<bool>>& dataGroups,
    const std::vector<int>& groupRatio,
    const parameters& params
);

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
