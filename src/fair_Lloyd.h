#pragma once
#include "Lloyd.h"
#include "Utils_Struct.h"
#include "gurobi_c++.h"
#include <memory> 
#include <numeric>

// Status for fair assignment operations
enum class FairAssignStatus {
    SUCCESS = 0,           // Assignment successful and changed
    CONVERGED = 1,         // Assignment successful but no changes (converged)
    INFEASIBLE = 2,        // Problem is infeasible
    UNBOUNDED = 3,         // Problem is unbounded
    ERROR = 4              // Other error
};

/*
Most functions can be inherent from Lloyd
We need to adjust the assignment and main iteration
*/
double find_simplified_fraction_Tau(int numerator, int K, double target_factor);

std::vector<double> tau_fairParam_adjustment(const std::vector<int> &groupRatio, double fairness_param, int N, int K);

std::pair<std::unique_ptr<GRBModel>, std::vector<std::vector<GRBVar>>> GRB_buildFairAssignmentModel(GRBEnv& env, int numClusters,
    int numGroups,
    const std::vector<std::vector<bool>>& dataGroups,
    const std::vector<int>& groupRatio,
    std::vector<double> fairness_param);

std::pair<std::unique_ptr<GRBModel>, std::vector<std::vector<GRBVar>>> GRB_buildTauFairAssignmentModel(GRBEnv& env, int numClusters,
    int numGroups,
    const std::vector<std::vector<bool>>& dataGroups,
    const std::vector<int>& groupRatio,
    std::vector<double> fairness_param);

FairAssignStatus GRB_fairAssignClusters(const VectorXdList& dataPoints, VectorXdList& centroids, 
    std::vector<int>& assignment, GRBModel& model,std::vector<std::vector<GRBVar>>& x);

std::pair<double, std::vector<int>> runFairKMeans(const VectorXdList& dataPoints, int k, int maxIterations,
     int random_seed, GRBModel& model,std::vector<std::vector<GRBVar>>& x);


