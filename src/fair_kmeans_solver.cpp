#include "fair_kmeans_solver.h"

#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <fstream>
#include <tuple>
#include <utility>
#include <sstream>

#include "construct_LPK.h"
#include "iterative_cutting_plane.h"
#include "branch_and_bound.h"
#include "Rounding_heuristic.h"
#include "fair_Lloyd.h"

FairKMeansResult solveFairKMeans(
    const VectorXdList& dataPoints,
    int K,
    const std::vector<std::vector<bool>>& dataGroups,
    const std::vector<int>& groupRatio,
    const parameters& params
) {
    if (dataPoints.empty()) {
        throw std::invalid_argument("Dataset must contain at least one point");
    }
    if (K <= 0 || K > static_cast<int>(dataPoints.size())) {
        throw std::invalid_argument("Number of clusters must be in [1, N]");
    }
    if (dataGroups.empty()) {
        throw std::invalid_argument("dataGroups must not be empty");
    }
    if (groupRatio.empty()) {
        throw std::invalid_argument("groupRatio must not be empty");
    }

    const int N = static_cast<int>(dataPoints.size());
    const int numGroups = static_cast<int>(groupRatio.size());
    const bool verbose = params.cutting_plane_output_level > 0;
    
    Eigen::MatrixXd dis_matrix(N, N);
    Eigen::MatrixXd Xsol = Eigen::MatrixXd::Zero(N, N);
    Eigen::MatrixXd Lloyd_Xsol = Eigen::MatrixXd::Zero(N, N);
    std::vector<validInequality> cutting_planes;

    // Compute squared Euclidean distances
    for (int i = 0; i < N; ++i) {
        dis_matrix(i, i) = 0.0;
        for (int j = i + 1; j < N; ++j) {
            dis_matrix(i, j) = (dataPoints[i] - dataPoints[j]).squaredNorm();
            dis_matrix(j, i) = dis_matrix(i, j);
        }
    }

    double initialLloydObj = kInfinity;
    cutLPKSolveInfo cutLPK_info{};

    if (verbose) {
        std::cout << "Running fair clustering with K = " << K << " and " << numGroups << " groups" << std::endl;
        std::cout << "Fairness type: " << params.fair_clustering_fairness_type << std::endl;
    }

    // Setup Gurobi environment
    GRBEnv env = GRBEnv(true);
    env.set(GRB_IntParam_OutputFlag, 0);
    setupGurobiWLS(env);
    env.start();

    // Build fair assignment model and adjust fairness parameters
    std::unique_ptr<GRBModel> model;
    std::vector<std::vector<GRBVar>> x_vars;
    std::vector<double> fairness_param_adjusted;

    if (params.fair_clustering_fairness_type == "alpha") {
        //fairness_param_adjusted = alpha_fairParam_adjustment(groupRatio, params.fair_clustering_fairness_param, N, K);
        fairness_param_adjusted = std::vector<double>(numGroups, params.fair_clustering_fairness_param);
        auto result = GRB_buildFairAssignmentModel(env, K, numGroups, dataGroups, groupRatio, fairness_param_adjusted);
        model = std::move(result.first);
        x_vars = std::move(result.second);
    } else if (params.fair_clustering_fairness_type == "tau") {
        fairness_param_adjusted = tau_fairParam_adjustment(groupRatio, params.fair_clustering_fairness_param, N, K);
        auto result = GRB_buildTauFairAssignmentModel(env, K, numGroups, dataGroups, groupRatio, fairness_param_adjusted);
        model = std::move(result.first);
        x_vars = std::move(result.second);
    } else {
        throw std::runtime_error("Unknown fairness type: " + params.fair_clustering_fairness_type);
    }

    if (!model) {
        throw std::runtime_error("Failed to create Gurobi model for fair assignment.");
    }

    // Construct the fair LPK
    LPK fairlp;
    constructFairLPK(fairlp, dis_matrix, N, K, const_cast<std::vector<std::vector<bool>>&>(dataGroups), 
                     const_cast<std::vector<int>&>(groupRatio), fairness_param_adjusted, params.fair_clustering_fairness_type);

    // Warm start with fair Lloyd clustering
    if (params.cutting_plane_warm_start > 0) {
        double bestClusteringCost = kInfinity;
        std::vector<int> bestlloydAssignment;

        for (int i = 0; i < params.lloyd_num_random_starts; ++i) {
            double clusteringCost;
            std::vector<int> lloydAssignment;
            std::tie(clusteringCost, lloydAssignment) = runFairKMeans(dataPoints, K, 100000, 
                                                                       params.random_seed + i, *model, x_vars);

            if (bestClusteringCost > clusteringCost) {
                bestClusteringCost = clusteringCost;
                bestlloydAssignment = std::move(lloydAssignment);
            }
        }

        if (!bestlloydAssignment.empty()) {
            initialLloydObj = bestClusteringCost;
            Lloyd_Xsol = createPartitionMatrix(bestlloydAssignment, K);
            if (verbose) {
                std::cout << "Fair Lloyd objective: " << initialLloydObj << std::endl;
            }
            addInitialCuts(params, N, Lloyd_Xsol, fairlp, cutting_planes);
            cutLPK_info.best_upper_bound_solution = Lloyd_Xsol;
        }
    }

    fairlp.setupLPK();
    RoundingHeuristic roundingHeuristic(dataPoints, dis_matrix, K, Xsol, model.get(), &x_vars);
    cutLPK_info.upper_bound = initialLloydObj;

    ICPStatus retcode = iterative_cutting_plane_solver(
        N,
        K,
        cutLPK_info,
        cutting_planes,
        fairlp,
        roundingHeuristic,
        params
    );
    if (retcode == ICPStatus::ERROR) {
        std::cerr << "Error in iterative cutting plane solver: " << static_cast<int>(retcode) << std::endl;
    }

    FairKMeansResult result;
    result.lloyd_objective = initialLloydObj;
    result.cut_info = cutLPK_info;
    result.icp_status = retcode;

    // Print cutLPK information right after cutting plane finishes
    if (params.cutting_plane_verbose > 0) {
        std::cout << "\n=== Cutting Plane Algorithm Complete ===" << std::endl;
        std::cout << "cutLPK return code: " << result.cut_info.retcode << std::endl;
        std::cout << "Fair Lloyd objective: " << std::fixed << std::setprecision(8) << initialLloydObj << std::endl;
        std::cout << "Final lower bound: " << std::fixed << std::setprecision(8) << result.cut_info.lower_bound << std::endl;
        std::cout << "Final upper bound: " << std::fixed << std::setprecision(8) << result.cut_info.upper_bound << std::endl;
        std::cout << "Final Optimality Gap: " << std::fixed << std::setprecision(8) << result.cut_info.optimality_gap << std::endl;
    }

    if (params.bnb_node_limit > 0) {
        if (params.bnb_verbose > 0) {
            std::cout << "\n=== Starting Branch-and-Bound ===" << std::endl;
            std::cout << "Node limit: " << params.bnb_node_limit << std::endl;
        }
        BnBStatus bnb_status = branch_and_bound_solver(
            N,
            K,
            result.cut_info,
            cutting_planes,
            fairlp,
            roundingHeuristic,
            params
        );
        result.bnb_executed = true;
        result.bnb_status = bnb_status;

        if (params.bnb_verbose > 0) {
            std::cout << "\n=== Branch-and-Bound Complete ===" << std::endl;
            if (bnb_status == BnBStatus::OPTIMAL) {
                std::cout << "Branch-and-Bound found the optimal solution." << std::endl;
            } else {
                std::cout << "Branch-and-Bound finished with status: " << static_cast<int>(bnb_status) << std::endl;
            }

        }
    }

    if (result.cut_info.best_upper_bound_solution.size() > 0) {
        result.assignment = createAssignment(result.cut_info.best_upper_bound_solution, K);
    }

    // Write to output file if specified
    if (!params.cutting_plane_output_file.empty() && params.cutting_plane_output_level > 0) {
        std::ofstream file(params.cutting_plane_output_file, std::ios::app);
        if (file.is_open()) {
            file << "cutLPK return code: " << result.cut_info.retcode << std::endl;
            file << "Fair Lloyd objective: " << std::fixed << std::setprecision(8) << initialLloydObj << std::endl;
            file << "Final lower bound: " << std::fixed << std::setprecision(8) << result.cut_info.lower_bound << std::endl;
            file << "Final upper bound: " << std::fixed << std::setprecision(8) << result.cut_info.upper_bound << std::endl;
            file << "Final optimality gap: " << std::fixed << std::setprecision(8) << result.cut_info.optimality_gap << std::endl;
            file.close();
        }
    }

    return result;
}
