#include "ordinary_kmeans_solver.h"

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <tuple>
#include <utility>

#include "construct_LPK.h"
#include "iterative_cutting_plane.h"
#include "branch_and_bound.h"
#include "Rounding_heuristic.h"
#include "Lloyd.h"

OrdinaryKMeansResult solveOrdinaryKMeans(
    const VectorXdList& dataPoints,
    int K,
    const parameters& params
) {
    if (dataPoints.empty()) {
        throw std::invalid_argument("Dataset must contain at least one point");
    }
    if (K <= 0 || K > static_cast<int>(dataPoints.size())) {
        throw std::invalid_argument("Number of clusters must be in [1, N]");
    }

    const int N = static_cast<int>(dataPoints.size());
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
        std::cout << "Running ordinary clustering with K = " << K << std::endl;
    }
    LPK lp;
    constructLPK(lp, dis_matrix, N, K);

    if (params.cutting_plane_warm_start > 0) {
        double bestClusteringCost = kInfinity;
        std::vector<int> bestlloydAssignment;

        for (int i = 0; i < params.lloyd_num_random_starts; ++i) {
            double clusteringCost;
            std::vector<int> lloydAssignment;
            std::tie(clusteringCost, lloydAssignment) = runKMeans(dataPoints, K, 100000, params.random_seed + i + 1);

            if (bestClusteringCost > clusteringCost) {
                bestClusteringCost = clusteringCost;
                bestlloydAssignment = std::move(lloydAssignment);
            }
        }

        if (!bestlloydAssignment.empty()) {
            initialLloydObj = bestClusteringCost;
            Lloyd_Xsol = createPartitionMatrix(bestlloydAssignment, K);
            if (verbose) {
                std::cout << "Lloyd objective: " << initialLloydObj << std::endl;
            }
            addInitialCuts(params, N, Lloyd_Xsol, lp, cutting_planes);
            cutLPK_info.best_upper_bound_solution = Lloyd_Xsol;
        }
    }

    lp.setupLPK();
    RoundingHeuristic roundingHeuristic(dataPoints, dis_matrix, K, Xsol);
    cutLPK_info.upper_bound = initialLloydObj;

    ICPStatus retcode = iterative_cutting_plane_solver(
        N,
        K,
        cutLPK_info,
        cutting_planes,
        lp,
        roundingHeuristic,
        params
    );
    if (retcode == ICPStatus::ERROR) {
        std::cerr << "Error in iterative cutting plane solver: " << static_cast<int>(retcode) << std::endl;
    }

    OrdinaryKMeansResult result;
    result.lloyd_objective = initialLloydObj;
    result.cut_info = cutLPK_info;
    result.icp_status = retcode;

    if (params.cutting_plane_verbose > 0) {
        std::cout << "\n=== Cutting Plane Algorithm Complete ===" << std::endl;
        std::cout << "cutLPK return code: " << result.cut_info.retcode << std::endl;
        std::cout << "Lloyd objective: " << std::fixed << std::setprecision(8) << initialLloydObj << std::endl;
        std::cout << "Final lower bound: " << std::fixed << std::setprecision(8) << result.cut_info.lower_bound << std::endl;
        std::cout << "Final upper bound: " << std::fixed << std::setprecision(8) << result.cut_info.upper_bound << std::endl;
        std::cout << "Final Optimality Gap: " << std::fixed << std::setprecision(8) << result.cut_info.optimality_gap << std::endl;
    }

    if (params.bnb_node_limit > 0) {
        if (params.bnb_verbose > 0) {
            std::cout << "Starting Branch-and-Bound with node limit: " << params.bnb_node_limit << std::endl;
        }
        BnBStatus bnb_status = branch_and_bound_solver(
            N,
            K,
            result.cut_info,
            cutting_planes,
            lp,
            roundingHeuristic,
            params
        );
        result.bnb_executed = true;
        result.bnb_status = bnb_status;

        if (params.bnb_verbose > 0) {
            if (bnb_status == BnBStatus::OPTIMAL) {
                std::cout << "Branch-and-Bound found the optimal solution." << std::endl;
            } else {
                std::cout << "Branch-and-Bound did not find the optimal solution." << std::endl;
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
            file << "Lloyd objective: " << std::fixed << std::setprecision(8) << initialLloydObj << std::endl;
            file << "Final lower bound: " << std::fixed << std::setprecision(8) << result.cut_info.lower_bound << std::endl;
            file << "Final upper bound: " << std::fixed << std::setprecision(8) << result.cut_info.upper_bound << std::endl;
            file << "Final optimality gap: " << std::fixed << std::setprecision(8) << result.cut_info.optimality_gap << std::endl;
            file.close();
        }
    }

    return result;
}