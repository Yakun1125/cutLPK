#include "cutLPK.h"

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <iomanip>
#include <sstream>

#include "construct_LPK.h"
#include "iterative_cutting_plane.h"
#include "branch_and_bound.h"
#include "Rounding_heuristic.h"
#include "Lloyd.h"
#include "fair_Lloyd.h"
#include "spectral_heuristic.h"
#include "separation.h"

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

    if (params.cutting_plane_warm_start > 0 || params.heuristic_only) {
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
            cutLPK_info.upper_bound = initialLloydObj;
            cutLPK_info.lower_bound = -kInfinity;
            cutLPK_info.optimality_gap = kInfinity;
            cutLPK_info.retcode = static_cast<int>(ICPStatus::HEURISTIC_ONLY);
            cutLPK_info.best_upper_bound_solution = Lloyd_Xsol;
            cutLPK_info.final_lp_Xsol = Lloyd_Xsol;

            if (params.heuristic_only) {
                OrdinaryKMeansResult result;
                result.lloyd_objective = initialLloydObj;
                result.cut_info = cutLPK_info;
                result.icp_status = ICPStatus::HEURISTIC_ONLY;
                result.assignment = createAssignment(Lloyd_Xsol, K);

                if (params.cutting_plane_verbose > 0) {
                    std::cout << "\n=== Heuristic Only Complete ===" << std::endl;
                    std::cout << "cutLPK return code: " << result.cut_info.retcode << std::endl;
                    std::cout << "Lloyd upper bound: " << std::fixed << std::setprecision(8) << initialLloydObj << std::endl;
                }
                if (!params.cutting_plane_output_file.empty() && params.cutting_plane_output_level > 0) {
                    std::ofstream file(params.cutting_plane_output_file, std::ios::app);
                    if (file.is_open()) {
                        file << "cutLPK return code: " << result.cut_info.retcode << std::endl;
                        file << "Heuristic only: Yes" << std::endl;
                        file << "Lloyd upper bound: " << std::fixed << std::setprecision(8) << initialLloydObj << std::endl;
                    }
                }
                return result;
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
        std::cout << "Fair assignment solver: " << params.fair_assignment_solver << std::endl;
    }

    // Create fair assignment solver via factory
    std::unique_ptr<FairAssignmentSolver> fairSolver =
        createFairAssignmentSolver(params.fair_assignment_solver);

    // Build fair assignment model and adjust fairness parameters
    std::vector<double> fairness_param_adjusted;

    if (params.fair_clustering_fairness_type == "alpha") {
        fairness_param_adjusted = std::vector<double>(numGroups, params.fair_clustering_fairness_param);
        fairSolver->buildAlphaModel(N, K, numGroups, dataGroups, groupRatio, fairness_param_adjusted);
    } else if (params.fair_clustering_fairness_type == "tau") {
        fairness_param_adjusted = tau_fairParam_adjustment(groupRatio, params.fair_clustering_fairness_param, N, K);
        fairSolver->buildTauModel(N, K, numGroups, dataGroups, groupRatio, fairness_param_adjusted);
    } else {
        throw std::runtime_error("Unknown fairness type: " + params.fair_clustering_fairness_type);
    }

    LPK fairlp;
    bool fairlp_constructed = false;

    // Warm start with fair Lloyd clustering
    if (params.cutting_plane_warm_start > 0 || params.heuristic_only) {
        double bestClusteringCost = kInfinity;
        std::vector<int> bestlloydAssignment;

        for (int i = 0; i < params.lloyd_num_random_starts; ++i) {
            double clusteringCost;
            std::vector<int> lloydAssignment;
            std::tie(clusteringCost, lloydAssignment) = runFairKMeans(dataPoints, K, 100000, 
                                                                       params.random_seed + i, *fairSolver);

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
            cutLPK_info.upper_bound = initialLloydObj;
            cutLPK_info.lower_bound = -kInfinity;
            cutLPK_info.optimality_gap = kInfinity;
            cutLPK_info.retcode = static_cast<int>(ICPStatus::HEURISTIC_ONLY);
            cutLPK_info.best_upper_bound_solution = Lloyd_Xsol;
            cutLPK_info.final_lp_Xsol = Lloyd_Xsol;

            if (params.heuristic_only) {
                FairKMeansResult result;
                result.lloyd_objective = initialLloydObj;
                result.cut_info = cutLPK_info;
                result.icp_status = ICPStatus::HEURISTIC_ONLY;
                result.assignment = createAssignment(Lloyd_Xsol, K);

                if (params.cutting_plane_verbose > 0) {
                    std::cout << "\n=== Heuristic Only Complete ===" << std::endl;
                    std::cout << "cutLPK return code: " << result.cut_info.retcode << std::endl;
                    std::cout << "Fair Lloyd upper bound: " << std::fixed << std::setprecision(8) << initialLloydObj << std::endl;
                }
                if (!params.cutting_plane_output_file.empty() && params.cutting_plane_output_level > 0) {
                    std::ofstream file(params.cutting_plane_output_file, std::ios::app);
                    if (file.is_open()) {
                        file << "cutLPK return code: " << result.cut_info.retcode << std::endl;
                        file << "Heuristic only: Yes" << std::endl;
                        file << "Fair Lloyd upper bound: " << std::fixed << std::setprecision(8) << initialLloydObj << std::endl;
                    }
                }
                return result;
            }
            if (!fairlp_constructed) {
                constructFairLPK(fairlp, dis_matrix, N, K, const_cast<std::vector<std::vector<bool>>&>(dataGroups),
                                 const_cast<std::vector<int>&>(groupRatio), fairness_param_adjusted, params.fair_clustering_fairness_type);
                fairlp_constructed = true;
            }
            addInitialCuts(params, N, Lloyd_Xsol, fairlp, cutting_planes);
            cutLPK_info.best_upper_bound_solution = Lloyd_Xsol;
        }
    }

    if (!fairlp_constructed) {
        constructFairLPK(fairlp, dis_matrix, N, K, const_cast<std::vector<std::vector<bool>>&>(dataGroups), 
                         const_cast<std::vector<int>&>(groupRatio), fairness_param_adjusted, params.fair_clustering_fairness_type);
    }

    fairlp.setupLPK();
    RoundingHeuristic roundingHeuristic(dataPoints, dis_matrix, K, Xsol, fairSolver.get());
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

SpectralKMeansResult solveSpectralKMeans(
    const Eigen::MatrixXd& L,
    int K,
    const parameters& params
) {
    SpectralKMeansResult result;
    
    int N = L.rows();
    
    // Construct spectral LPK
    LPK spectral_lpk;
    // Need to make a non-const copy for constructSpectralLPK
    Eigen::MatrixXd L_copy = L;
    constructSpectralLPK(spectral_lpk, L_copy, N, K);
    
    // Initialize solution matrices
    Eigen::MatrixXd Xsol; 
    Xsol.resize(N, N);
    Eigen::MatrixXd Spectral_Xsol; 
    Spectral_Xsol.resize(N, N);
    std::vector<validInequality> cutting_planes;
    
    // Run spectral heuristic
    int spectralHeuristicRetcode = spectralHeuristic(L_copy, K, Spectral_Xsol);
    double spectralObjective = L_copy.cwiseProduct(Spectral_Xsol).sum();
    
    std::cout << "Spectral heuristic completed with objective: " << spectralObjective << std::endl;
    if (params.heuristic_only) {
        result.spectral_objective = spectralObjective;
        result.cut_info.upper_bound = spectralObjective;
        result.cut_info.lower_bound = -kInfinity;
        result.cut_info.optimality_gap = kInfinity;
        result.cut_info.retcode = static_cast<int>(ICPStatus::HEURISTIC_ONLY);
        result.cut_info.best_upper_bound_solution = Spectral_Xsol;
        result.cut_info.final_lp_Xsol = Spectral_Xsol;
        result.icp_status = ICPStatus::HEURISTIC_ONLY;
        result.best_solution = Spectral_Xsol;

        if (params.cutting_plane_verbose > 0) {
            std::cout << "\n=== Heuristic Only Complete ===" << std::endl;
            std::cout << "cutLPK return code: " << result.cut_info.retcode << std::endl;
            std::cout << "Spectral heuristic upper bound: " << std::fixed << std::setprecision(8) << spectralObjective << std::endl;
        }
        if (!params.cutting_plane_output_file.empty() && params.cutting_plane_output_level > 0) {
            std::ofstream file(params.cutting_plane_output_file, std::ios::app);
            if (file.is_open()) {
                file << "cutLPK return code: " << result.cut_info.retcode << std::endl;
                file << "Heuristic only: Yes" << std::endl;
                file << "Spectral heuristic upper bound: " << std::fixed << std::setprecision(8) << spectralObjective << std::endl;
            }
        }
        return result;
    }
    
    // If the heuristic objective is less than zero (with tolerance 1e-6), it's already optimal
    if (spectralObjective < 1e-6) {
        std::cout << "Spectral heuristic found optimal solution with objective: " << spectralObjective << std::endl;
        result.spectral_objective = spectralObjective;
        result.cut_info.upper_bound = spectralObjective;
        result.cut_info.lower_bound = spectralObjective;
        result.cut_info.optimality_gap = 0.0;
        result.cut_info.retcode = static_cast<int>(ICPStatus::SUCCESS);
        result.icp_status = ICPStatus::SUCCESS;
        result.best_solution = Spectral_Xsol;

        // Write to output file
        if (!params.cutting_plane_output_file.empty() && params.cutting_plane_output_level > 0) {
            std::ofstream file(params.cutting_plane_output_file, std::ios::app);
            if (file.is_open()) {
                file << "cutLPK return code: " << result.cut_info.retcode << std::endl;
                file << "Spectral heuristic objective: " << std::fixed << std::setprecision(8) << spectralObjective << std::endl;
                file << "Final lower bound: " << std::fixed << std::setprecision(8) << result.cut_info.lower_bound << std::endl;
                file << "Final upper bound: " << std::fixed << std::setprecision(8) << result.cut_info.upper_bound << std::endl;
                file << "Final optimality gap: " << std::fixed << std::setprecision(8) << result.cut_info.optimality_gap << std::endl;
                file << "Optimal solution found by spectral heuristic." << std::endl;
                file.close();
            }
        }
        return result;
    }
    
    // Add initial cuts
    addInitialCuts(params, N, Spectral_Xsol, spectral_lpk, cutting_planes);
    spectral_lpk.setupLPK();
    
    // Setup rounding heuristic and solve info
    RoundingHeuristic roundingHeuristic(L_copy, K, Xsol);
    result.cut_info.upper_bound = spectralObjective;
    
    // Run iterative cutting plane solver
    ICPStatus retcode = iterative_cutting_plane_solver(
        N,
        K, 
        result.cut_info, 
        cutting_planes, 
        spectral_lpk, 
        roundingHeuristic,
        params
    );
    
    result.icp_status = retcode;
    result.spectral_objective = spectralObjective;
    
    if (retcode == ICPStatus::ERROR) {
        std::cerr << "Error in iterative cutting plane solver: " << static_cast<int>(retcode) << std::endl;
        return result;
    }
    
    // Print cutLPK information right after cutting plane finishes
    if (params.cutting_plane_verbose > 0) {
        std::cout << "\n=== Cutting Plane Algorithm Complete ===" << std::endl;
        std::cout << "cutLPK return code: " << result.cut_info.retcode << std::endl;
        std::cout << "Spectral heuristic objective: " << std::fixed << std::setprecision(8) << spectralObjective << std::endl;
        std::cout << "Final lower bound: " << std::fixed << std::setprecision(8) << result.cut_info.lower_bound << std::endl;
        std::cout << "Final upper bound: " << std::fixed << std::setprecision(8) << result.cut_info.upper_bound << std::endl;
        std::cout << "Final Optimality Gap: " << std::fixed << std::setprecision(8) << result.cut_info.optimality_gap << std::endl;
    }
    
    // Check if we should run Branch and Bound
    bool should_run_bnb = (params.bnb_node_limit > 0);
    
    if (should_run_bnb) {
        if (params.bnb_verbose > 0) {
            std::cout << "\n=== Starting Branch and Bound ===" << std::endl;
            std::cout << "Node limit: " << params.bnb_node_limit << std::endl;
        }
        
        result.bnb_executed = true;
        
        // Run Branch and Bound
        BnBStatus bnb_status = branch_and_bound_solver(
            N,
            K,
            result.cut_info,
            cutting_planes,
            spectral_lpk,
            roundingHeuristic,
            params
        );
        
        result.bnb_status = bnb_status;

        if (params.bnb_verbose > 0) {
            std::cout << "\n=== Branch and Bound Complete ===" << std::endl;
            if (bnb_status == BnBStatus::OPTIMAL) {
                std::cout << "Branch-and-Bound found the optimal solution." << std::endl;
            } else {
                std::cout << "Branch-and-Bound finished with status: " << static_cast<int>(bnb_status) << std::endl;
            }
        }
    }
    
    // Set the best solution
    if (result.cut_info.best_upper_bound_solution.size() > 0) {
        result.best_solution = result.cut_info.best_upper_bound_solution;
    } else {
        result.best_solution = result.cut_info.final_lp_Xsol;
    }

    // Write to output file if specified
    if (!params.cutting_plane_output_file.empty() && params.cutting_plane_output_level > 0) {
        std::ofstream file(params.cutting_plane_output_file, std::ios::app);
        if (file.is_open()) {
            file << "cutLPK return code: " << result.cut_info.retcode << std::endl;
            file << "Spectral heuristic objective: " << std::fixed << std::setprecision(8) << result.spectral_objective << std::endl;
            file << "Final lower bound: " << std::fixed << std::setprecision(8) << result.cut_info.lower_bound << std::endl;
            file << "Final upper bound: " << std::fixed << std::setprecision(8) << result.cut_info.upper_bound << std::endl;
            file << "Final optimality gap: " << std::fixed << std::setprecision(8) << result.cut_info.optimality_gap << std::endl;
            if (result.bnb_executed) {
                file << "Branch and Bound executed: Yes" << std::endl;
                file << "Branch and Bound status: " << static_cast<int>(result.bnb_status) << std::endl;
            }
            file.close();
        }
    }
    
    return result;
}
