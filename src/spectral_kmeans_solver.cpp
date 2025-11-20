#include "spectral_kmeans_solver.h"
#include "construct_LPK.h"
#include "separation.h"
#include "Rounding_heuristic.h"
#include "spectral_heuristic.h"
#include "iterative_cutting_plane.h"
#include "branch_and_bound.h"
#include <iostream>
#include <iomanip>
#include <fstream>

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
