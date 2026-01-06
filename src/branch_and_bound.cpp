#include "branch_and_bound.h"
#include "LP_solvers.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include "iterative_cutting_plane.h"

// BnB status printing function
void printBnBStatus(
    const BBTree& tree,
    const std::shared_ptr<BBNode>& current_node,
    double elapsed_time,
    double node_solve_time,
    const std::string& node_status,
    const std::string& cp_status,
    bool found_better_ub,
    std::ofstream* log_file,
    int verbose,
    int output_level
) {
    // output_level controls what gets printed:
    // 0: Nothing
    // 1: Only summary and key events (optimal, better UB, etc.)
    // 2: All node information (default)
    
    bool should_print_to_console = (verbose >= 1) && 
        ((output_level >= 2) || 
         (output_level == 1 && (found_better_ub || node_status == "INTEGRAL" || node_status == "INFEASIBLE")));
    
    if (!should_print_to_console && (!log_file || !log_file->is_open())) {
        return; // Nothing to print
    }
    
    double gap_percent = (tree.global_upper_bound - tree.global_lower_bound) / (std::abs(tree.global_upper_bound) + 1e-10) * 100.0;
    
    std::ostringstream line;
    line << "Node " << std::setw(4) << current_node->node_id 
         << " D:" << std::setw(2) << current_node->depth
         << " | E:" << std::setw(4) << tree.nodes_explored
         << " O:" << std::setw(4) << tree.open_nodes.size()
         << " P:" << std::setw(4) << tree.nodes_pruned
         << " | GLB:" << std::setw(12) << std::setprecision(6) << std::fixed << tree.global_lower_bound
         << " GUB:" << std::setw(12) << std::setprecision(6) << std::fixed << tree.global_upper_bound
         << " Gap:" << std::setw(8) << std::setprecision(4) << std::fixed << gap_percent << "%"
         << " | T:" << std::setw(7) << std::setprecision(2) << std::fixed << elapsed_time << "s"
         << " NT:" << std::setw(6) << std::setprecision(3) << std::fixed << node_solve_time << "s"
         << " | NLB:" << std::setw(12) << std::setprecision(6) << std::fixed << current_node->lower_bound
         << " | CP:" << std::setw(8) << cp_status
         << " | " << node_status;
    if (found_better_ub)
        line << " *UB*";
    
    // Print to console based on verbose and output level
    if (should_print_to_console) {
        std::cout << line.str() << std::endl;
    }
    
    // Print to log file if provided (always log everything to file)
    if (log_file && log_file->is_open()) {
        *log_file << line.str() << std::endl;
        log_file->flush();
    }
}

// Helper function to convert ICPStatus to string
std::string icpStatusToString(ICPStatus status) {
    switch(status) {
        case ICPStatus::SUCCESS: return "SUCCESS";
        case ICPStatus::NO_VIOLATED_CUTS: return "NO_CUTS";
        case ICPStatus::NO_IMPROVEMENT: return "NO_IMPROV";
        case ICPStatus::TIME_OR_LIMIT: return "TIME_LIM";
        case ICPStatus::ERROR: return "ERROR";
        case ICPStatus::MAX_ITER: return "MAX_ITER";
        case ICPStatus::INFEASIBLE: return "INFEASBL";
        default: return "UNKNOWN";
    }
}

void printBnBHeader(std::ofstream* log_file, int verbose) {
    if (verbose < 1) return;
    
    std::ostringstream header;
    header << "\n" << std::string(160, '=') << "\n";
    header << "                                                BRANCH AND BOUND SOLVER\n";
    header << std::string(160, '=') << "\n";
    header << "Node   D | E:Explored O:Open P:Pruned |      GLB(Global LB)      GUB(Global UB)   Gap(%) | T:Total  NT:Node |      NLB(Node LB) |   CP:Cut |   Status\n";
    header << std::string(160, '-') << "\n";
    
    // Print to console
    std::cout << header.str();
    
    // Print to log file if provided
    if (log_file && log_file->is_open()) {
        *log_file << header.str();
        log_file->flush();
    }
}

void printBnBSummary(const BBTree& tree, double total_time, BnBStatus status, std::ofstream* log_file, int verbose) {
    if (verbose < 1) return;
    
    double final_gap = (tree.global_upper_bound - tree.global_lower_bound) / (std::abs(tree.global_upper_bound) + 1e-10);
    
    std::ostringstream summary;
    summary << std::string(160, '-') << "\n";
    summary << "                               BRANCH AND BOUND SUMMARY\n";
    summary << std::string(160, '=') << "\n";
    summary << "  Status:           ";
    switch(status) {
        case BnBStatus::OPTIMAL: summary << "OPTIMAL\n"; break;
        case BnBStatus::TIME_LIMIT: summary << "TIME LIMIT\n"; break;
        case BnBStatus::ALL_NODES_EXPLORED: summary << "ALL NODES EXPLORED\n"; break;
        case BnBStatus::ERROR: summary << "ERROR\n"; break;
    }
    summary << "  Total nodes explored: " << tree.nodes_explored << "\n";
    summary << "  Total nodes pruned:   " << tree.nodes_pruned << "\n";
    summary << "  Final Global LB:      " << std::setprecision(10) << tree.global_lower_bound << "\n";
    summary << "  Final Global UB:      " << std::setprecision(10) << tree.global_upper_bound << "\n";
    summary << "  Final Gap:            " << std::setprecision(6) << (final_gap * 100) << "%\n";
    summary << "  Total Time:           " << std::setprecision(4) << total_time << " s\n";
    summary << std::string(160, '=') << "\n";
    
    // Print to console
    std::cout << summary.str();
    std::cout << std::setprecision(6); // Reset precision
    
    // Print to log file if provided
    if (log_file && log_file->is_open()) {
        *log_file << summary.str();
        log_file->flush();
    }
}

// Function to select branching variable (i,j) based on LP solution
// Returns pair (i, j) with j > i that violates clustering constraints most
std::pair<int, int> selectBranchingVariable(const Eigen::MatrixXd& Xsol, int N) {
    double best_metric = -1.0;
    int best_i = -1, best_j = -1;

    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            double x_ij = Xsol(i, j);
            // squared Euclidean distance between row i and row j
            double dist2 = (Xsol.row(i) - Xsol.row(j)).squaredNorm();
            double metric = std::min(x_ij, dist2);
            if (metric > best_metric) {
                best_metric = metric;
                best_i = i;
                best_j = j;
            }
        }
    }
    // print out Xsol(best_i, best_j), Xsol(best_i, best_i), Xsol(best_j, best_j)
    // std::cout << "Selected branching variable: (" << best_i << ", " << best_j << ")" << std::endl;
    // std::cout << "X(" << best_i << ", " << best_j << ") = " << Xsol(best_i, best_j) << std::endl;
    // std::cout << "X(" << best_i << ", " << best_i << ") = " << Xsol(best_i, best_i) << std::endl;
    // std::cout << "X(" << best_j << ", " << best_j << ") = " << Xsol(best_j, best_j) << std::endl;

    return {best_i, best_j};
}

bool isIntegerSolution(const Eigen::MatrixXd& Xsol, double tolerance) {
    Eigen::MatrixXd X_squared = Xsol * Xsol;
    Eigen::MatrixXd diff = X_squared - Xsol;
    double norm = diff.norm();
    
    return norm < tolerance;
}

void applyBranchConstraints(LPK& lp, const std::vector<BranchConstraint>& constraints, int N) {
    // Clear existing branch constraints
    lp.cons_lb_branch.clear();
    lp.cons_ub_branch.clear();
    lp.triplets_branch.clear();
    
    //std::cout << "Applying " << constraints.size() << " branch constraints:" << std::endl;
    
    // Branch constraints come AFTER basic constraints in the constraint matrix
    int branch_row_start = lp.cons_lb_basic.size();
    int constraint_row = 0;  // Relative row within branch constraints
    
    for (const auto& bc : constraints) {
        int i = bc.i;
        int j = bc.j;  // j > i by construction
        
        // Get variable indices
        int idx_ij = getPairIndex(i, j, N);
        int idx_ii = getPairIndex(i, i, N);
        int idx_jj = getPairIndex(j, j, N);
        
        if (bc.type == BranchType::SAME_CLUSTER) {
            // std::cout << "  Constraint " << (constraint_row/2 + 1) << ": points (" << i << "," << j << ") SAME_CLUSTER" << std::endl;
            // std::cout << "    Adding: X(" << i << "," << j << ") = X(" << i << "," << i << ") [matrix row " << (branch_row_start + constraint_row) << "]" << std::endl;
            // std::cout << "    Adding: X(" << i << "," << j << ") = X(" << j << "," << j << ") [matrix row " << (branch_row_start + constraint_row + 1) << "]" << std::endl;
            // Constraint 1: X(i,j) = X(i,i)  =>  X(i,j) - X(i,i) = 0
            lp.triplets_branch.emplace_back(branch_row_start + constraint_row, idx_ij, 1);
            lp.triplets_branch.emplace_back(branch_row_start + constraint_row, idx_ii, -1);
            lp.cons_lb_branch.push_back(0.0);
            lp.cons_ub_branch.push_back(0.0);
            constraint_row++;
            
            // Constraint 2: X(i,j) = X(j,j)  =>  X(i,j) - X(j,j) = 0
            lp.triplets_branch.emplace_back(branch_row_start + constraint_row, idx_ij, 1);
            lp.triplets_branch.emplace_back(branch_row_start + constraint_row, idx_jj, -1);
            lp.cons_lb_branch.push_back(0.0);
            lp.cons_ub_branch.push_back(0.0);
            constraint_row++;
            
        } else {  // DIFF_CLUSTER
            // std::cout << "  Constraint " << (constraint_row + 1) << ": points (" << i << "," << j << ") DIFF_CLUSTER" << std::endl;
            // std::cout << "    Adding: X(" << i << "," << j << ") = 0 [matrix row " << (branch_row_start + constraint_row) << "]" << std::endl;
            // Constraint: X(i,j) = 0
            lp.triplets_branch.emplace_back(branch_row_start + constraint_row, idx_ij, 1);
            lp.cons_lb_branch.push_back(0.0);
            lp.cons_ub_branch.push_back(0.0);
            constraint_row++;
        }
    }
}

std::pair<std::shared_ptr<BBNode>, std::shared_ptr<BBNode>> 
createBranchNodes(std::shared_ptr<BBNode> parent, int i, int j, BBTree& tree) {
    // Create left child: i and j in same cluster
    auto left_child = std::make_shared<BBNode>(tree.next_node_id++, parent->node_id, parent->depth + 1);
    left_child->branch_constraints = parent->branch_constraints;
    left_child->branch_constraints.emplace_back(i, j, BranchType::SAME_CLUSTER);
    
    // Create right child: i and j in different clusters
    auto right_child = std::make_shared<BBNode>(tree.next_node_id++, parent->node_id, parent->depth + 1);
    right_child->branch_constraints = parent->branch_constraints;
    right_child->branch_constraints.emplace_back(i, j, BranchType::DIFF_CLUSTER);
    
    return {left_child, right_child};
}

BnBStatus branch_and_bound_solver(
    int N,
    int K,
    cutLPKSolveInfo& cutLPKInfo,
    std::vector<validInequality>& cutting_planes,
    LPK& lp,
    RoundingHeuristic& roundingHeuristic,
    const parameters& params
){
    auto bb_start_time = std::chrono::high_resolution_clock::now();

    // Setup BnB log file
    std::ofstream bnb_log_file;
    if (!params.bnb_output_file.empty()) {
        bnb_log_file.open(params.bnb_output_file, std::ios::app);
        if (!bnb_log_file.is_open()) {
            std::cerr << "Warning: Unable to open BnB log file: " << params.bnb_output_file << std::endl;
        } else {
            // Add timestamp to log file
            auto now = std::chrono::system_clock::now();
            auto time_t = std::chrono::system_clock::to_time_t(now);
            bnb_log_file << "\n\n=== Branch and Bound Log - " << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S") << " ===\n";
        }
    }

    printBnBHeader(bnb_log_file.is_open() ? &bnb_log_file : nullptr, params.bnb_verbose);

    BBTree tree;

    auto root = std::make_shared<BBNode>(tree.next_node_id++, -1, 0);
    
    // Initialize with solution from cutting plane method
    tree.global_lower_bound = cutLPKInfo.lower_bound;
    tree.global_upper_bound = cutLPKInfo.upper_bound;
    tree.best_solution = cutLPKInfo.best_upper_bound_solution;
    
    // Set root node from cutLPK solution
    root->lower_bound = cutLPKInfo.lower_bound;
    root->upper_bound = cutLPKInfo.upper_bound;
    root->lp_solution = cutLPKInfo.final_lp_Xsol;
    root->primal_solution = cutLPKInfo.primal_solution;
    root->dual_solution = cutLPKInfo.dual_solution;
    root->is_explored = true;  // Mark as solved since we have cutLPK solution

    // print with high precision
    // std::cout << "Root node (from cutLPK): LB=" << std::fixed << std::setprecision(6) << root->lower_bound << ", UB=" << std::fixed << std::setprecision(6) << root->upper_bound << std::endl;
    // recover precision

    double initial_gap = (tree.global_upper_bound - tree.global_lower_bound) / (std::abs(tree.global_upper_bound) + 1e-10);
    if (initial_gap <= params.bnb_gap_tol || isIntegerSolution(root->lp_solution, 1e-3)) {
        std::cout << "Root solution is already optimal or integral. No branching needed." << std::endl;
        cutLPKInfo.optimality_gap = initial_gap;
        return BnBStatus::OPTIMAL;
    }

    // Branch from root
    auto [branch_i, branch_j] = selectBranchingVariable(root->lp_solution, N);
    if (branch_i == -1) {
        std::cout << "No suitable branching variable found in root solution." << std::endl;
        return BnBStatus::ERROR;
    }

    // std::cout << "*** BRANCHING ROOT on points (" << branch_i << "," << branch_j << ") ***" << std::endl;
    
    auto [left_child, right_child] = createBranchNodes(root, branch_i, branch_j, tree);
    left_child->lower_bound = root->lower_bound;
    right_child->lower_bound = root->lower_bound;
    // Pass parent's primal/dual solutions to children for warm start
    left_child->primal_solution = root->primal_solution;
    left_child->dual_solution = root->dual_solution;
    right_child->primal_solution = root->primal_solution;
    right_child->dual_solution = root->dual_solution;

    tree.addNode(left_child);
    tree.addNode(right_child);
    tree.nodes_explored = 1; // Root is considered explored

    while (!tree.open_nodes.empty()) {
        auto current_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = current_time - bb_start_time;

        if (tree.nodes_explored >= params.bnb_node_limit || elapsed.count() >= params.bnb_time_limit) {
            std::cout << "Branch and bound terminated: limits reached" << std::endl;
            auto bb_end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> bb_elapsed = bb_end_time - bb_start_time;
            tree.total_time = bb_elapsed.count();
            printBnBSummary(tree, tree.total_time, BnBStatus::TIME_LIMIT, bnb_log_file.is_open() ? &bnb_log_file : nullptr, params.bnb_verbose);
            return BnBStatus::TIME_LIMIT;
        }

        auto current_node = tree.getNextNode();
        if (!current_node){
            auto bb_end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> bb_elapsed = bb_end_time - bb_start_time;
            tree.total_time = bb_elapsed.count();
            printBnBSummary(tree, tree.total_time, BnBStatus::ALL_NODES_EXPLORED, bnb_log_file.is_open() ? &bnb_log_file : nullptr, params.bnb_verbose);
            return BnBStatus::ALL_NODES_EXPLORED;
        }

        tree.nodes_explored++;
        current_node->is_explored = true;
        
        // Start timing for this node
        auto node_start_time = std::chrono::high_resolution_clock::now();

        // std::cout << "\n--- Node " << current_node->node_id << " (depth " << current_node->depth << ") ---" << std::endl;

        applyBranchConstraints(lp, current_node->branch_constraints, N);

        if (lp.cons_lb_cuts.size() != cutting_planes.size()) {
            // std::cout<<"Warning: Mismatch in cuts size. Resetting cuts." << std::endl;
            lp.cons_lb_cuts.clear();
            lp.cons_ub_cuts.clear();
            for (int i = 0; i < cutting_planes.size(); ++i) {
                lp.cons_lb_cuts.push_back(-kInfinity);
                lp.cons_ub_cuts.push_back(0.0);
            }
        }
        if (lp.triplets_cuts.empty() && !cutting_planes.empty()) {
            // std::cout << "Warning: Mismatch in triplets size. Resetting triplets." << std::endl;
            lp.triplets_cuts.clear();
            int cuts_row_start = lp.cons_lb_basic.size() + lp.cons_lb_branch.size();
            
            for (int cut_idx = 0; cut_idx < cutting_planes.size(); ++cut_idx) {
                const auto& cut = cutting_planes[cut_idx];
                int cut_row = cuts_row_start + cut_idx;
                
                auto it = cut.ineq_idx.begin();
                int firstElement = *it;
                int firstTerm = firstElement * (2 * N - firstElement + 1) / 2;
                lp.triplets_cuts.emplace_back(cut_row, firstTerm, -1);

                for (auto j = std::next(it); j != cut.ineq_idx.end(); ++j) {
                    int currentJ = *j;
                    int temp_i = std::min(firstElement, currentJ);
                    int temp_j = std::max(firstElement, currentJ);
                    int indexValue = temp_i * (2 * N - temp_i + 1) / 2 + temp_j - temp_i;
                    lp.triplets_cuts.emplace_back(cut_row, indexValue, 1);

                    for (auto k = std::next(j); k != cut.ineq_idx.end(); ++k) {
                        int currentK = *k;
                        lp.triplets_cuts.emplace_back(cut_row, currentJ * (2 * N - currentJ + 1) / 2 + currentK - currentJ, -1);
                    }
                }
            }
        }

        std::vector<validInequality> node_cutting_planes = cutting_planes;

        cutLPKSolveInfo node_info;
        node_info.lower_bound = current_node->lower_bound;
        node_info.upper_bound = kInfinity;

        roundingHeuristic.setConstraints(current_node->branch_constraints);

        parameters node_params = params;
        node_params.cutting_plane_opt_gap = node_params.bnb_gap_tol;
        node_params.cutting_plane_max_iter = node_params.bnb_cut_iter_limit;
        node_params.cutting_plane_time_limit = node_params.bnb_time_limit;
        node_params.bnb_global_ub = tree.global_upper_bound;
        node_params.cutting_plane_exact_separation = false;
        node_params.cutting_plane_max_separation_time = 30;
        node_params.cutting_plane_num_iter_no_improve = 1;
        node_params.cutting_plane_verbose = 0;
        node_params.cutting_plane_output_level = 0;
        node_params.cutting_plane_lb_solver_tol = 1e-5;
        node_params.cutting_plane_max_cuts_added_iter = node_cutting_planes.size();
        node_params.cutting_plane_remove_inactive_cuts = false;


        // std::cout<<"cutLPK opt gap tol: "<<node_params.cutting_plane_opt_gap<<", max iter: "<<node_params.cutting_plane_max_iter
        //          <<", time limit: "<<node_params.cutting_plane_time_limit<<"s"<<std::endl;

        lp.setupLPK();
        
        // Prepare warm start from current node's primal/dual solutions
        const std::vector<double>* primal_init = nullptr;
        const std::vector<double>* dual_init = nullptr;
        if (!current_node->primal_solution.empty()) {
            primal_init = &current_node->primal_solution;
        }
        if (!current_node->dual_solution.empty()) {
            dual_init = &current_node->dual_solution;
        }
        
        ICPStatus cut_retcode = iterative_cutting_plane_solver(
            N, K, node_info, node_cutting_planes, lp, roundingHeuristic, node_params,
            primal_init, dual_init  // Pass warm start solutions
        );

        if (cut_retcode == ICPStatus::ERROR) {
            std::cout << "Error solving LP at node " << current_node->node_id << std::endl;
            auto bb_end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> bb_elapsed = bb_end_time - bb_start_time;
            tree.total_time = bb_elapsed.count();
            printBnBSummary(tree, tree.total_time, BnBStatus::ERROR, bnb_log_file.is_open() ? &bnb_log_file : nullptr, params.bnb_verbose);
            return BnBStatus::ERROR;
        }
        
        if (cut_retcode == ICPStatus::INFEASIBLE) {
            // End timing for this node
            auto node_end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> node_elapsed = node_end_time - node_start_time;
            double node_solve_time = node_elapsed.count();
            std::chrono::duration<double> total_elapsed = node_end_time - bb_start_time;
            
            current_node->is_pruned = true;
            tree.nodes_pruned++;
            tree.pruneNodes();
            tree.updateGlobalLowerBound();
            
            std::string node_status = "INFEASIBLE";
            std::string cp_status = icpStatusToString(cut_retcode);
            printBnBStatus(tree, current_node, total_elapsed.count(), node_solve_time, node_status, cp_status, false, bnb_log_file.is_open() ? &bnb_log_file : nullptr, params.bnb_verbose, params.bnb_output_level);
            continue;
        }

        // End timing for this node
        auto node_end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> node_elapsed = node_end_time - node_start_time;
        double node_solve_time = node_elapsed.count();

        // Track if we found a better upper bound
        double previous_global_ub = tree.global_upper_bound;

        // Update node with results 
        if (node_info.lower_bound > current_node->lower_bound) {
            current_node->lower_bound = node_info.lower_bound;
        }
        current_node->lp_solution = node_info.final_lp_Xsol;
        current_node->upper_bound = node_info.upper_bound;
        // Store primal and dual solutions from this node's solve
        current_node->primal_solution = node_info.primal_solution;
        current_node->dual_solution = node_info.dual_solution;
        cutting_planes = node_cutting_planes;

                // Update global bounds
        // tree.global_lower_bound = std::min(tree.global_lower_bound, current_node->lower_bound);
        bool found_better_ub = false;
        if (current_node->upper_bound < tree.global_upper_bound) {
            tree.global_upper_bound = current_node->upper_bound;
            tree.best_solution = node_info.best_upper_bound_solution;
            found_better_ub = true;
        }
        
        // Determine node status and print status
        std::string node_status;
        std::string cp_status = icpStatusToString(cut_retcode);
        std::chrono::duration<double> total_elapsed = node_end_time - bb_start_time;
        
        // std::cout << "Node " << current_node->node_id << " results: LB=" << std::setprecision(10) <<current_node->lower_bound 
        //           << ", UB=" <<std::setprecision(10) << current_node->upper_bound << std::endl;

            // check if we should prune this node. If node_info.lower_bound > tree.global_upper_bound, prune
        if (node_info.lower_bound > tree.global_upper_bound) {
            current_node->is_pruned = true;
            tree.nodes_pruned++;
            tree.pruneNodes();
            tree.updateGlobalLowerBound();
            node_status = "PRUNED";
            printBnBStatus(tree, current_node, total_elapsed.count(), node_solve_time, node_status, cp_status, found_better_ub, bnb_log_file.is_open() ? &bnb_log_file : nullptr, params.bnb_verbose, params.bnb_output_level);
            continue;
        }

        if (isIntegerSolution(current_node->lp_solution, 1e-3)) {
            current_node->is_integer = true;
            tree.updateGlobalLowerBound();
            node_status = "INTEGRAL";
            printBnBStatus(tree, current_node, total_elapsed.count(), node_solve_time, node_status, cp_status, found_better_ub, bnb_log_file.is_open() ? &bnb_log_file : nullptr, params.bnb_verbose, params.bnb_output_level);
            continue;
        }

        auto [branch_i, branch_j] = selectBranchingVariable(current_node->lp_solution, N);
        if (branch_i != -1) {
            auto [left_child, right_child] = createBranchNodes(current_node, branch_i, branch_j, tree);
            left_child->lower_bound = current_node->lower_bound;
            right_child->lower_bound = current_node->lower_bound;
            // Pass parent's primal/dual solutions to children for warm start
            left_child->primal_solution = current_node->primal_solution;
            left_child->dual_solution = current_node->dual_solution;
            right_child->primal_solution = current_node->primal_solution;
            right_child->dual_solution = current_node->dual_solution;
            tree.addNode(left_child);
            tree.addNode(right_child);
            node_status = "BRANCHING";
        } else {
            node_status = "NO_BRANCH_VAR";
        }
        
        tree.updateGlobalLowerBound();
        tree.pruneNodes();

        // Print status for this node
        printBnBStatus(tree, current_node, total_elapsed.count(), node_solve_time, node_status, cp_status, found_better_ub, bnb_log_file.is_open() ? &bnb_log_file : nullptr, params.bnb_verbose, params.bnb_output_level);
        
                // Check optimality gap
        double current_gap = (tree.global_upper_bound - tree.global_lower_bound) / (std::abs(tree.global_upper_bound) + 1e-10);
        if (current_gap <= params.bnb_gap_tol) {
            std::cout << "\n*** Optimality gap tolerance reached! ***" << std::endl;
            break;
        }
    }

    auto bb_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> bb_elapsed = bb_end_time - bb_start_time;
    tree.total_time = bb_elapsed.count();

    printBnBSummary(tree, tree.total_time, BnBStatus::OPTIMAL, bnb_log_file.is_open() ? &bnb_log_file : nullptr, params.bnb_verbose);

    return BnBStatus::OPTIMAL;
}