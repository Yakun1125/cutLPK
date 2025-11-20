#include "iterative_cutting_plane.h"
#include "Solver_cupdlp.h"
// #include "Solver_gurobi.h"

#include "separation.h"
#include <iomanip>
#include <chrono>
#include <iostream>
#include <fstream>

void printHeader() {
    std::cout << std::string(115, '=') << std::endl;
    std::cout << std::setw(60) << "ITERATIVE CUTTING PLANE" << std::endl;
    std::cout << std::string(115, '=') << std::endl;
    std::cout << std::setw(5) << "    "
        << std::setw(5) << "Iter"
        << std::setw(12) << "LB"
        << std::setw(12) << "UB"
        << std::setw(12) << "Gap"
        << std::setw(6) << "T"
        << std::setw(12) << "Active"
        << std::setw(12) << "Violated"
        << std::setw(12) << "Total"
        << std::setw(12) << "Solver(s)"
        << std::setw(12) << "Total(s)" << std::endl;
    std::cout << std::string(115, '-') << std::endl;
}

void saveHeaderToFile(const std::string& filename) {
    std::ofstream file(filename, std::ios::app);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }
    file << std::string(115, '=') << std::endl;
    file << std::setw(60) << "ITERATIVE CUTTING PLANE" << std::endl;
    file << std::string(115, '=') << std::endl;
    file << std::setw(5) << "    "
         << std::setw(5) << "Iter"
         << std::setw(12) << "LB"
         << std::setw(12) << "UB"
         << std::setw(12) << "Gap"
         << std::setw(6) << "T"
         << std::setw(12) << "Active"
         << std::setw(12) << "Violated"
         << std::setw(12) << "Total"
         << std::setw(12) << "Solver(s)"
         << std::setw(12) << "Total(s)" << std::endl;
    file << std::string(115, '-') << std::endl;
    file.close();
}

void printIteration(const std::vector<char>& signs, int cut_iter, double lower_bound,
    double upper_bound, double optimality_gap, double max_T,
    int cuts_active_size, int violation_size, int total_cuts,
    double solver_time, double elapsed_time) {
    // Print signs
    for (char sign : signs) {
        std::cout << sign;
    }
    std::cout << std::setw(5 - signs.size()) << "";
    std::cout << std::setw(5) << cut_iter
        << std::setw(12) << std::scientific << std::setprecision(3) << lower_bound
        << std::setw(12) << std::scientific << std::setprecision(3) << upper_bound
        << std::setw(12) << std::fixed << std::setprecision(5) << optimality_gap
        << std::setw(6) << std::fixed << std::setprecision(0) << max_T
        << std::setw(12) << cuts_active_size
        << std::setw(12) << violation_size
        << std::setw(12) << total_cuts
        << std::setw(12) << std::fixed << std::setprecision(3) << solver_time
        << std::setw(12) << std::fixed << std::setprecision(3) << elapsed_time
        << std::endl;
    std::cout << std::defaultfloat;
}

void saveIterationToFile(const std::string& filename, const std::vector<char>& signs, int cut_iter,
    double lower_bound, double upper_bound, double optimality_gap, double max_T,
    int cuts_active_size, int violation_size, int total_cuts,
    double solver_time, double elapsed_time) {
    std::ofstream file(filename, std::ios::app);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }
    
    // Print signs
    for (char sign : signs) {
        file << sign;
    }
    file << std::setw(5 - signs.size()) << "";
    file << std::setw(5) << cut_iter
         << std::setw(12) << std::scientific << std::setprecision(3) << lower_bound
         << std::setw(12) << std::scientific << std::setprecision(3) << upper_bound
         << std::setw(12) << std::fixed << std::setprecision(5) << optimality_gap
         << std::setw(6) << std::fixed << std::setprecision(0) << max_T
         << std::setw(12) << cuts_active_size
         << std::setw(12) << violation_size
         << std::setw(12) << total_cuts
         << std::setw(12) << std::fixed << std::setprecision(3) << solver_time
         << std::setw(12) << std::fixed << std::setprecision(3) << elapsed_time
         << "\n";
    
    file.close();
}

ICPStatus iterative_cutting_plane_solver( 
    int N,
    int K,
    cutLPKSolveInfo& cutLPKInfor, 
    std::vector<validInequality>& cutting_planes, 
    LPK& lp, 
    RoundingHeuristic& roundingHeuristic,
    const parameters& params,
    const std::vector<double>* primal_init,  // Initial primal solution for warm start
    const std::vector<double>* dual_init     // Initial dual solution for warm start
){
    if (params.cutting_plane_verbose >= 1) {
        printHeader();
        // save Header to params.cutting_plane_output_file if it is not ""
        if (params.cutting_plane_output_level>=1 && !params.cutting_plane_output_file.empty()) {
            saveHeaderToFile(params.cutting_plane_output_file);
        }
    }

    int cut_iter = 0;
    ICPStatus exit_status = ICPStatus::SUCCESS;
    int max_T = 2;
    int cuts_active_size = 0;
    int violation_size = 0;
    double best_primal_obj = 0.0;
    double primal_obj = 0.0;
    double upper_bound = cutLPKInfor.upper_bound;
    double lower_bound = -kInfinity;
    double last_optimality_gap = kInfinity;
    double optimality_gap = kInfinity;
    double solver_tolerance = params.cutting_plane_firstLP_solver_tol;
    double solver_time_limit = params.cutting_plane_firstLP_time_limit;

    double normValue = kInfinity;

    bool t_increased = false;
    bool tolerance_decreased = false;
    bool improvement_failed = false;
    bool solver_time_limit_increased = false;

    std::vector<int> added_cuts_record(params.cutting_plane_t_upper_bound - 1, 0);
    std::vector<int> solver_retcode_record;
    std::vector<int> t_separated_record;
    std::vector<bool> t_increased_record;
    std::vector<bool> gap_decreased_record;
    std::vector<double> gap_record;

    std::vector<double> last_primal_sol;
    std::vector<double> last_dual_sol;
    
    // Initialize with warm start solutions if provided
    if (primal_init != nullptr) {
        last_primal_sol = *primal_init;
    }
    if (dual_init != nullptr) {
        last_dual_sol = *dual_init;
    }
    
    auto cutLPK_start = std::chrono::high_resolution_clock::now();
    Eigen::MatrixXd Xsol; Xsol.resize(N, N);

    while (cut_iter < params.cutting_plane_max_iter) {
        std::vector<char> signs;
        cut_iter++;
        double current_dual_obj = 0;
        double current_primal_obj = 0;
        int solver_retcode = 0;

        // timing solver part
        auto solver_start = std::chrono::high_resolution_clock::now();
        if (cut_iter == 1){
            solver_tolerance = params.cutting_plane_firstLP_solver_tol;
        }
        else{
            solver_tolerance = params.cutting_plane_solver_tol;
        }
           
        if (params.solver == "cupdlpx") {
            std::vector<double> dual_init(lp.consLb.size(), 0.0);
            if (params.solver_warm_start && !last_dual_sol.empty()) {
                // Copy duals for non-cut constraints
                int cuts_idx_start = lp.consLb.size() - cutting_planes.size();
                for (int i = 0; i < cuts_idx_start; ++i) {
                    dual_init[i] = last_dual_sol[i];
                }
                // For cuts, use cutting_planes[cut_idx].dual_value
                for (int cut_idx = 0; cut_idx < cutting_planes.size(); ++cut_idx) {
                    dual_init[cuts_idx_start + cut_idx] = cutting_planes[cut_idx].dual_value;
                }
            }

            // Call solver with warm start
            solver_retcode = solver_cupdlpx(
                current_dual_obj, current_primal_obj, Xsol, cutting_planes, lp, solver_tolerance, solver_time_limit,
                (params.solver_warm_start && !last_primal_sol.empty() ? &last_primal_sol : nullptr),
                (params.solver_warm_start && !last_dual_sol.empty() ? &dual_init : nullptr),
                &last_primal_sol, &last_dual_sol
            );
            //solver_retcode = solver_cupdlpx(current_dual_obj, current_primal_obj, Xsol, cutting_planes, lp, solver_tolerance, solver_time_limit);
        }
        // else if (params.solver == "gurobi") {
        //     solver_retcode = solver_gurobi(current_dual_obj, current_primal_obj, Xsol, cutting_planes, lp, solver_tolerance, solver_time_limit);
        // }
        else {
            std::cerr << "Unsupported solver: " << params.solver << std::endl;
            exit_status = ICPStatus::ERROR;
            break;
        }
        auto solver_end = std::chrono::high_resolution_clock::now();
        auto solver_time = std::chrono::duration_cast<std::chrono::milliseconds>(solver_end - solver_start);
        cutLPKInfor.total_solver_time += solver_time.count() / 1e3;
        // std::cout << "Solver time: " << solver_time.count() / 1e3 << " seconds" << std::endl;

        if (solver_retcode > 1) {
            std::cout << "solve partial lpk failed with code " << solver_retcode << std::endl;
            // For now, treat severe solver failures as potential infeasibility
            // This may need refinement based on the specific solver error codes
            if (solver_retcode == 2) {
                std::cout << "Potential infeasibility detected by LP solver" << std::endl;
                exit_status = ICPStatus::INFEASIBLE;
            } else {
                exit_status = ICPStatus::ERROR;
            }
            break;
        }
        solver_retcode_record.push_back(solver_retcode);

        if (current_dual_obj > lower_bound) {
            lower_bound = current_dual_obj;
        }
        double primal_obj_improvement = (current_primal_obj - best_primal_obj) / (best_primal_obj);
        if (current_primal_obj > best_primal_obj && cut_iter >= 2) {
            best_primal_obj = current_primal_obj;
        }

        Eigen::MatrixXd normMatrix = Xsol * Xsol - Xsol;
        normValue = normMatrix.norm();

        // run rounding heuristic
        auto rounding_start = std::chrono::high_resolution_clock::now();
        roundingHeuristic.setSolutionMatrix(Xsol);
        //RoundingHeuristic roundingHeuristic(dataPoints, K, Xsol);
        if (!roundingHeuristic.run()) {
            if (roundingHeuristic.isInfeasible()) {
                std::cerr << "Rounding heuristic detected infeasibility!" << std::endl;
                exit_status = ICPStatus::INFEASIBLE;
                break;
            } else {
                std::cerr << "Rounding heuristic failed!" << std::endl;
                exit_status = ICPStatus::ERROR;
                break;
            }
        }
        Eigen::MatrixXd rounding_Xsol = roundingHeuristic.getFinalMatrix();
        double rounding_objective = roundingHeuristic.getFinalObjective();
        auto rounding_end = std::chrono::high_resolution_clock::now();
        auto rounding_time = std::chrono::duration_cast<std::chrono::milliseconds>(rounding_end - rounding_start);
        cutLPKInfor.total_post_heuristic_time += rounding_time.count() / 1e3;
        //std::cout << "Rounding heuristic time: " << rounding_time.count() / 1e3 << " seconds" << std::endl;

        if (rounding_objective < upper_bound) {
            upper_bound = rounding_objective;
            cutLPKInfor.best_upper_bound_solution = rounding_Xsol;
        }

        // check relative gap
        optimality_gap = (upper_bound - lower_bound)/upper_bound;
        gap_record.push_back(optimality_gap);
        if (optimality_gap < params.cutting_plane_opt_gap || lower_bound > params.bnb_global_ub) {
            //std::cout << "Optimality gap is within tolerance: " << optimality_gap << std::endl;
            exit_status = ICPStatus::SUCCESS;
            auto time_stamp2 = std::chrono::high_resolution_clock::now();
            auto stamp2_duration = std::chrono::duration_cast<std::chrono::milliseconds>(time_stamp2 - cutLPK_start);
            if (params.cutting_plane_verbose >= 1){
                printIteration(signs, cut_iter, lower_bound, upper_bound, optimality_gap, max_T, cuts_active_size, violation_size, cutting_planes.size(), solver_time.count() / 1e3, stamp2_duration.count() / 1e3);
            }
            if (params.cutting_plane_output_level>=1 && !params.cutting_plane_output_file.empty()) {
                saveIterationToFile(params.cutting_plane_output_file, signs, cut_iter, lower_bound, upper_bound, optimality_gap, max_T, cuts_active_size, violation_size, cutting_planes.size(), solver_time.count() / 1e3, stamp2_duration.count() / 1e3);
            }
            break;
        }

        int violation_size = 0;
        int cuts_active_size = 0;
        // update cuts
        auto updateCuts_start = std::chrono::high_resolution_clock::now();

        int update_cuts_retcode = update_cuts(lp, params, Xsol, max_T, N, signs, cutting_planes, violation_size, cuts_active_size);
        if (update_cuts_retcode != 0) {
            //std::cerr << "Error updating cuts: " << update_cuts_retcode << std::endl;
            exit_status = ICPStatus::NO_VIOLATED_CUTS;
            auto time_stamp2 = std::chrono::high_resolution_clock::now();
            auto stamp2_duration = std::chrono::duration_cast<std::chrono::milliseconds>(time_stamp2 - cutLPK_start);
            if (params.cutting_plane_verbose >= 1){
                printIteration(signs, cut_iter, lower_bound, upper_bound, optimality_gap, max_T, cuts_active_size, violation_size, cutting_planes.size(), solver_time.count() / 1e3, stamp2_duration.count() / 1e3);
            }
            if (params.cutting_plane_output_level>=1 && !params.cutting_plane_output_file.empty()) {
                saveIterationToFile(params.cutting_plane_output_file, signs, cut_iter, lower_bound, upper_bound, optimality_gap, max_T, cuts_active_size, violation_size, cutting_planes.size(), solver_time.count() / 1e3, stamp2_duration.count() / 1e3);
            }
            break;
        }
        auto updateCuts_end = std::chrono::high_resolution_clock::now();
        auto updateCuts_time = std::chrono::duration_cast<std::chrono::milliseconds>(updateCuts_end - updateCuts_start);
        cutLPKInfor.total_separation_time += updateCuts_time.count() / 1e3;
        //std::cout << "Update cuts time: " << updateCuts_time.count() / 1e3 << " seconds" << std::endl;

        lp.setupLPK();

        auto time_stamp3 = std::chrono::high_resolution_clock::now();
        auto stamp3_duration = std::chrono::duration_cast<std::chrono::milliseconds>(time_stamp3 - cutLPK_start);
        
        // update some parameters
        if (cut_iter >= 3) {
            if (solver_retcode_record[cut_iter - 1] == 1  && (last_optimality_gap-optimality_gap) < params.cutting_plane_opt_gap * 0.1) {
                int updated_time = solver_time_limit+params.cutting_plane_LP_time_limit;
                solver_time_limit = std::min(1800, updated_time);
                solver_time_limit_increased = true;
                signs.push_back('#');
            }
            else if (solver_retcode_record[cut_iter - 1] == 1 && (best_primal_obj > upper_bound || optimality_gap < params.cutting_plane_opt_gap * 10)) {
                if (solver_time_limit_increased == true) {
                    int updated_time = solver_time_limit+params.cutting_plane_LP_time_limit;
                    solver_time_limit = std::min(1800, updated_time);
                }
                else {
                    int updated_time = solver_time_limit+params.cutting_plane_LP_time_limit*0.5;
                    solver_time_limit = std::min(1800, updated_time);
                }
                solver_time_limit_increased = true;
                signs.push_back('#');
            }
            else {
                solver_time_limit_increased = false;
            }
        }


        int t_separated_current_iter = max_T;
        t_increased_record.push_back(t_increased);
        t_separated_record.push_back(t_separated_current_iter);
        if (t_increased == false && max_T < params.cutting_plane_t_upper_bound) {
            if (primal_obj_improvement < 1e-6 && best_primal_obj < upper_bound && improvement_failed == true) {
                max_T++;
                t_increased = true;
                t_increased_record[cut_iter - 1] = t_increased;
                signs.push_back('+');
            }
            else if (violation_size < 0.01 * cuts_active_size && primal_obj_improvement < 1e-6) {
                max_T++;
                t_increased = true;
                t_increased_record[cut_iter - 1] = t_increased;
                signs.push_back('+');
            }
            else if (violation_size < 0.005 * cuts_active_size) {
                max_T++;
                t_increased = true;
                t_increased_record[cut_iter - 1] = t_increased;
                signs.push_back('+');
            }
            else if ((last_optimality_gap-optimality_gap) < 1e-5) {
                max_T++;
                t_increased = true;
                t_increased_record[cut_iter - 1] = t_increased;
                signs.push_back('+');
            }
            else {
                t_increased = false;
                t_increased_record[cut_iter - 1] = t_increased;
            }
        }
        else {
            t_increased = false;
            t_increased_record[cut_iter - 1] = t_increased;
        }

        if (primal_obj_improvement < 1e-6) {
            improvement_failed = true;
        }
        else {
            improvement_failed = false;
        }


        if (optimality_gap < last_optimality_gap) {
            gap_decreased_record.push_back(true);
        }
        else {
            gap_decreased_record.push_back(false);
        }
        last_optimality_gap = optimality_gap;

        if (params.cutting_plane_verbose >= 1){
            printIteration(signs, cut_iter, lower_bound, upper_bound, optimality_gap, max_T, cuts_active_size, violation_size, cutting_planes.size(), solver_time.count() / 1e3, stamp3_duration.count() / 1e3);
        }
        if (params.cutting_plane_output_level>=1 && !params.cutting_plane_output_file.empty()) {
            saveIterationToFile(params.cutting_plane_output_file, signs, cut_iter, lower_bound, upper_bound, optimality_gap, max_T, cuts_active_size, violation_size, cutting_planes.size(), solver_time.count() / 1e3, stamp3_duration.count() / 1e3);
        }

        // terminate if stamp3_duration exceeds time limit
        if (stamp3_duration.count() / 1e3 > params.cutting_plane_time_limit) {
            //std::cout << "Time limit reached: " << params.time_limit_all << " seconds" << std::endl;
            exit_status = ICPStatus::TIME_OR_LIMIT;
            break;
        }
        // terminate if N iters gap is not improved
        int no_improve = params.cutting_plane_num_iter_no_improve;
        if (no_improve < 1)
            no_improve = 1;
        // need at least no_improve+1 iterations to compare
        if (cut_iter >= no_improve + 1)
        {
            int prev_idx = (cut_iter - 1) - no_improve; // index in gap_record for N iters before
            int curr_idx = cut_iter - 1;                // current iteration index in gap_record
            if (prev_idx >= 0 && curr_idx < static_cast<int>(gap_record.size()))
            {
                double gap_prev = gap_record[prev_idx];
                double gap_curr = gap_record[curr_idx];
                double gap_tol = params.cutting_plane_opt_gap * 0.1;
                if ((gap_prev - gap_curr) <= gap_tol &&
                    max_T == params.cutting_plane_t_upper_bound &&
                    prev_idx < static_cast<int>(t_separated_record.size()) &&
                    t_separated_record[prev_idx] == params.cutting_plane_t_upper_bound &&
                    solver_retcode == 0)
                {
                    // no meaningful improvement compared to N iterations ago
                    exit_status = ICPStatus::NO_IMPROVEMENT;
                    break;
                }
            }
        }
    }
    if (params.cutting_plane_verbose >= 1) {
        std::cout << std::string(115, '=') << std::endl;
    }

    if (cut_iter >= params.cutting_plane_max_iter && exit_status == ICPStatus::SUCCESS) {
        exit_status = ICPStatus::MAX_ITER;
        if (params.cutting_plane_verbose >= 1) {
            std::cout << "Reached maximum iterations " << params.cutting_plane_max_iter << " without meeting stopping criteria." << std::endl;
        }
        if (params.cutting_plane_output_level>=1 && !params.cutting_plane_output_file.empty()) {
            std::ofstream file(params.cutting_plane_output_file, std::ios::app);
            if (file.is_open()) {
                file << "Reached maximum iterations " << params.cutting_plane_max_iter << " without meeting stopping criteria." << std::endl;
            }
        }
    }

    cutLPKInfor.lower_bound = lower_bound;
    cutLPKInfor.upper_bound = upper_bound;
    cutLPKInfor.optimality_gap = optimality_gap;
    cutLPKInfor.final_lp_Xsol = Xsol;
    cutLPKInfor.retcode = static_cast<int>(exit_status);
    // Store final primal and dual solutions for warm starting
    cutLPKInfor.primal_solution = last_primal_sol;
    cutLPKInfor.dual_solution = last_dual_sol;

    if (params.cutting_plane_output_level>=1 && !params.cutting_plane_output_file.empty()) {
        std::ofstream file(params.cutting_plane_output_file, std::ios::app);
        if (file.is_open()) {
            file << std::string(115, '=') << std::endl;
        }
    }

    // identify active cuts reset LP
    // cut_selection_active(N, max_T, Xsol, cutting_planes, lp, params.cutting_plane_cuts_act_tol);
    // lp.setupLPK();

    return exit_status;
}