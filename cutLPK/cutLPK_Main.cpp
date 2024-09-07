#include "cutLPK_Main.h"
#include <iostream>
#include <iomanip>

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
        << std::setw(12) << std::fixed << std::setprecision(3) << lower_bound
        << std::setw(12) << std::fixed << std::setprecision(3) << upper_bound
        << std::setw(12) << std::fixed << std::setprecision(4) << optimality_gap
        << std::setw(6) << std::fixed << std::setprecision(0) << max_T
        << std::setw(12) << cuts_active_size
        << std::setw(12) << violation_size
        << std::setw(12) << total_cuts
        << std::setw(12) << std::fixed << std::setprecision(3) << solver_time
        << std::setw(12) << std::fixed << std::setprecision(3) << elapsed_time
        << std::endl;
}

KMeansClustering::KMeansClustering(std::vector<Eigen::VectorXd>& dataPoints, int K)
    : dataPoints(dataPoints), K(K) {
    if (K <= 1) {
        throw std::invalid_argument("K must be a positive integer >= 2");
    }
    if (dataPoints.empty()) {
        throw std::invalid_argument("Data points vector cannot be empty");
    }
}

KMeansClustering::KMeansClustering(const char* filename, int K) : K(K) {
    if (K <= 1) {
        throw std::invalid_argument("K must be a positive integer >= 2");
    }
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file: " + std::string(filename));
    }

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        std::vector<double> point;
        while (std::getline(lineStream, cell, ',')) {
            try {
                point.push_back(std::stod(cell));
            }
            catch (const std::invalid_argument& e) {
                throw std::runtime_error("Invalid data format in file: " + std::string(filename));
            }
        }
        dataPoints.emplace_back(Eigen::Map<Eigen::VectorXd>(point.data(), point.size()));
    }

    if (dataPoints.empty()) {
        throw std::runtime_error("No data points were read from the file: " + std::string(filename));
    }
}

int KMeansClustering::Solve() {
    auto alg_start = std::chrono::high_resolution_clock::now();

    int N = dataPoints.size();
    Eigen::MatrixXd dis_matrix;dis_matrix.resize(N, N);
    Eigen::MatrixXd Xsol; Xsol.resize(N, N);
    Eigen::MatrixXd Lloyd_Xsol; Lloyd_Xsol.resize(N, N);
    std::vector<validInequality> cutting_planes;


    // Compute squared Euclidean distances_
    for (int i = 0; i < N; ++i) {
        dis_matrix(i, i) = 0;
        for (int j = i + 1; j < N; ++j) {
            dis_matrix(i, j) = (dataPoints[i] - dataPoints[j]).squaredNorm();
            dis_matrix(j, i) = dis_matrix(i, j);
        }
    }


    
    LPK lp;
    // objective function, variables bounds and basic constraints
    std::vector<Eigen::Triplet<int>> triplets_basic;
    constructLPK(lp, dis_matrix, N, K, triplets_basic);

    // consMatrix for cuts
    int cuts_idx_start = N + 1;
    std::vector<Eigen::Triplet<int>> triplets_cuts;
    initializationInfo initInfo = addInitialCuts(params, N, K, cuts_idx_start, dataPoints, Lloyd_Xsol, triplets_cuts, cutting_planes);

    // if output level == 2, output the initInfo to log file and print to consol; if output level == 1, only print to consol; if output level == 0, do nothing
    if (params.output_level == 2) {
		std::ofstream logFile(params.output_file, std::ios::app);
		logFile << "Initialization Information: " << std::endl;
		//logFile << "Number of active cuts: " << initInfo.act_cuts_size << std::endl;
		logFile << "Number of added cuts: " << initInfo.added_cuts_size << std::endl;
		logFile << "Lloyd Objective: " << initInfo.Lloyd_Obj << std::endl;
		logFile.close();
	}
    else if (params.output_level >= 1) {
		std::cout << "Initialization Information: " << std::endl;
		//std::cout << "Number of active cuts: " << initInfo.act_cuts_size << std::endl;
		std::cout << "Number of added cuts: " << initInfo.added_cuts_size << std::endl;
		std::cout << "Lloyd Objective: " << initInfo.Lloyd_Obj << std::endl;
	}

    
    // merge basic and cuts constraints
    std::vector<Eigen::Triplet<int>> combinedTriplets;
    combinedTriplets.reserve(triplets_basic.size() + triplets_cuts.size());
    combinedTriplets.insert(combinedTriplets.end(), triplets_basic.begin(), triplets_basic.end());
    combinedTriplets.insert(combinedTriplets.end(), triplets_cuts.begin(), triplets_cuts.end());
    lp.ConsMatrix.resize(cuts_idx_start + initInfo.added_cuts_size, N * (N + 1) / 2);
    lp.ConsMatrix.setFromTriplets(combinedTriplets.begin(), combinedTriplets.end());
    std::vector<Eigen::Triplet<int>>().swap(triplets_cuts);

    std::vector<double> cons_lb_basic = lp.consLb;
    std::vector<double> cons_ub_basic = lp.consUb;
    std::vector<double> cons_lb_cuts(initInfo.added_cuts_size, -kInfinity);
    std::vector<double> cons_ub_cuts(initInfo.added_cuts_size, 0);
    lp.consLb.insert(lp.consLb.end(), cons_lb_cuts.begin(), cons_lb_cuts.end());
    lp.consUb.insert(lp.consUb.end(), cons_ub_cuts.begin(), cons_ub_cuts.end());



    /**********************************************
     Iterative Cutting Plane Part
    **********************************************/
    
    if (params.output_level >= 1) {
        printHeader();
    }
    int cut_iter = 0;
    int exit_status = 0;
    int max_T = 2;
    int cuts_active_size = 0;
    int violation_size = 0;
    double best_primal_obj = 0.0;
    double primal_obj = 0.0;
    double upper_bound = initInfo.Lloyd_Obj;
    double lower_bound = -kInfinity;
    double optimality_gap = kInfinity;
    double solver_tolerance = params.initial_solver_tol;
    double solver_time_limit = params.initial_lp_time_limit;
    double normValue = kInfinity;

    bool t_increased = false;
    bool tolerance_decreased = false;
    bool improvement_failed = false;
    bool solver_time_limit_increased = false;

    std::vector<int> added_cuts_record(params.t_upper_bound - 1, 0);
    std::vector<int> solver_retcode_record;
    std::vector<bool> t_increased_record;

    Eigen::MatrixXd partitionMatrix = Lloyd_Xsol;

    auto cutLPK_start = std::chrono::high_resolution_clock::now();


    while (true) {
        std::vector<char> signs;
        cut_iter++;
        double current_lb = 0;
        int solver_retcode = 0;

        // timing solver part
        auto solver_start = std::chrono::high_resolution_clock::now();
        if (params.solver == "cupdlp") {
            solver_retcode = solver_cupdlp(current_lb, primal_obj, Xsol, cutting_planes, lp, solver_tolerance, solver_time_limit);
        }
        else {
            std::cout << "solver not supported" << std::endl;
            exit_status = 1;
            break;
        }
        auto solver_end = std::chrono::high_resolution_clock::now();
        auto solver_time = std::chrono::duration_cast<std::chrono::milliseconds>(solver_end - solver_start);
        auto curelapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(solver_end - cutLPK_start);
        

        if (solver_retcode > 1) {
            std::cout << "solve partial lpk failed" << std::endl;
            exit_status = 1;
            break;
        }
        solver_retcode_record.push_back(solver_retcode);

        if (current_lb > lower_bound) {
            lower_bound = current_lb;
        }
        double primal_obj_improvement = (primal_obj - best_primal_obj) / (best_primal_obj);

        if (primal_obj > best_primal_obj) {
            best_primal_obj = primal_obj;
        }

        Eigen::MatrixXd normMatrix = Xsol * Xsol - Xsol;
        normValue = normMatrix.norm();

        if ((normValue < params.cuts_vio_tol * 10 && (Xsol-partitionMatrix).norm()>params.cuts_vio_tol) || (primal_obj_improvement < 1e-6 && improvement_failed == false)) {
            auto post_heuristic_start = std::chrono::high_resolution_clock::now();
            Eigen::MatrixXd temp_partitionMatrix = postHeuristic(Xsol, dataPoints, K, 1e7);
            Eigen::MatrixXd cost_matrix = temp_partitionMatrix.cwiseProduct(dis_matrix);
            double temp_upper_bound = cost_matrix.sum() / 2;
            auto post_heuristic_end = std::chrono::high_resolution_clock::now();
            auto post_heuristic_duration = std::chrono::duration_cast<std::chrono::milliseconds>(post_heuristic_end - post_heuristic_start);

            signs.push_back('h');

            if (temp_upper_bound < upper_bound) {
				upper_bound = temp_upper_bound;
                if ((temp_partitionMatrix - partitionMatrix).norm() > 0.0) {
                    partitionMatrix = temp_partitionMatrix;
                }
			}
        }

        optimality_gap = (upper_bound - lower_bound) / upper_bound;

        if (optimality_gap < params.opt_gap || normValue < params.cuts_vio_tol) {
            auto time_stamp2 = std::chrono::high_resolution_clock::now();
            auto stamp2_duration = std::chrono::duration_cast<std::chrono::milliseconds>(time_stamp2 - cutLPK_start);
            if (params.output_level >= 1) { printIteration(signs, cut_iter, lower_bound, upper_bound, optimality_gap, max_T, cuts_active_size, violation_size, cutting_planes.size(), solver_time.count() / 1e3, stamp2_duration.count() / 1e3); }
            break;
        }

        if (cut_iter == 1) {// start actual iteration, recover tolerance and time limit for solver
            solver_tolerance = params.solver_tolerance_per_iter;
            solver_time_limit = params.time_limit_lp;
        }

        // separating cuts
        auto separation_start = std::chrono::high_resolution_clock::now();
        std::vector<std::list<validInequality>> violated_cuts(max_T - 1);
        violation_size = 0;
        while (true) {
            separation_scheme(Xsol, violated_cuts, max_T, N, params.max_separation_size, params.cuts_vio_tol);
            for (int i = 0; i < max_T - 1; ++i) {
                violation_size += violated_cuts[i].size();
            }
            if (violation_size != 0 || max_T == params.t_upper_bound) {
                break;
            }
            else {
                if (max_T < params.t_upper_bound) {
                    //std::cout << "no violated cuts found for t up to: " << max_T << " increase it" << std::endl;
                    max_T++;
                    signs.push_back('+');
                    violated_cuts.resize(max_T - 1);
                }
            }
        }
        auto separation_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> separation_time = std::chrono::duration_cast<std::chrono::milliseconds>(separation_end - separation_start);


        // before constructing new lp, check stopping criteria
        tolerance_decreased = false;
        if (violation_size == 0 && max_T == params.t_upper_bound) {
            if (solver_tolerance <= params.lb_solver_tol && solver_retcode == 0) {
                exit_status = 1;
                break;
			}
            else {
                if (solver_tolerance > params.lb_solver_tol + 1e-8) {
                    tolerance_decreased = true;
                    solver_tolerance = solver_tolerance * 0.01;
                    signs.push_back('!');
                }
                if (solver_retcode == 1) {
                    solver_time_limit = params.time_limit_all;
                }
			}
        }

        auto time_stamp1 = std::chrono::high_resolution_clock::now();
        auto stamp1_duration = std::chrono::duration_cast<std::chrono::milliseconds>(time_stamp1 - cutLPK_start);
        if (stamp1_duration.count() > params.time_limit_all * 1e3) {
            exit_status = 2;
            break;
        }


        auto cut_selection_start = std::chrono::high_resolution_clock::now();
        // identify active cuts
        std::vector<validInequality> active_cuts;
        active_cuts.reserve(cutting_planes.size());
        std::vector<Eigen::Triplet<int>> kept_cuts_triplets;
        kept_cuts_triplets.reserve(cutting_planes.size()* (max_T + 2)); // Estimate max size

        int cuts_idx = 0;
        for (const auto& element : cutting_planes) {
            if (std::abs(element.violation) < params.cuts_act_tol) {
                active_cuts.push_back(element);
                int newRow = cuts_idx_start + cuts_idx;
                auto it = element.ineq_idx.begin();
                int firstElement = *it;
                int firstTerm = firstElement * (2 * N - firstElement + 1) / 2;
                kept_cuts_triplets.emplace_back(newRow, firstTerm, -1);

                for (auto j = std::next(it); j != element.ineq_idx.end(); ++j) {
                    int currentJ = *j;
                    int temp_i = std::min(firstElement, currentJ);
                    int temp_j = std::max(firstElement, currentJ);
                    int indexValue = temp_i * (2 * N - temp_i + 1) / 2 + temp_j - temp_i;
                    kept_cuts_triplets.emplace_back(newRow, indexValue, 1);

                    for (auto k = std::next(j); k != element.ineq_idx.end(); ++k) {
                        int currentK = *k;
                        kept_cuts_triplets.emplace_back(newRow, currentJ * (2 * N - currentJ + 1) / 2 + currentK - currentJ, -1);
                    }
                }
                ++cuts_idx;
            }
        }
        kept_cuts_triplets.shrink_to_fit();
        cutting_planes = active_cuts;
        vector<validInequality>().swap(active_cuts);

        cuts_active_size = cuts_idx;

        // sorting and adding violated cuts
        int remaining_capacity = std::max(params.max_cuts_per_iter - cuts_idx, 0);// add as many as possible but control the size of LP
        remaining_capacity = std::min(remaining_capacity, params.max_cuts_added_iter);
        if (remaining_capacity == 0) {
            remaining_capacity += 100000;
        }
        std::vector<Eigen::Triplet<int>> violated_cuts_triplets;

        if (violation_size <= remaining_capacity) {
            cutting_planes.reserve(cutting_planes.size() + violation_size);
            violated_cuts_triplets.reserve(violation_size * (max_T + 2)); // Estimate max size

            for (int i = 0; i < violated_cuts.size(); i++) {
                added_cuts_record[i] += violated_cuts[i].size();
                for (const auto& element : violated_cuts[i]) {
                    cutting_planes.push_back(element);
                    int newRow = cuts_idx_start + cuts_idx;
                    auto it = element.ineq_idx.begin();
                    int firstElement = *it;
                    int firstTerm = firstElement * (2 * N - firstElement + 1) / 2;
                    violated_cuts_triplets.emplace_back(newRow, firstTerm, -1);

                    for (auto j = std::next(it); j != element.ineq_idx.end(); ++j) {
						int currentJ = *j;
						int temp_i = std::min(firstElement, currentJ);
						int temp_j = std::max(firstElement, currentJ);
						int indexValue = temp_i * (2 * N - temp_i + 1) / 2 + temp_j - temp_i;
						violated_cuts_triplets.emplace_back(newRow, indexValue, 1);

                        for (auto k = std::next(j); k != element.ineq_idx.end(); ++k) {
							int currentK = *k;
							violated_cuts_triplets.emplace_back(newRow, currentJ * (2 * N - currentJ + 1) / 2 + currentK - currentJ, -1);
						}
					}
                    ++cuts_idx;
				}
            }
        }
        else {
            cutting_planes.reserve(cutting_planes.size() + remaining_capacity);
            violated_cuts_triplets.reserve(remaining_capacity * (max_T + 2)); // Estimate max size

            std::vector<validInequality> violated_cuts_sorted;
            violated_cuts_sorted.reserve(violation_size);
            for (int i = 0; i < violated_cuts.size(); i++) {
                for (const auto& element : violated_cuts[i]) {
					violated_cuts_sorted.push_back(element);
				}
			}
            // sorting based on violation
            std::sort(violated_cuts_sorted.begin(), violated_cuts_sorted.end(), [](const validInequality& a, const validInequality& b) {
				return a.violation < b.violation;
			});
            // add remaining capacity of most violated cuts
            for (int i = 0; i < remaining_capacity; i++) {
                added_cuts_record[violated_cuts_sorted[i].ineq_idx.size()-3] += 1;
                cutting_planes.push_back(violated_cuts_sorted[i]);
                int newRow = cuts_idx_start + cuts_idx;
                auto it = violated_cuts_sorted[i].ineq_idx.begin();
                int firstElement = *it;
                int firstTerm = firstElement * (2 * N - firstElement + 1) / 2;
                violated_cuts_triplets.emplace_back(newRow, firstTerm, -1);

                for (auto j = std::next(it); j != violated_cuts_sorted[i].ineq_idx.end(); ++j) {
                    int currentJ = *j;
                    int temp_i = std::min(firstElement, currentJ);
                    int temp_j = std::max(firstElement, currentJ);
                    int indexValue = temp_i * (2 * N - temp_i + 1) / 2 + temp_j - temp_i;
                    violated_cuts_triplets.emplace_back(newRow, indexValue, 1);

                    for (auto k = std::next(j); k != violated_cuts_sorted[i].ineq_idx.end(); ++k) {
                        int currentK = *k;
                        violated_cuts_triplets.emplace_back(newRow, currentJ * (2 * N - currentJ + 1) / 2 + currentK - currentJ, -1);
                    }
                }
                ++cuts_idx;
            }
        }
        violated_cuts_triplets.shrink_to_fit();

        auto cut_selection_end = std::chrono::high_resolution_clock::now();
        auto cut_selection_time = std::chrono::duration_cast<std::chrono::milliseconds>(cut_selection_end - cut_selection_start);


        // merge active and violated cuts

        std::vector<Eigen::Triplet<int>> combined_cuts_triplets;
        combined_cuts_triplets.reserve(triplets_basic.size()+kept_cuts_triplets.size() + violated_cuts_triplets.size());
        combined_cuts_triplets.insert(combined_cuts_triplets.end(), triplets_basic.begin(), triplets_basic.end());
        combined_cuts_triplets.insert(combined_cuts_triplets.end(), kept_cuts_triplets.begin(), kept_cuts_triplets.end());
        combined_cuts_triplets.insert(combined_cuts_triplets.end(), violated_cuts_triplets.begin(), violated_cuts_triplets.end());
        lp.ConsMatrix.resize(cutting_planes.size()+cons_lb_basic.size(), N* (N + 1) / 2);
        lp.ConsMatrix.setFromTriplets(combined_cuts_triplets.begin(), combined_cuts_triplets.end());
        std::vector<Eigen::Triplet<int>>().swap(kept_cuts_triplets);
        std::vector<Eigen::Triplet<int>>().swap(violated_cuts_triplets);



        // update consLb and consUb
        lp.consLb = cons_lb_basic;
        lp.consUb = cons_ub_basic;
        std::vector<double> cons_lb_newcuts(cutting_planes.size(), -kInfinity);
        std::vector<double> cons_ub_newcuts(cutting_planes.size(), 0);
        lp.consLb.insert(lp.consLb.end(), cons_lb_newcuts.begin(), cons_lb_newcuts.end());
        lp.consUb.insert(lp.consUb.end(), cons_ub_newcuts.begin(), cons_ub_newcuts.end());


        // update some parameters
        // consider decrase the tolerance
        if (tolerance_decreased == false && solver_tolerance > params.lb_solver_tol + 1e-8 && current_lb < lower_bound) {
            if (optimality_gap < params.opt_gap * 10 || normValue < params.cuts_vio_tol * 10) {
                solver_tolerance = solver_tolerance * 0.01;
                tolerance_decreased = true;
                signs.push_back('!');
            }
            else if(optimality_gap < params.opt_gap * 100 && ((upper_bound - best_primal_obj)/upper_bound)<1e-8){
                solver_tolerance = solver_tolerance * 0.01;
                tolerance_decreased = true;
                signs.push_back('!');
            }
            else if (best_primal_obj > upper_bound && cut_iter >= 3) {
                solver_tolerance = solver_tolerance * 0.01;
                tolerance_decreased = true;
                signs.push_back('!');
            }
            else if (improvement_failed == true && primal_obj_improvement < 1e-6 && max_T == params.t_upper_bound) {
                solver_tolerance = solver_tolerance * 0.01;
                tolerance_decreased = true;
                signs.push_back('!');
            }
        }

        // consider increase time limit
        if (cut_iter >= 3) {
            if (solver_retcode_record[cut_iter - 1] == 1 && solver_retcode_record[cut_iter - 2] == 1) {
                if (solver_time_limit_increased == true){
                solver_time_limit += params.time_limit_lp;
                }
                else{
                solver_time_limit += params.time_limit_lp * 0.5;
                }
                solver_time_limit_increased = true;
                signs.push_back('#');
            }
            else if (solver_retcode_record[cut_iter - 1] == 1 && (best_primal_obj > upper_bound || optimality_gap < params.opt_gap * 10)) {
                  if (solver_time_limit_increased == true){
                  solver_time_limit += params.time_limit_lp;
                  }
                  else{
                  solver_time_limit += params.time_limit_lp * 0.5;
                  }
                  solver_time_limit_increased = true;
                  signs.push_back('#');
            }
            else{
              solver_time_limit_increased = false;
            }
        }

        // consider increase the max_T
        int t_separated_current_iter = max_T;
        t_increased_record.push_back(t_increased);
        if (t_increased == false && max_T < params.t_upper_bound) {
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
	    
        // print info
        auto time_stamp3 = std::chrono::high_resolution_clock::now();
        auto stamp3_duration = std::chrono::duration_cast<std::chrono::milliseconds>(time_stamp3 - cutLPK_start);
        if (params.output_level >= 1) { printIteration(signs, cut_iter, lower_bound, upper_bound, optimality_gap, t_separated_current_iter, cuts_active_size, violation_size, cutting_planes.size(), solver_time.count() / 1e3, stamp3_duration.count() / 1e3); }
    }

    auto cutLPK_end = std::chrono::high_resolution_clock::now();
    auto elpased_time = std::chrono::duration_cast<std::chrono::milliseconds>(cutLPK_end - alg_start);
    double time_used = elpased_time.count() / 1e3;
    std::cout << std::string(115, '=') << std::endl;
    std::cout << std::defaultfloat <<"cutting plane solved the problem in " << cut_iter << " iterations with " << time_used << " seconds" << std::endl;
    std::cout << "status: " << exit_status << std::endl;
    std::cout <<  "relative gap: "<<optimality_gap<<std::endl;
    std::cout <<  "obj: " << std::fixed << std::setprecision(4) << upper_bound << std::endl;
    std::cout << std::defaultfloat;
    //std::cout << " Kmeans++ sol same as final sol: " << (Lloyd_Xsol - partitionMatrix).norm() << std::endl;
    std::cout << "added cuts record: ";
    for (int i = 0; i < params.t_upper_bound - 1; ++i) {
        std::cout << "t = " << i + 2 << ": " << added_cuts_record[i] << " ";
    }

    return 0;
}


