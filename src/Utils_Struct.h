#pragma once
#include <vector>
#include <list>
#include <string>
#include <limits>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>

#include <iostream>
#include <iomanip>

constexpr double kInfinity = std::numeric_limits<double>::infinity();

struct validInequality {
	std::vector<int> ineq_idx;// index start with i and then subset S
	double violation;
	double dual_value;
    // Default constructor
    validInequality() : violation(kInfinity), dual_value(0.0) {}

    // Parameterized constructor
    validInequality(std::vector<int> ineq_idx, double violation = kInfinity)
        : ineq_idx(std::move(ineq_idx)), violation(violation), dual_value(0.0) {}
};

// struct LPK {
// 	 int N; 
// 	std::vector<double> varLb; std::vector<double> varUb; std::vector<double> objCoef;
// 	std::vector<double> consLb; std::vector<double> consUb; Eigen::SparseMatrix<double, Eigen::ColMajor> ConsMatrix;
// 	std::vector<double> cons_lb_basic; std::vector<double> cons_ub_basic;
// 	std::vector<double> cons_lb_cuts; std::vector<double> cons_ub_cuts;
// 	std::vector<Eigen::Triplet<int>> triplets_basic;
// 	std::vector<Eigen::Triplet<int>> triplets_cuts;
// };

class LPK{
	public:
		 int N; 
	std::vector<double> varLb; std::vector<double> varUb; std::vector<double> objCoef;
	std::vector<double> consLb; std::vector<double> consUb; Eigen::SparseMatrix<double, Eigen::ColMajor> ConsMatrix;
    std::vector<double> cons_lb_basic; std::vector<double> cons_ub_basic;
	std::vector<double> cons_lb_cuts; std::vector<double> cons_ub_cuts;
	std::vector<Eigen::Triplet<int>> triplets_basic;
	std::vector<Eigen::Triplet<int>> triplets_cuts;

	//methods to combine sparsematrix and cons bounds
	void setupLPK(){
			std::vector<Eigen::Triplet<int>> combinedTriplets;
			combinedTriplets.reserve(triplets_basic.size() + triplets_cuts.size());
			combinedTriplets.insert(combinedTriplets.end(), triplets_basic.begin(), triplets_basic.end());
			combinedTriplets.insert(combinedTriplets.end(), triplets_cuts.begin(), triplets_cuts.end());
			ConsMatrix.resize(cons_lb_basic.size() + cons_lb_cuts.size(), N * (N + 1) / 2);
			ConsMatrix.setFromTriplets(combinedTriplets.begin(), combinedTriplets.end());
			triplets_cuts.clear();

			consLb.assign(cons_lb_basic.begin(), cons_lb_basic.end());
			consLb.insert(consLb.end(), cons_lb_cuts.begin(), cons_lb_cuts.end());

			// Combine upper bounds
			consUb.assign(cons_ub_basic.begin(), cons_ub_basic.end());
			consUb.insert(consUb.end(), cons_ub_cuts.begin(), cons_ub_cuts.end());

			cons_lb_cuts.clear();
            cons_ub_cuts.clear();
	};

    void printDebugInfo() const {
        std::cout << "\n--- LPK Debug Info ---" << std::endl;
        std::cout << std::left << std::setw(25) << "N:" << N << std::endl;
        
        std::cout << "\n--- Variable Vectors ---" << std::endl;
        std::cout << std::left << std::setw(25) << "varLb.size():" << varLb.size() << std::endl;
        std::cout << std::left << std::setw(25) << "varUb.size():" << varUb.size() << std::endl;
        std::cout << std::left << std::setw(25) << "objCoef.size():" << objCoef.size() << std::endl;
        
        std::cout << "\n--- Final Constraint Vectors ---" << std::endl;
        std::cout << std::left << std::setw(25) << "consLb.size():" << consLb.size() << std::endl;
        std::cout << std::left << std::setw(25) << "consUb.size():" << consUb.size() << std::endl;
        
        std::cout << "\n--- Basic Constraint Parts ---" << std::endl;
        std::cout << std::left << std::setw(25) << "cons_lb_basic.size():" << cons_lb_basic.size() << std::endl;
        std::cout << std::left << std::setw(25) << "cons_ub_basic.size():" << cons_ub_basic.size() << std::endl;
        std::cout << std::left << std::setw(25) << "triplets_basic.size():" << triplets_basic.size() << std::endl;

        std::cout << "\n--- Cut Constraint Parts ---" << std::endl;
        std::cout << std::left << std::setw(25) << "cons_lb_cuts.size():" << cons_lb_cuts.size() << std::endl;
        std::cout << std::left << std::setw(25) << "cons_ub_cuts.size():" << cons_ub_cuts.size() << std::endl;
        std::cout << std::left << std::setw(25) << "triplets_cuts.size():" << triplets_cuts.size() << std::endl;

        std::cout << "\n--- Final Constraint Matrix ---" << std::endl;
        std::cout << std::left << std::setw(25) << "ConsMatrix.rows():" << ConsMatrix.rows() << std::endl;
        std::cout << std::left << std::setw(25) << "ConsMatrix.cols():" << ConsMatrix.cols() << std::endl;
        std::cout << std::left << std::setw(25) << "ConsMatrix.nonZeros():" << ConsMatrix.nonZeros() << std::endl;
        std::cout << "------------------------\n" << std::endl;
    }

};

struct parameters {
	std::string solver;
	std::string output_file;
	int output_level;
	int random_seed;
	int max_cuts_init;
	int max_cuts_per_iter;
	int max_cuts_added_iter;
	int max_separation_size;
	int max_active_cuts_size;
	int warm_start;
	int t_upper_bound;
	double initial_lp_time_limit;
	double time_limit_lp;
	double time_limit_all;
	double initial_solver_tol;
	double solver_tolerance_per_iter;
	double lb_solver_tol;
	double cuts_vio_tol;
	double cuts_act_tol;
	double opt_gap;
	double max_separation_time;

    int lloyd_random_starts;
	std::string fairness_type;
	double fairness_param;
	std::string group_file;
	bool is_spectral_clustering;
	    // Constructor with default values
		parameters() : 
        solver("cupdlp"),
        output_file(""),
        output_level(3),
        random_seed(42),
        max_cuts_init(1.5e7),
        max_cuts_per_iter(3e7),
        max_cuts_added_iter(1e7),
        max_separation_size(1.5e7),
        //max_active_cuts_size(500),
        warm_start(1),
        t_upper_bound(10),
        initial_lp_time_limit(180.0),
        time_limit_lp(180.0),
        time_limit_all(7200.0),
        initial_solver_tol(1e-6),
        solver_tolerance_per_iter(1e-6),
        lb_solver_tol(1e-6),
        cuts_vio_tol(1e-4),
        cuts_act_tol(1e-4),
        opt_gap(1e-4),
        lloyd_random_starts(100),
        fairness_type(""),
		is_spectral_clustering(false),
        fairness_param(1.0),
        group_file(""),
        max_separation_time(300.0)
    {}
};

struct cutLPKSolveInfo {
	double total_solver_time;
	double total_post_heuristic_time;
	double total_separation_time;
	double lower_bound;
	double upper_bound;
	double optimality_gap;
	int retcode; // 0 optimal; 1 no violated cuts; 2 no improvement; 3 time or iter limit reached; 4 error;
	Eigen::MatrixXd best_upper_bound_solution;
    Eigen::MatrixXd final_lp_Xsol;
};