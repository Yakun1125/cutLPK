// Final_Version.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
#include <vector>
#include <list>
#include <random>
#include <limits>
#include "Highs.h"
#include <omp.h>
#include <string>
#include <unordered_set>


#include "ortools/base/init_google.h"
#include "ortools/pdlp/iteration_stats.h"
#include "ortools/pdlp/primal_dual_hybrid_gradient.h"
#include "ortools/pdlp/quadratic_program.h"
#include "ortools/pdlp/solve_log.pb.h"
#include "ortools/pdlp/solvers.pb.h"
//#include "gurobi_c++.h"
#include "wrapper_highs.h"
#include "../cupdlp/cupdlp.h"
#include "mps_lp.h"
#include "Lloyd.h"


constexpr double kInfinity = std::numeric_limits<double>::infinity();

struct validInequality {
	std::list<int> ineq_idx;// index start with i and then subset S
	double violation;
	double dual_value;
	validInequality(std::list<int> ineq_idx, double violation = kInfinity)
		: ineq_idx(ineq_idx), violation(violation), dual_value(0.0) {}
};

struct LPK {
	 int N; 
	std::vector<double> varLb; std::vector<double> varUb; std::vector<double> objCoef;
	std::vector<double> consLb; std::vector<double> consUb; Eigen::SparseMatrix<double, Eigen::ColMajor> ConsMatrix;
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

	std::string fairness_type;
	double fairness_param;
};

struct initializationInfo {
	unsigned long long act_cuts_size;
	int added_cuts_size;
	double Lloyd_Obj;
	double Lloyd_time;
	double cuts_Identified_time;
};

struct cutLPKSolveInfo {
	double total_pdlp_time;
	double total_post_heuristic_time;
	double total_separation_time;
	double other_time;

};

int solver_cupdlp(double& dual_obj, double& primal_obj, Eigen::MatrixXd& Xsol, std::vector<validInequality>& cuts, LPK& lp, float tolerance, float time_limit);
//int solver_gurobi(const LPK& lp, float tolerance, float time_limit);

// utility of cutLPK
void constructLPK(LPK& lp, Eigen::MatrixXd& dis_matrix, int N, int K, std::vector<Eigen::Triplet<int>>& basic_triplets);
initializationInfo addInitialCuts(const parameters& params, int N, int K, int cuts_idx_start, const std::vector<Eigen::VectorXd>& dataPoints, const std::vector<std::vector<bool>>& dataGroups, const std::vector<int>& groupRatio, Eigen::MatrixXd& Lloyd_Xsol, std::vector<Eigen::Triplet<int>>& cuts_triplets, std::vector<validInequality>& cuts);
std::vector<Eigen::VectorXd> postHeuristic(const Eigen::MatrixXd& Xsol, const std::vector<Eigen::VectorXd>& dataPoints, int k);
void separation_scheme(const Eigen::MatrixXd& Xsol, std::vector<std::list<validInequality>>& violated_cuts, int max_T, int N, int maxSize, double cuts_vio_tol);
void exact_separation_scheme(const Eigen::MatrixXd& Xsol, std::vector<std::list<validInequality>>& violated_cuts, int max_T, int N, int maxSize, double cuts_vio_tol);