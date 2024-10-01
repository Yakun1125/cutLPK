#pragma once
#include "cutLPK_Utils.h"
#include "fairLPK_Utils.h"
#include "Lloyd.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <fstream>
#include <string>



class KMeansClustering {
private:
	std::vector<Eigen::VectorXd> dataPoints;
	std::vector<std::vector<bool>> dataGroups;
	std::vector<int> groupRatio;
public:
    KMeansClustering(std::vector<Eigen::VectorXd>& dataPoints, int K);
	KMeansClustering(const char* filename, int K);
	void readGroupInfo(const char* filename);
	int K;
	parameters params;
	void setDefaultParams() {
        params.solver = "cupdlp";
        params.output_file = "logFile.txt";
		params.output_level = 1;
		params.random_seed = 1;
		params.max_cuts_init = 1.5e7;
		params.max_cuts_per_iter = 3e7;
		params.max_cuts_added_iter = 1e7;
		params.max_separation_size = 1.5e7;
		params.max_active_cuts_size = 1e6;
		params.warm_start = 2;
		params.t_upper_bound = K;
		params.initial_lp_time_limit = 180;
		params.time_limit_lp = 180;
		params.time_limit_all = 14400;
		params.initial_solver_tol = 1e-6;
		params.solver_tolerance_per_iter = 1e-4;
		params.lb_solver_tol = 1e-6;
		params.cuts_vio_tol = 1e-4;
		params.cuts_act_tol = 1e-4;
		params.opt_gap = 1e-4;

		params.fairness_type = "none";
		params.fairness_param = 0.5;
	}

	Eigen::MatrixXd LlyodClustering(std::vector<Eigen::VectorXd>& centroids);
	int Solve();
};
