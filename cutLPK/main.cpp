#include "cutLPK_Main.h"

int main(int argc, char* argv[]) {
	if (argc < 3) {
		std::cerr << "Usage: " << argv[0] << " <data_file> <K>" << std::endl;
		return 1;
	}

	const char* dataFile = argv[1];
	int K = std::stoi(argv[2]);

	KMeansClustering kmeans(dataFile, K);
	
	std::string fairFile = "";

	kmeans.setDefaultParams();

	// if read extra parameters from command line
	int nextArgIndex = 3;

	for (int i = nextArgIndex; i < argc; i++) {
		std::string arg = argv[i];
		std::string param_name = arg.substr(1, arg.find('=') - 1);
		std::string param_value = arg.substr(arg.find('=') + 1);

		if (param_name == "solver") {
			kmeans.params.solver = param_value;
		}
		else if (param_name == "output_file") {
			kmeans.params.output_file = param_value;
		}
		else if (param_name == "output_level") {
			kmeans.params.output_level = std::stoi(param_value);
		}
		else if (param_name == "random_seed") {
			kmeans.params.random_seed = std::stoi(param_value);
		}
		else if (param_name == "max_cuts_init") {
			kmeans.params.max_cuts_init = std::stod(param_value);
		}
		else if (param_name == "max_cuts_per_iter") {
			kmeans.params.max_cuts_per_iter = std::stod(param_value);
		}
		else if (param_name == "max_cuts_added_iter") {
			kmeans.params.max_cuts_added_iter = std::stod(param_value);
		}
		else if (param_name == "max_separation_size") {
			kmeans.params.max_separation_size = std::stod(param_value);
		}
		else if (param_name == "warm_start") {
			kmeans.params.warm_start = std::stoi(param_value);
		}
		else if (param_name == "t_upper_bound") {
			kmeans.params.t_upper_bound = std::stoi(param_value);
		}
		else if (param_name == "initial_lp_time_limit") {
			kmeans.params.initial_lp_time_limit = std::stod(param_value);
		}
		else if (param_name == "time_limit_lp") {
			kmeans.params.time_limit_lp = std::stod(param_value);
		}
		else if (param_name == "time_limit_all") {
			kmeans.params.time_limit_all = std::stod(param_value);
		}
		else if (param_name == "initial_solver_tol") {
			kmeans.params.initial_solver_tol = std::stod(param_value);
		}
		else if (param_name == "solver_tolerance_per_iter") {
			kmeans.params.solver_tolerance_per_iter = std::stod(param_value);
		}
		else if (param_name == "lb_solver_tol") {
			kmeans.params.lb_solver_tol = std::stod(param_value);
		}
		else if (param_name == "cuts_vio_tol") {
			kmeans.params.cuts_vio_tol = std::stod(param_value);
		}
		else if (param_name == "cuts_act_tol") {
			kmeans.params.cuts_act_tol = std::stod(param_value);
		}
		else if (param_name == "opt_gap") {
			kmeans.params.opt_gap = std::stod(param_value);
		}
		else if (param_name == "fairness_type") {
			kmeans.params.fairness_type = param_value;
		}
		else if (param_name == "fairness_param") {
			kmeans.params.fairness_param = std::stod(param_value);
		}
		else if (param_name == "fair_info") {
			fairFile = param_value;
		}
		else {
			std::cerr << "Invalid parameter name: " << param_name << std::endl;
			return 1;
		}
	}


	if (kmeans.params.fairness_type != "none") {
		kmeans.readGroupInfo(fairFile.c_str());
	}
	//kmeans.LlyodClustering();

	return kmeans.Solve();

	// change kmeans.params through command line arguments
	// command line arguments should be in the form of "-param_name=param_value"


}
