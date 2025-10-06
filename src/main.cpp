#include <iostream>
#include <fstream>
#include <vector>
#include <Eigen/Dense>
#include "Solver_cupdlp.h"
#include "construct_LPK.h"
#include "separation.h"
#include "Rounding_heuristic.h"
#include "spectral_heuristic.h"
#include "iterative_cutting_plane.h"
#include <limits>
#include <chrono>
#include <unordered_map>

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <data_file> <K>" << std::endl;
        return 1;
    }
    const char* dataFile = argv[1];
    int K = std::stoi(argv[2]);

    parameters params;
    params.cutting_plane_t_upper_bound = K;
    
    // Parse additional parameters from command line
    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        auto pos = arg.find('=');
        if (pos == std::string::npos) continue;
        std::string key = arg.substr(0, pos);
        std::string value = arg.substr(pos + 1);

        if (key == "max_cuts_init") params.cutting_plane_max_cuts_firstLP = std::stoi(value);
        else if (key == "time_limit_lp") params.cutting_plane_LP_time_limit = std::stod(value);
        else if (key == "solver") params.solver = value;
        else if (key == "fairness_param") params.fair_clustering_fairness_param = std::stod(value);
        else if (key == "fairness_type") params.fair_clustering_fairness_type = value;
        else if (key == "output_level") params.cutting_plane_output_level = std::stoi(value);
        else if (key == "random_seed") params.random_seed = std::stoi(value);
        else if (key == "max_cuts_per_iter") params.cutting_plane_max_cuts_per_iter = std::stoi(value);
        else if (key == "max_cuts_added_iter") params.cutting_plane_max_cuts_added_iter = std::stoi(value);
        else if (key == "max_separation_size") params.cutting_plane_max_cuts_separation_size = std::stoi(value);
        else if (key == "max_active_cuts_size") params.cutting_plane_max_active_cuts_size = std::stoi(value);
        else if (key == "max_iter") params.cutting_plane_max_iter = std::stoi(value);
        else if (key == "warm_start") params.cutting_plane_warm_start = std::stoi(value);
        else if (key == "t_upper_bound") params.cutting_plane_t_upper_bound = std::stoi(value);
        else if (key == "initial_lp_time_limit") params.cutting_plane_firstLP_time_limit = std::stod(value);
        else if (key == "time_limit_all") params.cutting_plane_time_limit = std::stod(value);
        else if (key == "initial_solver_tol") params.cutting_plane_firstLP_solver_tol = std::stod(value);
        else if (key == "solver_tolerance_per_iter") params.cutting_plane_solver_tol = std::stod(value);
        else if (key == "lb_solver_tol") params.cutting_plane_lb_solver_tol = std::stod(value);
        else if (key == "cuts_vio_tol") params.cutting_plane_cuts_vio_tol = std::stod(value);
        else if (key == "cuts_act_tol") params.cutting_plane_cuts_act_tol = std::stod(value);
        else if (key == "opt_gap") params.cutting_plane_opt_gap = std::stod(value);
        else if (key == "lloyd_random_starts") params.lloyd_num_random_starts = std::stoi(value);
        else if (key == "is_spectral_clustering") params.is_spectral_clustering = (value == "true" || value == "1");
        else if (key == "output_file") params.cutting_plane_output_file = value;    
        else if (key == "group_file") params.fair_clustering_group_file = value;    
    }

    if (params.cutting_plane_output_file.empty()) {
        params.cutting_plane_output_file = std::string(dataFile) + "_K" + std::to_string(K);
        if (!params.fair_clustering_fairness_type.empty()) {
            params.cutting_plane_output_file += "_" + params.fair_clustering_fairness_type;
        }
        if (params.is_spectral_clustering) {
            params.cutting_plane_output_file += "_spectral";
        }
        params.cutting_plane_output_file += "_output.txt";
    }

    // time stamp to output file
    std::ofstream outputFile(params.cutting_plane_output_file, std::ios::app);
    if (!outputFile.is_open()) {
        std::cerr << "Unable to open output file: " << params.cutting_plane_output_file << std::endl;
        return 1;
    }
    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    outputFile << "Run started at: " << std::ctime(&now_c) << std::endl;
    outputFile.close();


    // ordinary clustering
    if (!params.is_spectral_clustering) {
        // Load the original dataset
        std::ifstream file(dataFile);
        if (!file.is_open()) {
            throw std::runtime_error("Unable to open file: " + std::string(dataFile));
        }
        std::vector<Eigen::VectorXd> dataPoints;
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
                    throw std::runtime_error("Invalid data format in file: " + std::string(dataFile));
                }
            }
            dataPoints.emplace_back(Eigen::Map<Eigen::VectorXd>(point.data(), point.size()));
        }
        if (dataPoints.empty()) {
            throw std::runtime_error("No data points were read from the file: " + std::string(dataFile));
        }
        file.close();

        int N = dataPoints.size();
        Eigen::MatrixXd dis_matrix; dis_matrix.resize(N, N);
        Eigen::MatrixXd Xsol;
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


        double initialLloydObj = kInfinity;
        cutLPKSolveInfo cutLPK_info;
        if (params.fair_clustering_fairness_type == ""){
            std::cout << "Running ordinary clustering with K = " << K << std::endl;
            LPK lp;
            constructLPK(lp, dis_matrix, N, K);
        
            // always perform KMeans if warm start is enabled

            if (params.cutting_plane_warm_start > 0) {
                double bestClusteringCost = kInfinity;
                std::vector<int> bestlloydAssignment;
                
                for (int i = 0; i < params.lloyd_num_random_starts; i++) {
                    double ClusteringCost;
                    std::vector<int> lloydAssignment;
                    std::tie(ClusteringCost, lloydAssignment) = runKMeans(dataPoints, K, 100000, i+3);
                    
                    if (bestClusteringCost > ClusteringCost) {
                        bestClusteringCost = ClusteringCost;
                        bestlloydAssignment = std::move(lloydAssignment);
                    }
                }
                
                initialLloydObj = bestClusteringCost;
                Lloyd_Xsol = createPartitionMatrix(bestlloydAssignment, K);
                std::cout << "Lloyd objective: " << initialLloydObj << std::endl;
                addInitialCuts(params, N, Lloyd_Xsol, lp, cutting_planes);
            }
            
            lp.setupLPK();
            RoundingHeuristic roundingHeuristic(dataPoints,dis_matrix, K, Xsol);
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
                return static_cast<int>(retcode);
            }
        }
        else{
            std::cout << "Running fair clustering with K = " << K << " and fairness type: " << params.fair_clustering_fairness_type << std::endl;
            if (params.fair_clustering_group_file.empty()) {
                throw std::runtime_error("Fair clustering requires a 'group_file' parameter.");
            }

            std::ifstream fair_file(params.fair_clustering_group_file);
            if (!fair_file.is_open())
            {
                throw std::runtime_error("Unable to open file: " + params.fair_clustering_group_file);
            }

            std::unordered_map<int, int> groupMap; // Maps group number to index in dataGroups
            int numGroups = 0;                     // Total number of groups found
            std::vector<int> groupAffiliations;    // Temp storage for group affiliation of each point
            std::vector<int> groupRatio;
            std::vector<std::vector<bool>> dataGroups;

            // Read the file line by line
            std::string fair_line;
            while (std::getline(fair_file, fair_line))
            {
                int group;
                std::stringstream ss(fair_line);
                ss >> group;

                // Check if group is new, if so, add it to the map
                if (groupMap.find(group) == groupMap.end())
                {
                    groupMap[group] = numGroups++;
                    groupRatio.push_back(0); // Initialize ratio for new group
                }

                // Increase the ratio for this group
                groupRatio[groupMap[group]]++;

                // Add group affiliation to temporary storage
                groupAffiliations.push_back(groupMap[group]);
            }

            // Now fill the dataGroups 2D vector based on groupAffiliations
            int numPoints = groupAffiliations.size();
            dataGroups.resize(numPoints, std::vector<bool>(numGroups, false));

            for (int i = 0; i < numPoints; ++i)
            {
                int groupIdx = groupAffiliations[i];
                dataGroups[i][groupIdx] = true;
            }
            fair_file.close();

            GRBEnv env = GRBEnv(true);
            env.set(GRB_IntParam_OutputFlag, 0);   
            setupGurobiWLS(env);
            env.start();

            std::cout << "numGroups: " << numGroups << std::endl;
            std::cout << "dataPoints size: " << dataPoints.size() << std::endl;

            // we have two fairness types: alpha and tau
            std::unique_ptr<GRBModel> model;
            std::vector<std::vector<GRBVar>> x_vars;
            std::vector<double> fairness_param_adjusted;

            if (params.fair_clustering_fairness_type == "alpha") {
                //fairness_param_adjusted = alpha_fairParam_adjustment(groupRatio, params.fairness_param, dataPoints.size());
                fairness_param_adjusted = std::vector<double>(numGroups, params.fair_clustering_fairness_param);
                auto result = GRB_buildFairAssignmentModel(env, K, numGroups, dataGroups, groupRatio, fairness_param_adjusted);
                model = std::move(result.first);
                x_vars = std::move(result.second);
            } else if (params.fair_clustering_fairness_type == "tau") {
                fairness_param_adjusted = tau_fairParam_adjustment(groupRatio, params.fair_clustering_fairness_param, dataPoints.size(), K);
                auto result = GRB_buildTauFairAssignmentModel(env, K, numGroups, dataGroups, groupRatio, fairness_param_adjusted);
                model = std::move(result.first);
                x_vars = std::move(result.second);
            } else {
                throw std::runtime_error("Unknown fair type: " + params.fair_clustering_fairness_type);
            }
            if (!model) {
                throw std::runtime_error("Failed to create Gurobi model for fair assignment.");
            }
            LPK fairlp;

            constructFairLPK(fairlp, dis_matrix, N, K, dataGroups, groupRatio, fairness_param_adjusted, params.fair_clustering_fairness_type);

            if (params.cutting_plane_warm_start > 0) {
                double bestClusteringCost = kInfinity;
                std::vector<int> bestlloydAssignment;
                for (int i = 0; i < params.lloyd_num_random_starts; i++) {
                    double ClusteringCost;
                    std::vector<int> lloydAssignment;
                    std::tie(ClusteringCost, lloydAssignment) = runFairKMeans(dataPoints, K, 100000, i, *model, x_vars);
                    if (bestClusteringCost > ClusteringCost) {
                        bestClusteringCost = ClusteringCost;
                        bestlloydAssignment = std::move(lloydAssignment);
                    }
                }
                initialLloydObj = bestClusteringCost;
                Lloyd_Xsol = createPartitionMatrix(bestlloydAssignment, K);
                addInitialCuts(params, N, Lloyd_Xsol, fairlp, cutting_planes);
                std::cout << "Fair Lloyd objective: " << initialLloydObj << std::endl;
            }

            
            fairlp.setupLPK();

            RoundingHeuristic roundingHeuristic(dataPoints, dis_matrix, K, Xsol, model.get(), &x_vars);            
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
                return static_cast<int>(retcode);
            }
        }
      
        // print out some final information
        // print Lloyd objective
        if (!params.cutting_plane_output_file.empty())
        {
            std::ofstream file(params.cutting_plane_output_file, std::ios::app);
            if (file.is_open())
            {
                file << "cutLPK return code: " << cutLPK_info.retcode << std::endl;
                file << "Lloyd objective: " << initialLloydObj << std::endl;
                file << "Final lower bound: " << std::fixed << std::setprecision(8) <<cutLPK_info.lower_bound << std::endl;
                file << "Final upper bound: " << std::fixed << std::setprecision(8) <<cutLPK_info.upper_bound << std::endl;
                file << "Final optimality gap: " << std::fixed << std::setprecision(8) <<cutLPK_info.optimality_gap << std::endl;
            }
        }
        std::cout << "cutLPK return code: " << cutLPK_info.retcode << std::endl;
        std::cout << "Lloyd objective: " << initialLloydObj << std::endl;
        std::cout << "Final lower bound: " << cutLPK_info.lower_bound << std::endl;
        std::cout << "Final upper bound: " << cutLPK_info.upper_bound << std::endl;
        std::cout << "Final Optimality Gap: " << cutLPK_info.optimality_gap << std::endl;
        // time information
        // std::cout << "Total solver time: " << cutLPK_info.total_solver_time << " seconds" << std::endl;
        // std::cout << "Total post-heuristic time: " << cutLPK_info.total_post_heuristic_time << " seconds" << std::endl;
        // std::cout << "Total separation time: " << cutLPK_info.total_separation_time << " seconds" << std::endl;

        // save the final solution matrix
        // by default using name of output_file with "_output" replaced by final_lp_Xsol; best_upper_bound_solution;

        std::string outputFileName = params.cutting_plane_output_file;
        std::string final_lp_Xsol = "_final_lp_Xsol";
        std::string best_upper_bound_solution = "_best_upper_bound_solution";

        // Replace "_output" with final_lp_Xsol
        size_t pos = outputFileName.find("_output");
        if (pos != std::string::npos) {
            outputFileName.replace(pos, 7, final_lp_Xsol);
        }

        // Save the final solution matrix
        std::ofstream outputFile(outputFileName);
        if (outputFile.is_open()) {
            outputFile << "Final solution matrix:\n";
            outputFile << cutLPK_info.final_lp_Xsol << std::endl;
            outputFile.close();
        }

        // Replace "_output" with best_upper_bound_solution
        pos = outputFileName.find(final_lp_Xsol);
        if (pos != std::string::npos) {
            outputFileName.replace(pos, 7, best_upper_bound_solution);
        }
        // Save the best upper bound solution matrix
        std::ofstream bestUpperBoundFile(outputFileName);
        if (bestUpperBoundFile.is_open()) {
            bestUpperBoundFile << "Best upper bound solution matrix:\n";
            bestUpperBoundFile << cutLPK_info.best_upper_bound_solution << std::endl;
            bestUpperBoundFile.close();
        }   
    }

    if (params.is_spectral_clustering) {
        std::cout << "Running spectral clustering with K = " << K << std::endl;
        std::ifstream file1(dataFile);
        if (!file1.is_open()) {
            throw std::runtime_error("Unable to open file: " + std::string(dataFile));
        }
        std::vector<std::vector<double>> matrix_rows;
        std::string line;
        while (std::getline(file1, line)) {
            std::stringstream lineStream(line);
            std::string cell;
            std::vector<double> row_data;
            while (std::getline(lineStream, cell, ',')) {
                try {
                    row_data.push_back(std::stod(cell));
                }
                catch (const std::invalid_argument& e) {
                    throw std::runtime_error("Invalid data format in CSV file: " + std::string(dataFile));
                }
            }
            if (!row_data.empty()) {
                matrix_rows.push_back(row_data);
            }
        }
        file1.close();

        if (matrix_rows.empty()) {
            throw std::runtime_error("No data was read from the file: " + std::string(dataFile));
        }

        // Determine matrix dimensions and create the Eigen Matrix 'L'.
        // N is set to the number of rows.
        int N = matrix_rows.size();
        int cols = matrix_rows[0].size();
        Eigen::MatrixXd L(N, N);
        for (int i = 0; i < N; ++i) {
            if (matrix_rows[i].size() != cols) {
                throw std::runtime_error("Inconsistent number of columns in CSV file at row " + std::to_string(i + 1));
            }
            for (int j = 0; j < cols; ++j) {
                L(i, j) = matrix_rows[i][j];
            }
        }

        LPK spectral_lpk;
        constructSpectralLPK(spectral_lpk, L, N, K);
        Eigen::MatrixXd Xsol; Xsol.resize(N, N);
        Eigen::MatrixXd Spectral_Xsol; Spectral_Xsol.resize(N, N);
        std::vector<validInequality> cutting_planes;
        
        int spectralHeuristicRetcode = spectralHeuristic(L, K, Spectral_Xsol);
        double spectralObjective = L.cwiseProduct(Spectral_Xsol).sum();
        std::cout << "Spectral heuristic completed with objective: " << spectralObjective << std::endl;
        addInitialCuts(params, N, Spectral_Xsol, spectral_lpk, cutting_planes);
        spectral_lpk.setupLPK();

        RoundingHeuristic roundingHeuristic(L, K, Xsol);
        cutLPKSolveInfo cutLPK_info;
        cutLPK_info.upper_bound = spectralObjective;
        // set default output_file as data file name with "_output.txt" suffix

    ICPStatus retcode = iterative_cutting_plane_solver(
        N,
        K, 
        cutLPK_info, 
        cutting_planes, 
        spectral_lpk, 
        roundingHeuristic,
        params
        );
    if (retcode == ICPStatus::ERROR) {
        std::cerr << "Error in iterative cutting plane solver: " << static_cast<int>(retcode) << std::endl;
        return static_cast<int>(retcode);
    }

    // print out some final information
    // print Lloyd objective
        if (!params.cutting_plane_output_file.empty())
        {
            std::ofstream file(params.cutting_plane_output_file, std::ios::app);
            if (file.is_open())
            {
                file << "cutLPK return code: " << cutLPK_info.retcode << std::endl;
                file << "Lloyd objective: " << spectralObjective << std::endl;
                file << "Final lower bound: " << std::fixed << std::setprecision(8) <<cutLPK_info.lower_bound << std::endl;
                file << "Final upper bound: " << std::fixed << std::setprecision(8) <<cutLPK_info.upper_bound << std::endl;
                file << "Final optimality gap: " << std::fixed << std::setprecision(8) <<cutLPK_info.optimality_gap << std::endl;
            }
        }
        std::cout << "cutLPK return code: " << cutLPK_info.retcode << std::endl;
        std::cout << "Lloyd objective: " << spectralObjective << std::endl;
        std::cout << "Final lower bound: " << cutLPK_info.lower_bound << std::endl;
        std::cout << "Final upper bound: " << cutLPK_info.upper_bound << std::endl;
        std::cout << "Final Optimality Gap: " << cutLPK_info.optimality_gap << std::endl;
        // time information
        // std::cout << "Total solver time: " << cutLPK_info.total_solver_time << " seconds" << std::endl;
        // std::cout << "Total post-heuristic time: " << cutLPK_info.total_post_heuristic_time << " seconds" << std::endl;
        // std::cout << "Total separation time: " << cutLPK_info.total_separation_time << " seconds" << std::endl;

        // save the final solution matrix
        // by default using name of output_file with "_output" replaced by final_lp_Xsol; best_upper_bound_solution;

        std::string outputFileName = params.cutting_plane_output_file;
        std::string final_lp_Xsol = "_final_lp_Xsol";
        std::string best_upper_bound_solution = "_best_upper_bound_solution";

        // Replace "_output" with final_lp_Xsol
        size_t pos = outputFileName.find("_output");
        if (pos != std::string::npos) {
            outputFileName.replace(pos, 7, final_lp_Xsol);
        }

        // Save the final solution matrix
        std::ofstream outputFile(outputFileName);
        if (outputFile.is_open()) {
            outputFile << "Final solution matrix:\n";
            outputFile << cutLPK_info.final_lp_Xsol << std::endl;
            outputFile.close();
        }

        // Replace "_output" with best_upper_bound_solution
        pos = outputFileName.find(final_lp_Xsol);
        if (pos != std::string::npos) {
            outputFileName.replace(pos, 7, best_upper_bound_solution);
        }
        // Save the best upper bound solution matrix
        std::ofstream bestUpperBoundFile(outputFileName);
        if (bestUpperBoundFile.is_open()) {
            bestUpperBoundFile << "Best upper bound solution matrix:\n";
            bestUpperBoundFile << cutLPK_info.best_upper_bound_solution << std::endl;
            bestUpperBoundFile.close();
        }   
    } 
    
    return 0;
}