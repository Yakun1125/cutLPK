#include <iostream>
#include <fstream>
#include <vector>
#include <Eigen/Dense>
#include "construct_LPK.h"
#include "separation.h"
#include "Rounding_heuristic.h"
#include "spectral_heuristic.h"
#include "iterative_cutting_plane.h"
#include "branch_and_bound.h"
#include "cutLPK.h"
#include <limits>
#include <chrono>
#include <unordered_map>
#include "Utils_Struct.h"

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
        else if (key == "cutting_plane_verbose") params.cutting_plane_verbose = std::stoi(value);
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
        else if (key == "heuristic_only") params.heuristic_only = (value == "true" || value == "1" || value == "yes");
        else if (key == "exact_separation") params.cutting_plane_exact_separation = (value == "true" || value == "1" || value == "yes");
        else if (key == "output_file") params.cutting_plane_output_file = value;    
        else if (key == "bnb_output_file") params.bnb_output_file = value;    
        else if (key == "group_file") params.fair_clustering_group_file = value;   
        else if (key == "num_iter_no_improve") params.cutting_plane_num_iter_no_improve = std::stoi(value);
        else if (key == "bnb_node_limit") params.bnb_node_limit = std::stoi(value);
        else if (key == "bnb_time_limit") params.bnb_time_limit = std::stod(value);
        else if (key == "bnb_gap_tol") params.bnb_gap_tol = std::stod(value);
        else if (key == "bnb_global_ub") params.bnb_global_ub = std::stod(value); 
        else if (key == "bnb_verbose") params.bnb_verbose = std::stoi(value);
        else if (key == "bnb_output_level") params.bnb_output_level = std::stoi(value); 
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

    // Setup BnB output file with similar naming style
    if (params.bnb_output_file.empty()) {
        params.bnb_output_file = std::string(dataFile) + "_K" + std::to_string(K);
        if (!params.fair_clustering_fairness_type.empty()) {
            params.bnb_output_file += "_" + params.fair_clustering_fairness_type;
        }
        if (params.is_spectral_clustering) {
            params.bnb_output_file += "_spectral";
        }
        params.bnb_output_file += "_bnb_log.txt";
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
    VectorXdList dataPoints;
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

        const int N = static_cast<int>(dataPoints.size());
        cutLPKSolveInfo cutLPK_info{};
        double initialLloydObj = kInfinity;
        
        if (params.fair_clustering_fairness_type.empty()){
            OrdinaryKMeansResult result = solveOrdinaryKMeans(dataPoints, K, params);
            if (result.icp_status == ICPStatus::ERROR) {
                return static_cast<int>(result.icp_status);
            }
            initialLloydObj = result.lloyd_objective;
            cutLPK_info = result.cut_info;
        }
        else{
            // Load group information from file
            if (params.fair_clustering_group_file.empty()) {
                throw std::runtime_error("Fair clustering requires a 'group_file' parameter.");
            }

            std::ifstream fair_file(params.fair_clustering_group_file);
            if (!fair_file.is_open()) {
                throw std::runtime_error("Unable to open file: " + params.fair_clustering_group_file);
            }

            std::unordered_map<int, int> groupMap; // Maps group number to index in dataGroups
            int numGroups = 0;                     // Total number of groups found
            std::vector<int> groupAffiliations;    // Temp storage for group affiliation of each point
            std::vector<int> groupRatio;
            std::vector<std::vector<bool>> dataGroups;

            // Read the file line by line
            std::string fair_line;
            while (std::getline(fair_file, fair_line)) {
                int group;
                std::stringstream ss(fair_line);
                ss >> group;

                // Check if group is new, if so, add it to the map
                if (groupMap.find(group) == groupMap.end()) {
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

            for (int i = 0; i < numPoints; ++i) {
                int groupIdx = groupAffiliations[i];
                dataGroups[i][groupIdx] = true;
            }
            fair_file.close();

            // Solve fair kmeans clustering
            FairKMeansResult result = solveFairKMeans(dataPoints, K, dataGroups, groupRatio, params);
            if (result.icp_status == ICPStatus::ERROR) {
                return static_cast<int>(result.icp_status);
            }
            initialLloydObj = result.lloyd_objective;
            cutLPK_info = result.cut_info;
        }
      
        // print out some final information
        // print Lloyd objective
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
            outputFileName.replace(pos, final_lp_Xsol.length(), best_upper_bound_solution);
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

        // Solve spectral clustering using the wrapper
        SpectralKMeansResult result = solveSpectralKMeans(L, K, params);
        
        if (result.icp_status == ICPStatus::ERROR) {
            return static_cast<int>(result.icp_status);
        }

        // Print out final information

        // Save the final solution matrix
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
            outputFile << result.cut_info.final_lp_Xsol << std::endl;
            outputFile.close();
        }

        // Replace "_output" with best_upper_bound_solution
        pos = outputFileName.find(final_lp_Xsol);
        if (pos != std::string::npos) {
            outputFileName.replace(pos, final_lp_Xsol.length(), best_upper_bound_solution);
        }
        
        // Save the best upper bound solution matrix
        std::ofstream bestUpperBoundFile(outputFileName);
        if (bestUpperBoundFile.is_open()) {
            bestUpperBoundFile << "Best upper bound solution matrix:\n";
            bestUpperBoundFile << result.best_solution << std::endl;
            bestUpperBoundFile.close();
        }   
    } 
    
    return 0;
}