#include "construct_LPK.h"
#include "Lloyd.h"
#include <chrono>

void constructLPK(LPK& lp, Eigen::MatrixXd& dis_matrix, int N, int K) {
	lp.N = N;
	int numVars = N * (N + 1) / 2;
	lp.varLb = std::vector<double>(numVars, 0.0);
	lp.varUb = std::vector<double>(numVars, 1.0);
	lp.objCoef = std::vector<double>(numVars, 0.0);
	Eigen::SparseMatrix<double, Eigen::ColMajor> ConsMatrix;// constraint matrix
	int index = 0; // index mapping logic is (i,j) j>i index = i*(2*N-i+1)/2+j-i
	for (int i = 0; i < N; ++i) {
		for (int j = i; j < N; ++j) {
			lp.objCoef[index++] = dis_matrix(i, j);
		}
	}
	// basic part constraint bounds
	// lp.consLb = std::vector<double> (1 + N, 1);
	// lp.consUb = std::vector<double> (1 + N, 1);
	// lp.consLb[0] = K;
	// lp.consUb[0] = K;

    lp.cons_lb_basic = std::vector<double>(N + 1, 1.0);
    lp.cons_ub_basic = std::vector<double>(N + 1, 1.0);
    lp.cons_lb_basic[0] = K;
    lp.cons_ub_basic[0] = K;

	// Reserve space for triplets based on an estimate of non-zero elements
    std::vector<Eigen::Triplet<int>> basic_triplets;
	basic_triplets.reserve((N + 1) * N);
	// Add non-zeros for the original matrix part
	for (int i = 0; i < N; ++i) {
		int col = i * (2 * N - i + 1) / 2;
		basic_triplets.emplace_back(0, col, 1); // First row, diagonal elements set to 1
	}

	for (int i = 1; i <= N; ++i) {
		for (int j = 0; j < N; ++j) {
			int col = std::min(i - 1, j) * (2 * N - std::min(i - 1, j) + 1) / 2 + std::max(i - 1, j) - std::min(i - 1, j);
			basic_triplets.emplace_back(i, col, 1); // Subsequent rows
		}
	}

    lp.triplets_basic = basic_triplets;
}

void constructSpectralLPK(LPK& lp, Eigen::MatrixXd& L, int N, int K){
    	lp.N = N;
	int numVars = N * (N + 1) / 2;
	lp.varLb = std::vector<double>(numVars, 0.0);
	lp.varUb = std::vector<double>(numVars, 1.0);
	lp.objCoef = std::vector<double>(numVars, 0.0);
	Eigen::SparseMatrix<double, Eigen::ColMajor> ConsMatrix;// constraint matrix
	int index = 0; // index mapping logic is (i,j) j>i index = i*(2*N-i+1)/2+j-i
	for (int i = 0; i < N; ++i) {
		for (int j = i; j < N; ++j) {
            if (i==j){
                lp.objCoef[index++] = L(i, j);
            }
            else{
                lp.objCoef[index++] = 2*L(i, j);
            }
		}
	}
	// basic part constraint bounds
	// lp.consLb = std::vector<double> (1 + N, 1);
	// lp.consUb = std::vector<double> (1 + N, 1);
	// lp.consLb[0] = K;
	// lp.consUb[0] = K;

    lp.cons_lb_basic = std::vector<double>(N + 1, 1.0);
    lp.cons_ub_basic = std::vector<double>(N + 1, 1.0);
    lp.cons_lb_basic[0] = K;
    lp.cons_ub_basic[0] = K;

	// Reserve space for triplets based on an estimate of non-zero elements
    std::vector<Eigen::Triplet<int>> basic_triplets;
	basic_triplets.reserve((N + 1) * N);
	// Add non-zeros for the original matrix part
	for (int i = 0; i < N; ++i) {
		int col = i * (2 * N - i + 1) / 2;
		basic_triplets.emplace_back(0, col, 1); // First row, diagonal elements set to 1
	}

	for (int i = 1; i <= N; ++i) {
		for (int j = 0; j < N; ++j) {
			int col = std::min(i - 1, j) * (2 * N - std::min(i - 1, j) + 1) / 2 + std::max(i - 1, j) - std::min(i - 1, j);
			basic_triplets.emplace_back(i, col, 1); // Subsequent rows
		}
	}

    lp.triplets_basic = basic_triplets;
}

void constructFairLPK(LPK& lp, Eigen::MatrixXd& dis_matrix, int N, int K, std::vector<std::vector<bool>>& dataGroups, std::vector<int>& groupRatio, std::vector<double> fairness_param, std::string fairness_type) {
    lp.N = N;
    int numVars = N * (N + 1) / 2;
    int numGroups = groupRatio.size();
    lp.varLb = std::vector<double>(numVars, 0.0);
    lp.varUb = std::vector<double>(numVars, 1.0);
    lp.objCoef = std::vector<double>(numVars, 0.0);
    
    int index = 0; 
    for (int i = 0; i < N; ++i) {
        for (int j = i; j < N; ++j) {
            lp.objCoef[index++] = dis_matrix(i, j);
        }
    }
    
    // First key fix: Use the correct size for constraint vectors
    int constraintSize = 1 + N + (numGroups * N);
    // if (std::abs(params.fairness_param - 1.0) <= 1e-6) {
    //     // For exact fairness, we use numGroups-1 groups
    //     constraintSize = 1 + N + ((numGroups-1) * N);
    // } else {
    //     // For inequality fairness, we need space for all groups
    //     constraintSize = 1 + N + (numGroups * N);
    // }
    
    lp.cons_lb_basic = std::vector<double>(constraintSize, 1);
    lp.cons_ub_basic = std::vector<double>(constraintSize, 1);

    lp.cons_lb_basic[0] = K;
    lp.cons_ub_basic[0] = K;
    
    std::vector<double> normalized_groupRatio(numGroups, 0.0);
    for (int g = 0; g < numGroups; g++) {
        normalized_groupRatio[g] = double(groupRatio[g]) / double(N);
    }
    
    int baseIndex = 1 + N;
    if (fairness_type == "alpha"){
        for (int g = 0; g < numGroups; g++) {        
            for (int i = 0; i < N; i++) {
                int index = baseIndex + g * N + i;
                if (index >= lp.cons_lb_basic.size()) {
                    return;
                }
                lp.cons_lb_basic[index] = normalized_groupRatio[g] * fairness_param[g];
                lp.cons_ub_basic[index] = normalized_groupRatio[g] / fairness_param[g];
            }
        }
    }
    else if(fairness_type == "tau"){
        for (int g = 0; g < numGroups; g++) {           
            for (int i = 0; i < N; i++) {
                int index = baseIndex + g * N + i;
                if (index >= lp.cons_lb_basic.size()) {
                    return;
                }
                lp.cons_lb_basic[index] = 0;//normalized_groupRatio[g] * params.group_factor[g];
                lp.cons_ub_basic[index] = 300;//std::numeric_limits<double>::infinity();
            }
        }
    }
    // Reserve space for triplets based on an estimate of non-zero elements
    int estimated_nonzeros = (N + 1) * N;
    for (int g = 0; g < numGroups; g++) {
        estimated_nonzeros += groupRatio[g] * N;
    }
    
    std::vector<Eigen::Triplet<int>> basic_triplets;
    basic_triplets.reserve(estimated_nonzeros);
    // Add non-zeros for the original matrix part
    for (int i = 0; i < N; ++i) {
        int col = i * (2 * N - i + 1) / 2;
        basic_triplets.emplace_back(0, col, 1); // First row, diagonal elements set to 1
    }

    for (int i = 1; i <= N; ++i) {
        for (int j = 0; j < N; ++j) {
            int col = std::min(i - 1, j) * (2 * N - std::min(i - 1, j) + 1) / 2 + std::max(i - 1, j) - std::min(i - 1, j);
            basic_triplets.emplace_back(i, col, 1); // Subsequent rows
        }
    }
    // Add non-zeros for the fairness constraints
    if (fairness_type == "alpha"){
        for (int g = 0; g < numGroups; g++) {
            for (int j = 0; j < N; j++) {
                for (int i = 0; i < N; ++i) {
                    int col = std::min(i, j) * (2 * N - std::min(i, j) + 1) / 2 + std::max(i, j) - std::min(i, j);
                    if (dataGroups[i][g]) {
                        int index = baseIndex + g * N + j;
                        if (index >= lp.cons_lb_basic.size()) {
                            return;
                        }
                        basic_triplets.emplace_back(index, col, 1);
                    }
                }
            }
        }
    }
    else if (fairness_type == "tau"){
        for (int g = 0; g < numGroups; g++)
        {
            for (int j = 0; j < N; j++)
            {
                int index = baseIndex + g * N + j;
                int nonzero_count = 0; // counter for (g, j)
                for (int i = 0; i < N; ++i)
                {
                    int col = std::min(i, j) * (2 * N - std::min(i, j) + 1) / 2 + std::max(i, j) - std::min(i, j);
                    if (i == j)
                    {
                        if (index >= lp.cons_lb_basic.size()){return;}                          
                        basic_triplets.emplace_back(index, col, dataGroups[i][g] - fairness_param[g] * groupRatio[g]);
                        nonzero_count++;
                    }
                    else if (dataGroups[i][g])
                    {
                        if (index >= lp.cons_lb_basic.size()){return;}                           
                        basic_triplets.emplace_back(index, col, 1);
                        nonzero_count++;
                    }
                }
            }
        }
    }

    lp.triplets_basic = basic_triplets;

}

// Helper function to add a cut to avoid code duplication
inline void addCut(std::vector<validInequality>& cuts, std::vector<Eigen::Triplet<int>>& cuts_triplets, 
    int newRow, int N, int i, int j, int k, double violation) {
cuts.emplace_back(validInequality(std::vector<int>{i, j, k}, violation));

// Compute indices once to avoid repeating calculation
const int ij_idx = std::min(i, j) * (2 * N - std::min(i, j) + 1) / 2 + std::max(i, j) - std::min(i, j);
const int ik_idx = std::min(i, k) * (2 * N - std::min(i, k) + 1) / 2 + std::max(i, k) - std::min(i, k);
const int ii_idx = i * (2 * N - i + 1) / 2;
const int jk_idx = j * (2 * N - j + 1) / 2 + k - j;

cuts_triplets.emplace_back(newRow, ij_idx, 1);
cuts_triplets.emplace_back(newRow, ik_idx, 1);
cuts_triplets.emplace_back(newRow, ii_idx, -1);
cuts_triplets.emplace_back(newRow, jk_idx, -1);
}

void addInitialCuts(const parameters& params, int N, Eigen::MatrixXd& Lloyd_Xsol, 
     LPK& lp, std::vector<validInequality>& cuts) {
    int cuts_idx_start = lp.cons_lb_basic.size();

    //initializationInfo initInfo;
    const unsigned long long totalCombinations = static_cast<unsigned long long>(N) * (N - 1) * (N - 2) / 2;
    const int initial_size = static_cast<int>(std::min(totalCombinations, static_cast<unsigned long long>(params.cutting_plane_max_cuts_firstLP)));

    //std::cout<< "Total combinations: " << totalCombinations << std::endl;
    
    // Reserve memory upfront to avoid reallocations
    std::vector<Eigen::Triplet<int>> cuts_triplets;
    cuts.reserve(initial_size);
    cuts_triplets.reserve(4 * initial_size);
    
    int added_count = 0;
    unsigned long long scaned_count = 0;
    unsigned long long act_size = 0;
    
    // Timing Identify cuts
    auto start = std::chrono::high_resolution_clock::now();

    if (params.cutting_plane_warm_start == 2) {
        // Random selection approach or approach for warm_start == 2
        std::mt19937 gen(params.random_seed);
        std::uniform_real_distribution<double> dis(0.0, 1.0);
        
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < i; ++j) {  // Optimization: Only check j < i to avoid duplicate triplets
                for (int k = j + 1; k < N; ++k) {
                    if (k == i) continue;
                    
                    double violation = 0.0;
                    //if (params.warm_start > 0) {
                        violation = Lloyd_Xsol(i, j) + Lloyd_Xsol(i, k) - Lloyd_Xsol(i, i) - Lloyd_Xsol(j, k);
                    //}

                    if (std::abs(violation) < params.cutting_plane_cuts_act_tol) {
                        unsigned long long remaining_combinations = totalCombinations - scaned_count;
                        double p = static_cast<double>(initial_size - added_count) / static_cast<double>(remaining_combinations);
                        act_size++;
                        
                        if (dis(gen) < p && added_count < initial_size) {
                            addCut(cuts, cuts_triplets, cuts_idx_start + added_count, N, i, j, k, violation);
                            added_count++;
                        }
                    }
                    
                    scaned_count++;
                }
            }
        }
    } else {
        // Warm start approach 1: Add first size_each_i triangle inequalities for each i,j
        const int size_each_i = params.cutting_plane_max_cuts_firstLP / N;
        
        for (int i = 0; i < N && added_count < initial_size; i++) {
            int added_count_i = 0;
            
            for (int j = 0; j < N && added_count_i < size_each_i; j++) {
                if (i == j) continue;
                
                for (int k = j + 1; k < N; ++k) {
                    if (k == i) continue;
                    
                    double violation = Lloyd_Xsol(i, j) + Lloyd_Xsol(i, k) - Lloyd_Xsol(i, i) - Lloyd_Xsol(j, k);
                    
                    if (std::abs(violation) < params.cutting_plane_cuts_act_tol) {//
                        addCut(cuts, cuts_triplets, cuts_idx_start + added_count, N, i, j, k, violation);//
                        added_count_i++;
                        added_count++;
                        act_size++;
                    }
                    
                    if (added_count_i >= size_each_i) break;
                }
            }
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto cuts_Identified_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    //std::cout<< "Identified " << added_count << " cuts in " << cuts_Identified_time << " ms." << std::endl;
    std::cout<<"Initial cuts size: " << added_count<< std::endl;

    lp.triplets_cuts = cuts_triplets;
    lp.cons_lb_cuts = std::vector<double>(cuts.size(), -kInfinity);
    lp.cons_ub_cuts = std::vector<double>(cuts.size(), 0.0);
}

void updateLPConstraints(
    LPK& lp,
    const std::vector<Eigen::Triplet<int>>& basicTriplets,
    const std::vector<Eigen::Triplet<int>>& additionalTriplets,
    const std::vector<double>& basicConsLb,
    const std::vector<double>& basicConsUb,
    const std::vector<double>& additionalConsLb,
    const std::vector<double>& additionalConsUb,
    int numBasicConstraints,
    int numVariables
) {
    // Combine basic and additional triplets
    std::vector<Eigen::Triplet<int>> combinedTriplets;
    combinedTriplets.reserve(basicTriplets.size() + additionalTriplets.size());
    combinedTriplets.insert(combinedTriplets.end(), basicTriplets.begin(), basicTriplets.end());
    combinedTriplets.insert(combinedTriplets.end(), additionalTriplets.begin(), additionalTriplets.end());

    // Update the constraint matrix
    lp.ConsMatrix.resize(numBasicConstraints + additionalConsLb.size(), numVariables);
    lp.ConsMatrix.setFromTriplets(combinedTriplets.begin(), combinedTriplets.end());

    // Update the constraint bounds
    lp.consLb = basicConsLb;
    lp.consUb = basicConsUb;
    lp.consLb.insert(lp.consLb.end(), additionalConsLb.begin(), additionalConsLb.end());
    lp.consUb.insert(lp.consUb.end(), additionalConsUb.begin(), additionalConsUb.end());
}