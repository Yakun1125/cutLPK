#include "fairLPK_Utils.h"

void constructFairLPK(LPK& lp, Eigen::MatrixXd& dis_matrix, int N, int K, std::vector<Eigen::Triplet<int>>& basic_triplets, std::vector<std::vector<bool>>& dataGroups, std::vector<int>& groupRatio, const parameters& params) {
	lp.N = N;
	int numVars = N * (N + 1) / 2;
	int numGroups = groupRatio.size();
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
	lp.consLb = std::vector<double>(1 + N + (numGroups * N), 1);//
	lp.consUb = std::vector<double>(1 + N + (numGroups * N), 1);
	lp.consLb[0] = K;
	lp.consUb[0] = K;

	std::vector<double> normalized_groupRatio(numGroups, 0.0);
	for (int g = 0; g < numGroups; g++) {
		normalized_groupRatio[g] = double(groupRatio[g]) / double(N);
	}
	// print the normalized group ratio
	int baseIndex = 1 + N;
	for (int g = 0; g < numGroups; g++) {
		for (int i = 0; i < N; i++) {
			int index = baseIndex + g * N + i;
			lp.consLb[index] = normalized_groupRatio[g] - params.fairness_param;
			lp.consUb[index] = normalized_groupRatio[g] + params.fairness_param;
		}
	}

	// Reserve space for triplets based on an estimate of non-zero elements
	int estimated_nonzeros = (N + 1) * N;
	for (int g = 0; g < numGroups; g++) {
		estimated_nonzeros += groupRatio[g] * N;
	}
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
	for (int g = 0; g < numGroups; g++) {
		for (int j = 0; j < N; j++) {
			for (int i = 0; i < N; ++i) {
				int col = std::min(i, j) * (2 * N - std::min(i, j) + 1) / 2 + std::max(i, j) - std::min(i, j);
				if (dataGroups[i][g]) {
					int index = baseIndex + g * N + j;
					basic_triplets.emplace_back(index, col, 1);
				}
			}
		}
	}
}
