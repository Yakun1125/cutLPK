#include "spectral_heuristic.h"
#include "Lloyd.h"
#include <iostream>
#include "Utils_Struct.h"

int spectralHeuristic(const Eigen::MatrixXd& L, int k, Eigen::MatrixXd& Spectral_Xsol) {
    int N = L.rows();
    // first do eigen decomposition to L
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(L);
    if (eigensolver.info() != Eigen::Success) {
        std::cout << "Eigenvalue decomposition failed!"<< std::endl;
        return -1;
    }
    // Get smallest k eigenvectors
    Eigen::MatrixXd eigenvectors = eigensolver.eigenvectors().leftCols(k);
    Eigen::VectorXd eigenvalues = eigensolver.eigenvalues().head(k);

    // Prepare data: rows of topk_eigenvectors
    std::vector<Eigen::VectorXd> stacked_eigvectors(N);
    for (int i = 0; i < N; ++i) {
        stacked_eigvectors[i] = eigenvectors.row(i).transpose();
    }

    std::vector<int> bestlloydAssignment;
    double bestClusteringCost = kInfinity;
    for (int i = 0; i < 100; i++) {
        double ClusteringCost;
        std::vector<int> lloydAssignment;
        std::tie(ClusteringCost, lloydAssignment) = runKMeans(stacked_eigvectors, k, 100000, i+3);
        
        if (bestClusteringCost > ClusteringCost) {
            bestClusteringCost = ClusteringCost;
            bestlloydAssignment = std::move(lloydAssignment);
        }
    }
    
    Spectral_Xsol = createPartitionMatrix(bestlloydAssignment, k);

    return 0;
}
