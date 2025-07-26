#include "Rounding_heuristic.h"
#include <iostream>
#include <random>

bool RoundingHeuristic::run(int maxIterations) {
    try {
        //std::cout << "Starting rounding heuristic..." << std::endl;
        bool success;
        if (is_fair_clustering) {
            Eigen::MatrixXd topk_eigenvectors = computeTopKEigenvectors();
            std::vector<Eigen::VectorXd> initial_centroids = generateInitialCentroids(topk_eigenvectors);
            success = runFairLloydWithGurobi(initial_centroids, maxIterations);
        } 
        else if (is_spectral_clustering){
            success = spectralRounding();
        }
        else {
            Eigen::MatrixXd topk_eigenvectors = computeTopKEigenvectors();
            std::vector<Eigen::VectorXd> initial_centroids = generateInitialCentroids(topk_eigenvectors);
            success = runRegularLloyd(initial_centroids, maxIterations);
        }
        
        // if (success) {
        //     final_objective = computeWCSS(dataPoints, final_centroids, final_assignment);
        //     std::cout << "Rounding heuristic completed successfully. Final objective: " 
        //              << final_objective << std::endl;
        // } else {
        //     std::cout << "Rounding heuristic failed during Lloyd's algorithm" << std::endl;
        // }
        
        return success;
        
    } catch (const std::exception& e) {
        std::cerr << "Error in rounding heuristic: " << e.what() << std::endl;
        return false;
    }
}

double RoundingHeuristic::getFinalObjective() {
    Eigen::MatrixXd final_matrix = createPartitionMatrix(final_assignment, k);
    double objective = 0.0;
    if (is_spectral_clustering){
        objective = dis_matrix.cwiseProduct(final_matrix).sum();
    }
    else{
        for (int i = 0; i < final_assignment.size(); ++i)
        {
            for (int j = i; j < final_assignment.size(); ++j)
            {
                objective += dis_matrix(i, j) * final_matrix(i, j);
            }
        }
    }
    return objective;
}

Eigen::MatrixXd RoundingHeuristic::computeTopKEigenvectors() {

    
    // Eigenvalue decomposition using dense solver for better numerical stability
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(Xsol);
    
    if (eigensolver.info() != Eigen::Success) {
        throw std::runtime_error("Eigenvalue decomposition failed!");
    }
    
    // Get eigenvalues and eigenvectors (sorted in ascending order)
    Eigen::VectorXd eigenvalues = eigensolver.eigenvalues();
    Eigen::MatrixXd eigenvectors = eigensolver.eigenvectors();
    
    // std::cout << "Eigenvalue range: [" << eigenvalues.minCoeff() 
    //           << ", " << eigenvalues.maxCoeff() << "]" << std::endl;
    
    // Get largest k eigenvectors (rightmost columns)
    Eigen::MatrixXd topk_eigenvectors = eigenvectors.rightCols(k);
    
    // std::cout << "Selected top " << k << " eigenvalues: ";
    // for (int i = 0; i < k; ++i) {
    //     std::cout << eigenvalues(eigenvalues.size() - k + i) << " ";
    // }
    // std::cout << std::endl;

    // The best rank-k approximation of Xsol is X_k = V_k * D_k * V_k^T
    // where V_k are the top k eigenvectors and D_k are the top k eigenvalues.
    Eigen::VectorXd topk_eigenvalues = eigenvalues.tail(k);
    Eigen::MatrixXd X_k = topk_eigenvectors * topk_eigenvalues.asDiagonal() * topk_eigenvectors.transpose();

    
    return X_k;
}

std::vector<Eigen::VectorXd> RoundingHeuristic::generateInitialCentroids(const Eigen::MatrixXd& X_k) {
    int N = dataPoints.size();
    int d = dataPoints[0].size(); // Data dimensionality

    // Convert std::vector<Eigen::VectorXd> dataPoints to an Eigen::MatrixXd (N x d)
    Eigen::MatrixXd D_matrix(N, d);
    for (int i = 0; i < N; ++i) {
        if (dataPoints[i].size() != d) {
            throw std::runtime_error("Inconsistent data point dimensions.");
        }
        D_matrix.row(i) = dataPoints[i];
    }

    Eigen::MatrixXd centroids_matrix = X_k * D_matrix;

    std::vector<Eigen::VectorXd> transformed_data_points(N);
    for (int i = 0; i < N; ++i) {
        transformed_data_points[i] = centroids_matrix.row(i);
    }

    std::vector<int> assignment(dataPoints.size(), -1);
    double clustering_cost;
    std::tie(clustering_cost, assignment) = runKMeans(transformed_data_points, k, 10000, 42);

    // get clustering results based on assignment
    std::vector<Eigen::VectorXd> initial_centroids(k, Eigen::VectorXd::Zero(d));
    updateCentroids(dataPoints, initial_centroids, assignment, k);
    
    return initial_centroids;
}

bool RoundingHeuristic::runFairLloydWithGurobi(const std::vector<Eigen::VectorXd>& initial_centroids, int maxIterations) {  
    if (!gurobi_model || !x_vars) {
        std::cerr << "Error: Gurobi model or variables not provided for fair clustering" << std::endl;
        return false;
    }
    
    try {
        // Set initial centroids
        std::vector<Eigen::VectorXd> current_centroids = initial_centroids;
        std::vector<int> current_assignment(dataPoints.size(), -1);
        
        int N = dataPoints.size();
        bool changed = true;
        
        // Initial assignment using fair clustering
        changed = GRB_fairAssignClusters(dataPoints, current_centroids, current_assignment, 
                                        *gurobi_model, *x_vars);
        
        double currentWCSS =  computeWCSS(dataPoints, current_centroids, current_assignment);
        //std::cout << "Initial fair assignment objective: " << currentWCSS << std::endl;
        
        for (int iter = 0; iter < maxIterations && changed; ++iter) {
            //std::cout << "Fair Lloyd iteration " << iter + 1 << std::endl;
            
            if (!changed) {
                //std::cout << "No assignment changes, converged." << std::endl;
                break;
            }
            
            // Store old centroids for convergence check
            std::vector<Eigen::VectorXd> oldCentroids = current_centroids;
            
            // Update centroids based on current assignment
            updateCentroids(dataPoints, current_centroids, current_assignment, k);
            
            // Compute new objective
            currentWCSS = computeWCSS(dataPoints, current_centroids, current_assignment);
            //std::cout << "Updated centroids, objective: " << currentWCSS << std::endl;
            
            // Check centroid convergence
            double centroidShiftSquared = 0.0;
            double centroidNormSquared = 0.0;
            
            for (size_t i = 0; i < current_centroids.size(); ++i) {
                centroidShiftSquared += (current_centroids[i] - oldCentroids[i]).squaredNorm();
                centroidNormSquared += current_centroids[i].squaredNorm();
            }
            
            // Check relative centroid shift
            if (centroidNormSquared > 0 && 
                centroidShiftSquared <= 1e-6 * centroidNormSquared) {
                //std::cout << "Centroids converged." << std::endl;
                break;
            }
            
            // Do fair assignment with new centroids
            changed = GRB_fairAssignClusters(dataPoints, current_centroids, current_assignment, 
                                           *gurobi_model, *x_vars);
        }
        
        // Store final results
        updateCentroids(dataPoints, current_centroids, current_assignment, k);
        currentWCSS = computeWCSS(dataPoints, current_centroids, current_assignment);
        //std::cout << "Final fair assignment objective: " << currentWCSS << std::endl;
        final_centroids = current_centroids;
        final_assignment = current_assignment;
        
        //std::cout << "Fair Lloyd completed successfully" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error in fair Lloyd with Gurobi: " << e.what() << std::endl;
        return false;
    }
}

bool RoundingHeuristic::runRegularLloyd(const std::vector<Eigen::VectorXd>& initial_centroids, int maxIterations) {
    try {
        std::vector<Eigen::VectorXd> current_centroids = initial_centroids;
        std::vector<int> current_assignment(dataPoints.size(), -1);

        int N = dataPoints.size();
        bool changed = true;
        
        // Initial assignment
        changed = assignClusters(dataPoints, current_centroids, current_assignment);
        double currentWCSS = computeWCSS(dataPoints, current_centroids, current_assignment);
        //std::cout << "Initial assignment objective: " << currentWCSS << std::endl;
        
        for (int iter = 0; iter < maxIterations && changed; ++iter) {
            // std::cout << "Lloyd iteration " << iter + 1 << std::endl;
            
            if (!changed) {
                std::cout << "No assignment changes, converged." << std::endl;
                break;
            }
            
            std::vector<Eigen::VectorXd> oldCentroids = current_centroids;
            
            // Update centroids
            updateCentroids(dataPoints, current_centroids, current_assignment, k);
            currentWCSS = computeWCSS(dataPoints, current_centroids, current_assignment);
            
            // Check convergence
            double centroidShiftSquared = 0.0;
            double centroidNormSquared = 0.0;
            
            for (size_t i = 0; i < current_centroids.size(); ++i) {
                centroidShiftSquared += (current_centroids[i] - oldCentroids[i]).squaredNorm();
                centroidNormSquared += current_centroids[i].squaredNorm();
            }

            if (centroidNormSquared > 0 && 
                centroidShiftSquared <= 1e-6 * centroidNormSquared) {
                // std::cout << "Centroids converged." << std::endl;
                break;
            }
            
            // Reassign clusters
            changed = assignClusters(dataPoints, current_centroids, current_assignment);
        }
        updateCentroids(dataPoints, current_centroids, current_assignment, k);
        currentWCSS = computeWCSS(dataPoints, current_centroids, current_assignment);
        //std::cout << "Final assignment objective: " << currentWCSS << std::endl;
        final_centroids = current_centroids;
        final_assignment = current_assignment;
        
        //std::cout << "Regular Lloyd completed successfully" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error in regular Lloyd: " << e.what() << std::endl;
        return false;
    }
}

bool RoundingHeuristic::spectralRounding(){
    // do eigen decomposition to Xsol and get top k eigenvectors
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(Xsol);
    if (eigensolver.info() != Eigen::Success) {
        std::cout << "Eigenvalue decomposition failed!"<< std::endl;
        return false;
    }
    // std::cout << "Eigenvalue range: [" << eigensolver.eigenvalues().minCoeff() 
    //           << ", " << eigensolver.eigenvalues().maxCoeff() << "]" << std::endl;

    // get largest k eigenvectors
    Eigen::MatrixXd eigenvectors = eigensolver.eigenvectors().rightCols(k);
    Eigen::VectorXd eigenvalues = eigensolver.eigenvalues().tail(k);
    // stack them
    std::vector<Eigen::VectorXd> stacked_eigvectors(Xsol.rows());
    for (int i = 0; i < Xsol.rows(); ++i) {
        stacked_eigvectors[i] = eigenvectors.row(i).transpose();
    }
    // run Lloyd's algorithm on the stacked eigenvectors
    double bestClusteringCost = kInfinity;
    for (int i = 0; i < 100; i++) {
        double ClusteringCost;
        std::vector<int> lloydAssignment;
        std::tie(ClusteringCost, lloydAssignment) = runKMeans(stacked_eigvectors, k, 100000, i+3);
        
        if (bestClusteringCost > ClusteringCost) {
            bestClusteringCost = ClusteringCost;
            final_assignment = std::move(lloydAssignment);
        }
    }

    return true;
}