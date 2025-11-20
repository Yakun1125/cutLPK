#include "Rounding_heuristic.h"
#include <iostream>
#include <random>

bool RoundingHeuristic::run(int maxIterations) {
    try {
        // Reset infeasibility flag
        is_infeasible = false;
        
        //std::cout << "Starting rounding heuristic..." << std::endl;
        bool success;
        if (is_fair_clustering) {
            Eigen::MatrixXd topk_eigenvectors = computeTopKEigenvectors();
            VectorXdList initial_centroids = generateInitialCentroids(topk_eigenvectors);
            success = runFairLloydWithGurobi(initial_centroids, maxIterations);
            // clear fairness constraints from gurobi model
            if (gurobi_model && constraints) {
                int num_constrs = gurobi_model->get(GRB_IntAttr_NumConstrs);
                GRBConstr* constrs = gurobi_model->getConstrs();
                for (int i = 0; i < num_constrs; ++i) {
                    std::string name = constrs[i].get(GRB_StringAttr_ConstrName);
                    if (name.find("same_cluster_") == 0 || name.find("diff_cluster_") == 0) {
                        gurobi_model->remove(constrs[i]);
                    }
                }
                gurobi_model->update();
                delete[] constrs; // Don't forget to free memory!
            }
        } 
        else if (is_spectral_clustering){
            success = spectralRounding();
        }
        else {
            Eigen::MatrixXd topk_eigenvectors = computeTopKEigenvectors();
            VectorXdList initial_centroids = generateInitialCentroids(topk_eigenvectors);
            if (constraints != nullptr && !constraints->empty()) {
                success = runConstrainedLloyd(initial_centroids, maxIterations);
            } else {
                success = runRegularLloyd(initial_centroids, maxIterations);
            }
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

void RoundingHeuristic::setConstraints(const std::vector<BranchConstraint>& constr) {
    constraints = &constr; 
    // if it is fair clustering, we add constraints to the gurobi model
    if (is_fair_clustering && gurobi_model && x_vars) {
        for (const auto& bc : *constraints) {
            if (bc.type == BranchType::SAME_CLUSTER) {
                // X(i,j) = X(i,i) = X(j,j)
                for (int c = 0; c < k; ++c) {
                    gurobi_model->addConstr((*x_vars)[bc.i][c] - (*x_vars)[bc.j][c] == 0, 
                                            "same_cluster_" + std::to_string(bc.i) + "_" + std::to_string(bc.j) + "_c" + std::to_string(c));
                }
            } else if (bc.type == BranchType::DIFF_CLUSTER) {
                // X(i,j) = 0
                for (int c = 0; c < k; ++c) {
                    gurobi_model->addConstr((*x_vars)[bc.i][c] + (*x_vars)[bc.j][c] <= 1, 
                                            "diff_cluster_" + std::to_string(bc.i) + "_" + std::to_string(bc.j) + "_c" + std::to_string(c));
                }
            }
        }
        gurobi_model->update();
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

VectorXdList RoundingHeuristic::generateInitialCentroids(const Eigen::MatrixXd& X_k) {
    int N = dataPoints.size();
    int d = dataPoints[0].size(); // Data dimensionality

    // Convert aligned dataPoints to an Eigen::MatrixXd (N x d)
    Eigen::MatrixXd D_matrix(N, d);
    for (int i = 0; i < N; ++i) {
        if (dataPoints[i].size() != d) {
            throw std::runtime_error("Inconsistent data point dimensions.");
        }
        D_matrix.row(i) = dataPoints[i];
    }

    Eigen::MatrixXd centroids_matrix = X_k * D_matrix;

    VectorXdList transformed_data_points(N);
    for (int i = 0; i < N; ++i) {
        transformed_data_points[i] = centroids_matrix.row(i);
    }

    std::vector<int> assignment(dataPoints.size(), -1);
    double clustering_cost;
    std::tie(clustering_cost, assignment) = runKMeans(transformed_data_points, k, 10000, 42);

    // get clustering results based on assignment
    VectorXdList initial_centroids(k, Eigen::VectorXd::Zero(d));
    updateCentroids(dataPoints, initial_centroids, assignment, k);
    
    return initial_centroids;
}

bool RoundingHeuristic::runFairLloydWithGurobi(const VectorXdList& initial_centroids, int maxIterations) {  
    if (!gurobi_model || !x_vars) {
        std::cerr << "Error: Gurobi model or variables not provided for fair clustering" << std::endl;
        return false;
    }
    
    try {
        // Set initial centroids
    VectorXdList current_centroids = initial_centroids;
        std::vector<int> current_assignment(dataPoints.size(), -1);
        
        int N = dataPoints.size();
        bool changed;
        
        // Initial assignment using fair clustering
        FairAssignStatus initial_status = GRB_fairAssignClusters(dataPoints, current_centroids, current_assignment, 
                                        *gurobi_model, *x_vars);
        
        if (initial_status == FairAssignStatus::INFEASIBLE || initial_status == FairAssignStatus::UNBOUNDED) {
            std::cerr << "Initial fair assignment is infeasible. Cannot proceed with fair Lloyd." << std::endl;
            is_infeasible = true;
            return false;
        }
        
        changed = (initial_status == FairAssignStatus::SUCCESS);
        
        double currentWCSS =  computeWCSS(dataPoints, current_centroids, current_assignment);
        //std::cout << "Initial fair assignment objective: " << currentWCSS << std::endl;
        
        for (int iter = 0; iter < maxIterations && changed; ++iter) {
            //std::cout << "Fair Lloyd iteration " << iter + 1 << std::endl;
            
            if (!changed) {
                //std::cout << "No assignment changes, converged." << std::endl;
                break;
            }
            
            // Store old centroids for convergence check
            VectorXdList oldCentroids = current_centroids;
            
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
            FairAssignStatus assign_status = GRB_fairAssignClusters(dataPoints, current_centroids, current_assignment, 
                                           *gurobi_model, *x_vars);
            
            if (assign_status == FairAssignStatus::INFEASIBLE || assign_status == FairAssignStatus::UNBOUNDED) {
                std::cerr << "Fair assignment became infeasible at iteration " << iter << ". Stopping early." << std::endl;
                is_infeasible = true;
                break;
            }
            
            changed = (assign_status == FairAssignStatus::SUCCESS);
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

bool RoundingHeuristic::runRegularLloyd(const VectorXdList& initial_centroids, int maxIterations) {
    try {
    VectorXdList current_centroids = initial_centroids;
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
            
            VectorXdList oldCentroids = current_centroids;
            
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
    VectorXdList stacked_eigvectors(Xsol.rows());
    for (int i = 0; i < Xsol.rows(); ++i) {
        stacked_eigvectors[i] = eigenvectors.row(i).transpose();
    }
    
    // Check if we have constraints to apply
    if (constraints != nullptr && !constraints->empty()) {
        // Run constrained Lloyd's algorithm on the stacked eigenvectors
        // std::cout << "Running constrained Lloyd's algorithm on stacked eigenvectors" << std::endl;
        
        double bestClusteringCost = kInfinity;
        std::vector<int> bestAssignment;
        
        // Try multiple random initializations with constrained Lloyd
        for (int i = 0; i < 100; i++) {
            // Randomly pick initial centroids using K-means++ initialization
            VectorXdList initial_centroids = initializeCentroidsPlusPlus(stacked_eigvectors, k, i + 42);
            
            // Get initial assignment based on these centroids
            std::vector<int> constrained_assignment(stacked_eigvectors.size(), -1);
            assignClusters(stacked_eigvectors, initial_centroids, constrained_assignment);
            
            // Now run constrained Lloyd iteratively until convergence
            bool changed = true;
            int iteration = 0;
            
            while (changed && iteration < 100000) {
                VectorXdList old_centroids = initial_centroids;
                
                // Update centroids
                updateCentroids(stacked_eigvectors, initial_centroids, constrained_assignment, k);
                
                // Check centroid convergence
                double centroid_shift_sq = 0.0;
                double centroid_norm_sq = 0.0;
                for (size_t c = 0; c < initial_centroids.size(); ++c) {
                    centroid_shift_sq += (initial_centroids[c] - old_centroids[c]).squaredNorm();
                    centroid_norm_sq += initial_centroids[c].squaredNorm();
                }
                
                if (centroid_norm_sq > 0 && centroid_shift_sq <= 1e-6 * centroid_norm_sq) {
                    break;  // Centroids converged
                }
                
                // Reassign clusters with constraints
                changed = ConstrainedAssignClusters(stacked_eigvectors, initial_centroids, constrained_assignment, *constraints);
                iteration++;
            }
            
            double constrained_cost = computeWCSS(stacked_eigvectors, initial_centroids, constrained_assignment);
            
            if (bestClusteringCost > constrained_cost) {
                bestClusteringCost = constrained_cost;
                bestAssignment = std::move(constrained_assignment);
            }
        }
        
        final_assignment = std::move(bestAssignment);
    } else {
        // Run regular Lloyd's algorithm on the stacked eigenvectors (no constraints)
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
    }

    return true;
}

bool RoundingHeuristic::runConstrainedLloyd(const VectorXdList& initial_centroids, int maxIterations) {
    try {
        if (constraints == nullptr) {
            throw std::runtime_error("Constraints not set for constrained Lloyd");
        }
        
    VectorXdList current_centroids = initial_centroids;
        std::vector<int> current_assignment(dataPoints.size(), -1);

        int N = dataPoints.size();
        bool changed = true;
        
        // Initial assignment with constraints
        changed = ConstrainedAssignClusters(dataPoints, current_centroids, current_assignment, *constraints);
        double currentWCSS = computeWCSS(dataPoints, current_centroids, current_assignment);
        //std::cout << "Initial constrained assignment objective: " << currentWCSS << std::endl;
        
        for (int iter = 0; iter < maxIterations && changed; ++iter) {
            // std::cout << "Constrained Lloyd iteration " << iter + 1 << std::endl;
            
            if (!changed) {
                std::cout << "No assignment changes, converged." << std::endl;
                break;
            }
            
            VectorXdList oldCentroids = current_centroids;
            
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
            
            // Reassign clusters with constraints
            changed = ConstrainedAssignClusters(dataPoints, current_centroids, current_assignment, *constraints);
        }
        
        // Final centroid update
        updateCentroids(dataPoints, current_centroids, current_assignment, k);
        currentWCSS = computeWCSS(dataPoints, current_centroids, current_assignment);
        //std::cout << "Final constrained assignment objective: " << currentWCSS << std::endl;
        
        final_centroids = current_centroids;
        final_assignment = current_assignment;
        
        //std::cout << "Constrained Lloyd completed successfully" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error in constrained Lloyd: " << e.what() << std::endl;
        return false;
    }
}