#pragma once
#include "Lloyd.h"
#include "fair_Lloyd.h"
#include "fair_assignment_solver.h"
#include "Utils_Struct.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>

class RoundingHeuristic {
private:
    // Data for clustering
    const VectorXdList& dataPoints;
    const Eigen::MatrixXd& dis_matrix;
    
    // Solution matrix
    Eigen::MatrixXd& Xsol;
    int k; // Number of clusters
    bool is_fair_clustering = false;
    bool is_spectral_clustering = false;
        
    // Constraints (for constrained clustering)
    const std::vector<BranchConstraint>* constraints = nullptr;

    // Fair assignment solver (replaces Gurobi model)
    FairAssignmentSolver* fair_solver = nullptr;
    
    // Results
    VectorXdList final_centroids;
    std::vector<int> final_assignment;
    double final_objective;
    bool is_infeasible = false;

public:
    // Constructor for fair clustering (using FairAssignmentSolver)
    RoundingHeuristic(const VectorXdList& dataPoints, const Eigen::MatrixXd& dis_matrix, int k, Eigen::MatrixXd& Xsol,
                      FairAssignmentSolver* solver)
        : dataPoints(dataPoints), dis_matrix(dis_matrix), Xsol(Xsol), k(k), is_fair_clustering(true), 
          fair_solver(solver) {}

    // Constructor for regular clustering
    RoundingHeuristic(const VectorXdList& dataPoints, const Eigen::MatrixXd& dis_matrix, int k, 
                      Eigen::MatrixXd& Xsol)
        : dataPoints(dataPoints), dis_matrix(dis_matrix), Xsol(Xsol), k(k), is_fair_clustering(false), 
          fair_solver(nullptr) {}

    // constructor for spectral clustering, leave dataPoints empty
    RoundingHeuristic(const Eigen::MatrixXd& dis_matrix, int k, Eigen::MatrixXd& Xsol)
    : dataPoints(getEmptyDataPoints()), dis_matrix(dis_matrix), Xsol(Xsol), k(k), 
          is_fair_clustering(false), is_spectral_clustering(true), fair_solver(nullptr) {}

private:
    // Sentinel empty vector for spectral constructor — avoids binding a reference to a temporary
    static const VectorXdList& getEmptyDataPoints() {
        static const VectorXdList empty;
        return empty;
    }

public:

    // Constructor for regular clustering with constraints
    RoundingHeuristic(const VectorXdList& dataPoints, const Eigen::MatrixXd& dis_matrix, int k, 
                      Eigen::MatrixXd& Xsol, const std::vector<BranchConstraint>& constraints)
        : dataPoints(dataPoints), dis_matrix(dis_matrix), Xsol(Xsol), k(k), is_fair_clustering(false), 
          constraints(&constraints), fair_solver(nullptr) {}

    // Set solution matrix
    void setSolutionMatrix(Eigen::MatrixXd& X) { Xsol = X; }

    // Set constraints for constrained clustering
    void setConstraints(const std::vector<BranchConstraint>& constr);// { constraints = &constr; }

    // Main method to run the rounding heuristic
    bool run(int maxIterations = 10000);
    
    // Get results
    const VectorXdList& getFinalCentroids() const { return final_centroids; }
    const std::vector<int>& getFinalAssignment() const { return final_assignment; }
    Eigen::MatrixXd getFinalMatrix(){
        return createPartitionMatrix(final_assignment, k);
    }
    double getFinalObjective();// const { return final_objective; }
    bool isInfeasible() const { return is_infeasible; }

private:
    // Helper methods
    Eigen::MatrixXd computeTopKEigenvectors();
    VectorXdList generateInitialCentroids(const Eigen::MatrixXd& X_k);
    bool runFairLloydWithGurobi(const VectorXdList& initial_centroids, int maxIterations);
    bool runRegularLloyd(const VectorXdList& initial_centroids, int maxIterations);
    bool runConstrainedLloyd(const VectorXdList& initial_centroids, int maxIterations);
    bool spectralRounding();
};