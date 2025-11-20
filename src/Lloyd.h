#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
#include <vector>
#include <random>
#include "Utils_Struct.h"

// Randomly Initialize centroids using kmeans++
VectorXdList initializeCentroidsPlusPlus(const VectorXdList& dataPoints, int k, int random_seed);
// Assign datapoints to closest center and check if assignment changes. If so, return true.
bool assignClusters(const VectorXdList& dataPoints, VectorXdList& centroids, std::vector<int>& assignment);
// Constrained version that respects must-link and cannot-link constraints
bool ConstrainedAssignClusters(const VectorXdList& dataPoints, VectorXdList& centroids, std::vector<int>& assignment, const std::vector<BranchConstraint>& constraints);
// Update centroids if assignment changes.
void updateCentroids(const VectorXdList& dataPoints, VectorXdList& centroids, const std::vector<int>& assignment, int k);
// Compute within cluster distance
double computeWCSS(const VectorXdList& dataPoints, const VectorXdList& centroids, const std::vector<int>& assignment);
// Main function for running Kmeans
std::pair<double, std::vector<int>> runKMeans(const VectorXdList& dataPoints, int k, int maxIterations, int random_seed);
// Compute obj of partition matrix 
double KMeansObjPartitionMatrix(const Eigen::MatrixXd& Xsol, const VectorXdList& dataPoints, int k);
// Compute obj using assignment
double KMeansObjAssignment(const std::vector<int>& assignment, const VectorXdList& dataPoints, int k);
// Convert assignment to partition matrix
Eigen::MatrixXd createPartitionMatrix(const std::vector<int>& assignment, int k);
// Convert partition matrix to assignment
std::vector<int> createAssignment(const Eigen::MatrixXd& Xsol, int k);