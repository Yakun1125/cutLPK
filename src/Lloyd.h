#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
#include <vector>
#include <random>

// Randomly Initialize centroids using kmeans++
std::vector<Eigen::VectorXd> initializeCentroidsPlusPlus(const std::vector<Eigen::VectorXd>& dataPoints, int k, int random_seed);
// Assign datapoints to closest center and check if assignment changes. If so, return true.
bool assignClusters(const std::vector<Eigen::VectorXd>& dataPoints, std::vector<Eigen::VectorXd>& centroids, std::vector<int>& assignment);
// Update centroids if assignment changes.
void updateCentroids(const std::vector<Eigen::VectorXd>& dataPoints, std::vector<Eigen::VectorXd>& centroids, const std::vector<int>& assignment, int k);
// Compute within cluster distance
double computeWCSS(const std::vector<Eigen::VectorXd>& dataPoints, const std::vector<Eigen::VectorXd>& centroids, const std::vector<int>& assignment);
// Main function for running Kmeans
std::pair<double, std::vector<int>> runKMeans(const std::vector<Eigen::VectorXd>& dataPoints, int k, int maxIterations, int random_seed);
// Compute obj of partition matrix 
double KMeansObjPartitionMatrix(const Eigen::MatrixXd& Xsol, const std::vector<Eigen::VectorXd>& dataPoints, int k);
// Compute obj using assignment
double KMeansObjAssignment(const std::vector<int>& assignment, const std::vector<Eigen::VectorXd>& dataPoints, int k);
// Convert assignment to partition matrix
Eigen::MatrixXd createPartitionMatrix(const std::vector<int>& assignment, int k);
// Convert partition matrix to assignment
std::vector<int> createAssignment(const Eigen::MatrixXd& Xsol, int k);