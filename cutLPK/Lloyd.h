#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
#include <vector>
#include <random>


std::vector<Eigen::VectorXd> initializeCentroidsPlusPlus(const std::vector<Eigen::VectorXd>& dataPoints, int k, std::mt19937& gen);
bool assignClusters(const std::vector<Eigen::VectorXd>& dataPoints, std::vector<Eigen::VectorXd>& centroids, std::vector<int>& assignment);
void updateCentroids(const std::vector<Eigen::VectorXd>& dataPoints, std::vector<Eigen::VectorXd>& centroids, const std::vector<int>& assignment, int k);
double computeWCSS(const std::vector<Eigen::VectorXd>& dataPoints, const std::vector<Eigen::VectorXd>& centroids, const std::vector<int>& assignment);
Eigen::MatrixXd createPartitionMatrix(const std::vector<int>& assignment, int n, int k);
std::pair<double, Eigen::MatrixXd> runKMeans(const std::vector<Eigen::VectorXd>& dataPoints, int k, int maxIterations, int numTrials, int random_seed);
double KMeansObj(const Eigen::SparseMatrix<double>& Xsol, const Eigen::MatrixXd& dis_matrix, int N);