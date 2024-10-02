#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
#include <vector>
#include <random>
#include "gurobi_c++.h"


std::vector<Eigen::VectorXd> initializeCentroidsPlusPlus(const std::vector<Eigen::VectorXd>& dataPoints, int k, std::mt19937& gen);
bool assignClusters(const std::vector<Eigen::VectorXd>& dataPoints, std::vector<Eigen::VectorXd>& centroids, std::vector<int>& assignment);
bool fairAssignment(GRBModel& model, std::vector<std::vector<GRBVar>>& x, const std::vector<Eigen::VectorXd>& dataPoints, std::vector<Eigen::VectorXd>& centroids, std::vector<int>& assignment, const std::vector<std::vector<bool>>& dataGroups, const std::vector<int>& groupRatio, double fairness_param);

void updateCentroids(const std::vector<Eigen::VectorXd>& dataPoints, std::vector<Eigen::VectorXd>& centroids, const std::vector<int>& assignment, int k);
double computeWCSS(const std::vector<Eigen::VectorXd>& dataPoints, const std::vector<Eigen::VectorXd>& centroids, const std::vector<int>& assignment);
Eigen::MatrixXd createPartitionMatrix(const std::vector<int>& assignment, int n, int k);
std::pair<double, Eigen::MatrixXd> runKMeans(const std::vector<Eigen::VectorXd>& dataPoints, int k, int maxIterations, int numTrials, int random_seed);
std::pair<double, Eigen::MatrixXd> runFairKMeans(const std::vector<Eigen::VectorXd>& dataPoints, int k, int maxIterations, int numTrials, int random_seed, const std::vector<std::vector<bool>>& dataGroups, const std::vector<int>& groupRatio, double fairness_param);
double KMeansObj(const Eigen::SparseMatrix<double>& Xsol, const Eigen::MatrixXd& dis_matrix, int N);
std::vector<std::vector<double>> computeRatio(const Eigen::MatrixXd& Xsol, const std::vector<std::vector<bool>>& dataGroups);
