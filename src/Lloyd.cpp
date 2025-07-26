#include "Lloyd.h"
#include <iostream>

std::vector<Eigen::VectorXd> initializeCentroidsPlusPlus(const std::vector<Eigen::VectorXd>& dataPoints, int k, int random_seed) {
	std::vector<Eigen::VectorXd> centroids;
	std::uniform_int_distribution<> dis(0, dataPoints.size() - 1);
    std::mt19937 gen(random_seed);
	centroids.push_back(dataPoints[dis(gen)]);

	for (int i = 1; i < k; ++i) {
		std::vector<double> distances(dataPoints.size(), std::numeric_limits<double>::max());

		for (size_t j = 0; j < dataPoints.size(); ++j) {
			for (size_t c = 0; c < centroids.size(); ++c) {
				double dist = (dataPoints[j] - centroids[c]).squaredNorm();
				distances[j] = std::min(distances[j], dist);
			}
		}

		std::discrete_distribution<> weightedDist(distances.begin(), distances.end());
		centroids.push_back(dataPoints[weightedDist(gen)]);
	}

	return centroids;
}

bool assignClusters(const std::vector<Eigen::VectorXd>& dataPoints, std::vector<Eigen::VectorXd>& centroids, std::vector<int>& assignment) {
	bool changed = false;
	for (size_t i = 0; i < dataPoints.size(); ++i) {
		double minDist = std::numeric_limits<double>::max();
		int closestCentroid = -1;
		for (int j = 0; j < centroids.size(); ++j) {
			double dist = (dataPoints[i]-centroids[j]).squaredNorm();
			if (dist < minDist) {
				minDist = dist;
				closestCentroid = j;
			}
		}
		if (assignment[i] != closestCentroid) {
			assignment[i] = closestCentroid;
			changed = true;
		}
	}
	return changed;
}

void updateCentroids(const std::vector<Eigen::VectorXd>& dataPoints, std::vector<Eigen::VectorXd>& centroids, const std::vector<int>& assignment, int k) {
	std::vector<int> clusterSizes(k, 0);
	std::vector<Eigen::VectorXd> newCentroids(k, Eigen::VectorXd::Zero(centroids[0].size()));

	for (size_t i = 0; i < dataPoints.size(); ++i) {
		newCentroids[assignment[i]] += dataPoints[i];
		clusterSizes[assignment[i]]++;
	}

	for (int j = 0; j < k; ++j) {
		if (clusterSizes[j] > 0) {
			centroids[j] = newCentroids[j] / clusterSizes[j];
		}
	}
}

double computeWCSS(const std::vector<Eigen::VectorXd>& dataPoints, const std::vector<Eigen::VectorXd>& centroids, const std::vector<int>& assignment) {
	double totalWCSS = 0.0;
	for (size_t i = 0; i < dataPoints.size(); ++i) {
		totalWCSS += (dataPoints[i]-centroids[assignment[i]]).squaredNorm();
	}
	return totalWCSS;
}

Eigen::MatrixXd createPartitionMatrix(const std::vector<int>& assignment, int k) {
	int n = assignment.size();
    Eigen::MatrixXd partitionMatrix = Eigen::MatrixXd::Zero(n, n);
	std::vector<int> clusterSizes(k, 0);

	for (int i = 0; i < n; ++i) {
		clusterSizes[assignment[i]]++;
	}

	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			if (assignment[i] == assignment[j]) {
				partitionMatrix(i, j) = 1.0 / clusterSizes[assignment[i]];
			}
		}
	}

	return partitionMatrix;
}

std::vector<int> createAssignment(const Eigen::MatrixXd& Xsol, int k){
    int n = Xsol.rows();
    std::vector<int> assignment(n, -1);
    int clusterCount = 0;
    
    for (int i = 0; i < n; ++i) {
        if (assignment[i] != -1) continue;

        assignment[i] = clusterCount;
        
        // All nodes with non-zero values in row i belong to the same cluster
        for (int j = 0; j < n; ++j) {
            if (Xsol(i, j) > 0) {
                assignment[j] = clusterCount;
            }
        }
        
        clusterCount++;
    }
    
    if (clusterCount != k) {
        std::cerr << "Warning: Found " << clusterCount 
                  << " clusters, but expected " << k << std::endl;
    }
    
    return assignment;
}

double KMeansObjPartitionMatrix(const Eigen::MatrixXd& Xsol, const std::vector<Eigen::VectorXd>& dataPoints, int k){
    std::vector<int> assignment = createAssignment(Xsol, k);

    return KMeansObjAssignment(assignment, dataPoints, k);
}

double KMeansObjAssignment(const std::vector<int>& assignment, const std::vector<Eigen::VectorXd>& dataPoints, int k){
    int n = dataPoints.size();
    int dim = dataPoints[0].size();
    
    std::vector<Eigen::VectorXd> centroids(k, Eigen::VectorXd::Zero(dim));
    std::vector<int> counts(k, 0);
    
    for (size_t i = 0; i < n; ++i) {
        int clusterId = assignment[i];
        centroids[clusterId] += dataPoints[i];
        counts[clusterId]++;
    }
    
    double objective = 0.0;
    
    for (int i = 0; i < k; ++i) {
        if (counts[i] > 0) {
            centroids[i] /= counts[i];
        }
    }
    
    for (size_t i = 0; i < n; ++i) {
        int clusterId = assignment[i];
        objective += (dataPoints[i] - centroids[clusterId]).squaredNorm();
    }
    
    return objective;
}

std::pair<double, std::vector<int>> runKMeans(const std::vector<Eigen::VectorXd>& dataPoints, int k, int maxIterations, int random_seed){
    int n = dataPoints.size();

    // initialized centroids and do assignment
    std::vector<Eigen::VectorXd> centroids = initializeCentroidsPlusPlus(dataPoints, k, random_seed);
    std::vector<int> assignment(n, -1);
    bool changed = assignClusters(dataPoints, centroids, assignment);
    double currentWCSS = computeWCSS(dataPoints, centroids, assignment);

    for (int iter = 0; iter < maxIterations; ++iter) {   
        if (!changed) {
            break;
        }
        std::vector<Eigen::VectorXd> oldCentroids = centroids;
        updateCentroids(dataPoints, centroids, assignment, k);
        currentWCSS = computeWCSS(dataPoints, centroids, assignment);
        // Compute centroid shift
        double centroidShiftSquared = 0.0;
        double centroidNormSquared = 0.0;

        for (size_t i = 0; i < centroids.size(); ++i) {
            centroidShiftSquared += (centroids[i] - oldCentroids[i]).squaredNorm();
            centroidNormSquared += centroids[i].squaredNorm();
        }

        // Check relative centroid shift
        if ((centroidNormSquared > 0 && 
                            centroidShiftSquared <= 1e-6 * centroidNormSquared)) {
            break;
        }
        // if new centroids shifted, do assignment
        changed = assignClusters(dataPoints, centroids, assignment);
    }

    //Eigen::MatrixXd PartitionMatrix = createPartitionMatrix(assignment, k);

    return std::make_pair(currentWCSS, assignment);
}