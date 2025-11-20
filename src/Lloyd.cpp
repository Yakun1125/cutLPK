#include "Lloyd.h"
#include <iostream>
#include <queue>
#include <algorithm>

VectorXdList initializeCentroidsPlusPlus(const VectorXdList& dataPoints, int k, int random_seed) {
	VectorXdList centroids;
	// print validate size of datapoints
	std::uniform_int_distribution<> dis(0, dataPoints.size() - 1);
    std::mt19937 gen(random_seed);
	centroids.push_back(dataPoints[dis(gen)]);

	for (int i = 1; i < k; ++i) {
		std::vector<double> distances(dataPoints.size(), std::numeric_limits<double>::max());

		for (int j = 0; j < dataPoints.size(); ++j) {
			for (int c = 0; c < centroids.size(); ++c) {
				double dist = (dataPoints[j] - centroids[c]).squaredNorm();
				distances[j] = std::min(distances[j], dist);
			}
		}

		std::discrete_distribution<> weightedDist(distances.begin(), distances.end());
		centroids.push_back(dataPoints[weightedDist(gen)]);
	}

	return centroids;
}

bool assignClusters(const VectorXdList& dataPoints, VectorXdList& centroids, std::vector<int>& assignment) {
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

bool ConstrainedAssignClusters(const VectorXdList& dataPoints, VectorXdList& centroids, std::vector<int>& assignment, const std::vector<BranchConstraint>& constraints) {
	bool changed = false;
	int n = dataPoints.size();
	int k = centroids.size();
	
	// Build constraint adjacency lists for efficient lookup
	std::vector<std::vector<int>> must_link(n), cannot_link(n);
	for (const auto& constraint : constraints) {
		if (constraint.type == BranchType::SAME_CLUSTER) {
			must_link[constraint.i].push_back(constraint.j);
			must_link[constraint.j].push_back(constraint.i);
		} else { // DIFF_CLUSTER
			cannot_link[constraint.i].push_back(constraint.j);
			cannot_link[constraint.j].push_back(constraint.i);
		}
	}
	
	// Find connected components for must-link constraints (transitive closure)
	std::vector<int> component_id(n, -1);
	int num_components = 0;
	for (int i = 0; i < n; ++i) {
		if (component_id[i] == -1) {
			// Start DFS/BFS to find connected component
			std::vector<int> component;
			std::vector<bool> visited(n, false);
			std::queue<int> q;
			q.push(i);
			visited[i] = true;
			
			while (!q.empty()) {
				int curr = q.front();
				q.pop();
				component.push_back(curr);
				
				for (int neighbor : must_link[curr]) {
					if (!visited[neighbor]) {
						visited[neighbor] = true;
						q.push(neighbor);
					}
				}
			}
			
			// Assign component ID to all points in this component
			for (int point : component) {
				component_id[point] = num_components;
			}
			num_components++;
		}
	}
	
	// For each connected component, assign all points to the same cluster
	std::vector<int> new_assignment = assignment; // Copy current assignment
	
	for (int comp = 0; comp < num_components; ++comp) {
		// Find all points in this component
		std::vector<int> component_points;
		for (int i = 0; i < n; ++i) {
			if (component_id[i] == comp) {
				component_points.push_back(i);
			}
		}
		
		if (component_points.empty()) continue;
		
		// Find the best cluster for this entire component
		// Try each possible cluster and check if it violates cannot-link constraints
		std::vector<double> component_costs(k, 0.0);
		std::vector<bool> cluster_valid(k, true);
		
		for (int cluster = 0; cluster < k; ++cluster) {
			bool valid = true;
			double total_cost = 0.0;
			
			// Check if this cluster assignment violates cannot-link constraints
			for (int point : component_points) {
				// Check cannot-link with points already assigned to this cluster
				for (int other_point = 0; other_point < n; ++other_point) {
					if (new_assignment[other_point] == cluster && 
						std::find(cannot_link[point].begin(), cannot_link[point].end(), other_point) != cannot_link[point].end()) {
						valid = false;
						break;
					}
				}
				if (!valid) break;
				
				// Add distance cost
				total_cost += (dataPoints[point] - centroids[cluster]).squaredNorm();
			}
			
			cluster_valid[cluster] = valid;
			component_costs[cluster] = total_cost;
		}
		
		// Find the best valid cluster
		int best_cluster = -1;
		double best_cost = std::numeric_limits<double>::max();
		
		for (int cluster = 0; cluster < k; ++cluster) {
			if (cluster_valid[cluster] && component_costs[cluster] < best_cost) {
				best_cost = component_costs[cluster];
				best_cluster = cluster;
			}
		}
		
		// If no valid cluster found, this is an infeasible assignment
		// For now, assign to closest cluster (this shouldn't happen in a well-formed problem)
		if (best_cluster == -1) {
			std::cerr << "Warning: No valid cluster found for component containing point " << component_points[0] << std::endl;
			best_cluster = 0;
			double min_dist = std::numeric_limits<double>::max();
			for (int cluster = 0; cluster < k; ++cluster) {
				double total_dist = 0.0;
				for (int point : component_points) {
					total_dist += (dataPoints[point] - centroids[cluster]).squaredNorm();
				}
				if (total_dist < min_dist) {
					min_dist = total_dist;
					best_cluster = cluster;
				}
			}
		}
		
		// Assign all points in component to the best cluster
		for (int point : component_points) {
			if (new_assignment[point] != best_cluster) {
				new_assignment[point] = best_cluster;
				changed = true;
			}
		}
	}
	
	assignment = new_assignment;
	return changed;
}

void updateCentroids(const VectorXdList& dataPoints, VectorXdList& centroids, const std::vector<int>& assignment, int k) {
	std::vector<int> clusterSizes(k, 0);
	VectorXdList newCentroids(k, Eigen::VectorXd::Zero(centroids[0].size()));

	for (size_t i = 0; i < dataPoints.size(); ++i) {
		if (assignment[i] >= 0 && assignment[i] < k) {
			newCentroids[assignment[i]] += dataPoints[i];
			clusterSizes[assignment[i]]++;
		} else {
			std::cerr << "Warning: Invalid assignment " << assignment[i] << " for point " << i << " in updateCentroids" << std::endl;
		}
	}

	for (int j = 0; j < k; ++j) {
		if (clusterSizes[j] > 0) {
			centroids[j] = newCentroids[j] / clusterSizes[j];
		}
	}
}

double computeWCSS(const VectorXdList& dataPoints, const VectorXdList& centroids, const std::vector<int>& assignment) {
	double totalWCSS = 0.0;
	for (size_t i = 0; i < dataPoints.size(); ++i) {
		if (assignment[i] >= 0 && assignment[i] < (int)centroids.size()) {
			totalWCSS += (dataPoints[i]-centroids[assignment[i]]).squaredNorm();
		} else {
			std::cerr << "Warning: Invalid assignment " << assignment[i] << " for point " << i << std::endl;
		}
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

double KMeansObjPartitionMatrix(const Eigen::MatrixXd& Xsol, const VectorXdList& dataPoints, int k){
    std::vector<int> assignment = createAssignment(Xsol, k);

    return KMeansObjAssignment(assignment, dataPoints, k);
}

double KMeansObjAssignment(const std::vector<int>& assignment, const VectorXdList& dataPoints, int k){
    int n = dataPoints.size();
    int dim = dataPoints[0].size();
    
	VectorXdList centroids(k, Eigen::VectorXd::Zero(dim));
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

std::pair<double, std::vector<int>> runKMeans(const VectorXdList& dataPoints, int k, int maxIterations, int random_seed){
    int n = dataPoints.size();

    // initialized centroids and do assignment
	VectorXdList centroids = initializeCentroidsPlusPlus(dataPoints, k, random_seed);
    std::vector<int> assignment(n, -1);
    bool changed = assignClusters(dataPoints, centroids, assignment);
    double currentWCSS = computeWCSS(dataPoints, centroids, assignment);

    for (int iter = 0; iter < maxIterations; ++iter) {
        if (!changed) {
            break;
        }
		VectorXdList oldCentroids = centroids;
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