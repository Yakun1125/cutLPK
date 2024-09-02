#include "Lloyd.h"

//Kmeans
double euclideanDistance(const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
	return (a - b).squaredNorm();
}

// K-means++ initialization
std::vector<Eigen::VectorXd> initializeCentroidsPlusPlus(const std::vector<Eigen::VectorXd>& dataPoints, int k, std::mt19937& gen) {
	std::vector<Eigen::VectorXd> centroids;
	std::uniform_int_distribution<> dis(0, dataPoints.size() - 1);

	centroids.push_back(dataPoints[dis(gen)]);

	for (int i = 1; i < k; ++i) {
		std::vector<double> distances(dataPoints.size(), std::numeric_limits<double>::max());

		for (size_t j = 0; j < dataPoints.size(); ++j) {
			for (size_t c = 0; c < centroids.size(); ++c) {
				double dist = euclideanDistance(dataPoints[j], centroids[c]);
				distances[j] = std::min(distances[j], dist);
			}
		}

		std::discrete_distribution<> weightedDist(distances.begin(), distances.end());
		centroids.push_back(dataPoints[weightedDist(gen)]);
	}

	return centroids;
}

// Assign data points to the nearest centroid
bool assignClusters(const std::vector<Eigen::VectorXd>& dataPoints, std::vector<Eigen::VectorXd>& centroids, std::vector<int>& assignment) {
	bool changed = false;
	for (size_t i = 0; i < dataPoints.size(); ++i) {
		double minDist = std::numeric_limits<double>::max();
		int closestCentroid = -1;
		for (int j = 0; j < centroids.size(); ++j) {
			double dist = euclideanDistance(dataPoints[i], centroids[j]);
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

// Recompute centroids based on current assignment
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

// Compute within-cluster sum of squares
double computeWCSS(const std::vector<Eigen::VectorXd>& dataPoints, const std::vector<Eigen::VectorXd>& centroids, const std::vector<int>& assignment) {
	double totalWCSS = 0.0;
	for (size_t i = 0; i < dataPoints.size(); ++i) {
		totalWCSS += euclideanDistance(dataPoints[i], centroids[assignment[i]]);
	}
	return totalWCSS;
}

// Create partition matrix based on assignment
Eigen::MatrixXd createPartitionMatrix(const std::vector<int>& assignment, int n, int k) {
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


// Main K-means function
std::pair<double, Eigen::MatrixXd> runKMeans(const std::vector<Eigen::VectorXd>& dataPoints, int k, int maxIterations, int numTrials, int random_seed) {
	std::mt19937 gen(random_seed);
	int n = dataPoints.size();
	Eigen::MatrixXd bestPartitionMatrix = Eigen::MatrixXd::Zero(n, n);
	double bestWCSS = std::numeric_limits<double>::max();

	for (int trial = 0; trial < numTrials; ++trial) {
		std::vector<Eigen::VectorXd> centroids = initializeCentroidsPlusPlus(dataPoints, k, gen);
		std::vector<int> assignment(n, 0);
		double currentWCSS = std::numeric_limits<double>::max();

		for (int iter = 0; iter < maxIterations; ++iter) {
			bool changed = assignClusters(dataPoints, centroids, assignment);
			updateCentroids(dataPoints, centroids, assignment, k);
			double newWCSS = computeWCSS(dataPoints, centroids, assignment);

			if (!changed || std::abs(newWCSS - currentWCSS) < 1e-6) {
				break;
			}
			currentWCSS = newWCSS;
		}

		if (currentWCSS < bestWCSS) {
			bestWCSS = currentWCSS;
			bestPartitionMatrix = createPartitionMatrix(assignment, n, k);
		}
	}

	return std::make_pair(bestWCSS,bestPartitionMatrix);
}

// from partition matrix to recover the cluster assignment and clustering cost
double KMeansObj(const Eigen::SparseMatrix<double>& Xsol, const Eigen::MatrixXd& dis_matrix, int N) {
	double sum = 0.0;
	// Iterate over all elements in the product and Xsol
	for (int i = 0; i < N; ++i) {
		for (Eigen::SparseMatrix<double>::InnerIterator it(Xsol, i); it; ++it) {
			double productValue = it.value();
			int row = it.row();   // row index
			int col = it.col();   // col index (here it is equal to 'i', because we are using a column iterator)

			double product = it.value() * dis_matrix(row, col);
			sum += product;
		}
	}
	return sum;
}