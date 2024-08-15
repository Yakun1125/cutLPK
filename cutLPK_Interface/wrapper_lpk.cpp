#include "wrapper_lpk.h"
#include "wrapper_highs.h"
#include "mps_lp.h"

#include <cassert>
#include <string>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
#include <vector>
#include <tuple>
#include <list>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <map>
#include <random>
#include <iterator>
#include <algorithm>
#include <chrono>
#include <limits>
#include "Highs.h"
#include <omp.h>
#include <iomanip>

#include "ortools/base/init_google.h"
#include "ortools/pdlp/iteration_stats.h"
#include "ortools/pdlp/primal_dual_hybrid_gradient.h"
#include "ortools/pdlp/quadratic_program.h"
#include "ortools/pdlp/solve_log.pb.h"
#include "ortools/pdlp/solvers.pb.h"

#include "gurobi_c++.h"


namespace pdlp = ::operations_research::pdlp;
constexpr double kInfinity = std::numeric_limits<double>::infinity();


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
Eigen::MatrixXd runKMeans(const std::vector<Eigen::VectorXd>& dataPoints, int k, int maxIterations, int numTrials, int random_seed) {
	//std::random_device rd;
	//std::mt19937 gen(rd());
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

	return bestPartitionMatrix;
}

Eigen::MatrixXd postHeuristic(const Eigen::MatrixXd& Xsol, const std::vector<Eigen::VectorXd>& dataPoints, int k, int maxIterations) {
	int N = dataPoints.size();
	// compute best rank-k approximation X_k for Xsol
	Eigen::SelfAdjointEigenSolver<Eigen::SparseMatrix<double>> eigensolver(Eigen::SparseMatrix<double>(Xsol.sparseView()));
	if (eigensolver.info() != Eigen::Success) {
		throw std::runtime_error("Eigenvalue decomposition failed!");
	}

	// Extract the eigenvalues and eigenvectors
	Eigen::VectorXd eigenvalues = eigensolver.eigenvalues();
	Eigen::MatrixXd eigenvectors = eigensolver.eigenvectors();

	// Create a list of indices based on the sorting of the eigenvalues by their absolute values
	std::vector<int> idx(eigenvalues.size());
	std::iota(idx.begin(), idx.end(), 0);

	// Sort indices based on comparing absolute values of corresponding eigenvalues
	std::sort(idx.begin(), idx.end(),
		[&eigenvalues](int i1, int i2) { return fabs(eigenvalues(i1)) > fabs(eigenvalues(i2)); });

	// Arrange the eigenvalues and eigenvectors in descending order of their absolute values
	Eigen::VectorXd sortedEigenvalues(k);
	Eigen::MatrixXd sortedEigenvectors(eigenvectors.rows(), k);

	for (int i = 0; i < k; ++i) {
		sortedEigenvalues(i) = eigenvalues(idx[i]);
		sortedEigenvectors.col(i) = eigenvectors.col(idx[i]);
	}

	// Construct the rank-k approximation matrix
	Eigen::MatrixXd X_k = sortedEigenvectors * sortedEigenvalues.asDiagonal() * sortedEigenvectors.transpose();

	double diff = (Xsol - X_k).norm();
	std::cout << "Difference between rank_k_approx and X : " << diff << std::endl;

	int d = dataPoints[0].size();

	// Use QR decomposition with column pivoting to find k independent rows
	Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(X_k.transpose());
	Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> P = qr.colsPermutation();

	std::unordered_set<int> selectedRows;
	for (int i = 0; i < k; ++i) {
		int idx = P.indices()(i);
		selectedRows.insert(idx);
	}

	std::vector<Eigen::VectorXd> centroids;
	for (const auto& rowIndex : selectedRows) {
		Eigen::VectorXd centroid = Eigen::VectorXd::Zero(d);
		for (int i = 0; i < d; ++i) {
			for (int j = 0; j < N; ++j) {
				centroid(i) += X_k(rowIndex, j) * dataPoints[j](i);
			}
		}
		centroids.push_back(centroid);
	}


	Eigen::MatrixXd bestPartitionMatrix = Eigen::MatrixXd::Zero(N, N);
	double bestWCSS = std::numeric_limits<double>::max();

	// run Llyod's method until centroids are fixed
	std::vector<int> assignment(N, 0);
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
	bestPartitionMatrix = createPartitionMatrix(assignment, N, k);

	//std::cout << std::fixed << std::setprecision(6) <<" ,Llyod's method obj = " << currentWCSS << std::endl; std::cout.unsetf(std::ios_base::fixed);

	return bestPartitionMatrix;

}

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


namespace std {
	template<>
	struct hash<tuple<int, int, int>> {
		size_t operator()(const tuple<int, int, int>& key) const {
			const int& x = std::get<0>(key);
			const int& y = std::get<1>(key);
			const int& z = std::get<2>(key);


			size_t hash_x = std::hash<int>()(x);
			size_t hash_y = std::hash<int>()(y);
			size_t hash_z = std::hash<int>()(z);

			return (hash_x << 16) ^ (hash_y << 8) ^ (hash_z) ^ (hash_x >> 16);

			//// Prime numbers for hash combination
			//const size_t prime1 = 101;
			//const size_t prime2 = 103;

			//size_t combined_hash = hash_x;
			//combined_hash = combined_hash * prime1 + hash_y;
			//combined_hash = combined_hash * prime2 + hash_z;

			//return combined_hash;
		}
	};
}

struct TriangleInequality {
	//int i, j, k;
	std::list<int> ineq_idx;
	double dual_value;
	TriangleInequality(std::list<int> ineq_idx, double dual_value = 1e5)
		: ineq_idx(ineq_idx), dual_value(dual_value) {}
};

// Function to calculate the total cost of a subgraph
double calculateSubgraphCost(int source, const std::vector<int>& subgraph, const Eigen::MatrixXd& Xsol) {
	double totalCost = 0;

	// Cost from source to the first node in the subgraph and from the last node back to the source
	totalCost -= Xsol(source, source); // Cost of choosing the source node

	// Iterate through the subgraph nodes and calculate the cost
	for (size_t i = 0; i < subgraph.size(); ++i) {
		if (i < subgraph.size() - 1) {
			totalCost -= Xsol(subgraph[i], subgraph[i + 1]); // Subtract cost between consecutive nodes
		}
		totalCost += Xsol(source, subgraph[i]); // Add cost from the source to the current node
	}

	return totalCost;
}

void extend_chain(int source, const Eigen::MatrixXd& Xsol, std::vector<int>& chain, double current_cost, const int N, const int max_T, double cuts_vio_tol, std::vector<std::list<std::pair<int, std::vector<int>>>>& violated_cuts, int& max_list_size, const int max_init) {
	if (chain.size() >= max_T || max_list_size >= max_init) {
		return; // Stop if the chain is too long or enough cuts have been found
	}

	int last_node = chain.back();
	for (int next = last_node + 1; next < N; ++next) {
		if (next != source && std::find(chain.begin(), chain.end(), next) == chain.end()) {
			double additional_cost = 0.0;
			for (int k : chain) {
				if (k < next) {
					additional_cost += Xsol(k, next);
				}
			}
			double next_cost = current_cost + Xsol(source, next) - additional_cost;

			// Record if this extension is a violated cut
			if (next_cost > cuts_vio_tol) {
#pragma omp critical
				{
					if (max_list_size < max_init) {
						std::vector<int> recording_chain = chain;
						recording_chain.push_back(next); // Include this node for recording
						violated_cuts[recording_chain.size() - 1].push_back(std::make_pair(source, recording_chain));
						max_list_size++;
					}
				}
			}

			// Always recurse to explore further, regardless of current cost meeting the violation threshold
			std::vector<int> new_chain = chain;  // Create a new chain including the next node
			new_chain.push_back(next);
			extend_chain(source, Xsol, new_chain, next_cost, N, max_T, cuts_vio_tol, violated_cuts, max_list_size, max_init);
		}
	}
}

void exact_separation(const Eigen::MatrixXd& Xsol, std::vector<std::list<std::pair<int, std::vector<int>>>>& violated_cuts, int max_T, int N, int max_init, double cuts_vio_tol) {
	int max_list_size = 0;
#pragma omp parallel for shared(violated_cuts, max_list_size)
	for (int source = 0; source < N; ++source) {
		for (int j = 0; j < N; ++j) {
			if (j != source) {
				std::vector<int> initial_chain = { j };
				double initial_cost = -Xsol(source, source) + Xsol(source, j);
				extend_chain(source, Xsol, initial_chain, initial_cost, N, max_T, cuts_vio_tol, violated_cuts, max_list_size, max_init);
			}
		}
	}
}

void separation_scheme(const Eigen::MatrixXd& Xsol, std::vector<std::list<std::pair<int, std::vector<int>>>>& violated_cuts, int max_T, int N, int max_init, double cuts_vio_tol) {
	int max_list_size = 0;
#pragma omp parallel for shared(violated_cuts, max_list_size)
	for (int source = 0; source < N; ++source) {
		for (int j = 0; j < N; ++j) {
			if (j != source) {
				std::vector<int> chain = { j };
				int current_node = j;
				double current_cost = -Xsol(source, source) + Xsol(source, j);

				for (int size = 2; size <= max_T; ++size) {
					if (max_list_size >= max_init) break; // Early exit check

					int best_next_node = -1;
					double max_next_cost = -std::numeric_limits<double>::infinity();
					std::vector<int> potential_chain;

					for (int next = current_node + 1; next < N; ++next) {
						if (next != source) {
							double additional_cost = 0.0;
							for (int k : chain) {
								if (k < next) {
									additional_cost += Xsol(k, next);
								}
							}
							double next_cost = current_cost + Xsol(source, next) - additional_cost;
							if (next_cost > max_next_cost) {
								max_next_cost = next_cost;
								best_next_node = next;
								potential_chain = chain;
								potential_chain.push_back(next);
							}
						}
					}

					if (best_next_node != -1 && max_list_size < max_init) {
						if (max_next_cost > cuts_vio_tol && potential_chain.size() > 1) {
#pragma omp critical
							{
								if (max_list_size < max_init) {
									violated_cuts[size - 2].push_back(std::make_pair(source, potential_chain));
#pragma omp atomic
									max_list_size++;
								}
							}
#pragma omp flush(max_list_size)
						}
						chain = potential_chain;
						current_cost = max_next_cost;
						current_node = best_next_node;
					}
					else {
						break;
					}
				}
			}
		}
	}
}


extern "C" int solve_lpk(const char* filename, int num_cluster, int max_init, int max_per_iter, int warm_start, int t_upper_bound, double initial_time_limit, double time_limit_iter, double time_limit_all, const char* solver
	, double initial_pdlp_tol, double tolerance_per_iter, double ub_pdlp_tol, double cuts_vio_tol, double cuts_act_tol, double opt_gap, int random_seed) {


	/*
	All global utility variables
	*/
	std::vector<Eigen::VectorXd> dataPoints;
	Eigen::MatrixXd dis_matrix;
	int N;
	Eigen::MatrixXd Xsol;// eventual solution that will be returned
	Eigen::MatrixXd kmeans_Xsol;// partition matrix obtained from Kmeans++
	std::vector<TriangleInequality> cutting_planes;
	int initial_size;

	int numVars;
	std::vector<double> lb;
	std::vector<double> ub;
	std::vector<double> obj_coef;
	std::vector<double> cons_lb;
	std::vector<double> cons_ub;




	auto start_time = std::chrono::high_resolution_clock::now();
	std::ifstream file(filename);
	std::string line;


	auto time1 = std::chrono::high_resolution_clock::now();
	// Read data points from file
	while (std::getline(file, line)) {
		std::stringstream lineStream(line);
		std::string cell;
		std::vector<double> dataPoint;

		while (std::getline(lineStream, cell, ',')) {
			dataPoint.push_back(std::stod(cell)); // Convert string to double
		}

		Eigen::VectorXd point = Eigen::Map<Eigen::VectorXd>(dataPoint.data(), dataPoint.size());
		dataPoints.push_back(point);
	}

	// for iterative solve usage
	N = dataPoints.size();
	if (N < 1) {
		std::cout << "read data failed\n";
		return 1;
	}

	dis_matrix.resize(N, N);
	std::cout << N << std::endl;
	// Compute squared Euclidean distances_
	for (int i = 0; i < N; ++i) {
		dis_matrix(i, i) = 0;
		for (int j = i + 1; j < N; ++j) {
			dis_matrix(i, j) = (dataPoints[i] - dataPoints[j]).squaredNorm();
			dis_matrix(j, i) = dis_matrix(i, j);
		}
	}



	/*
	Prepare the triangle inequalities and clean up the data
	*/

	Xsol.resize(N, N);


	unsigned long long totalCombinations = static_cast<unsigned long long>(N) * (N - 1) * (N - 2) / 2;
	//unsigned long long desiredSize = static_cast<unsigned long long>(N) * (N - 1) * (N - 2) / 2;
	initial_size = static_cast<int>(std::min(totalCombinations, static_cast<unsigned long long>(max_init)));
	//initial_size = max_init;
  //std::cout<<initial_size<<std::endl;
	cutting_planes.reserve(initial_size);

	/************************
	Randomly choose the triangle ineq
	*************************/

	int baseRow = N + 1; // this depends on the version of LPK; control the start row index of triangle inequalities
	// to do: verify if Triplet needs to be double
	std::vector<Eigen::Triplet<double>> triplets_cuts_initial;// used to store the initial cutting planes' variables and their coefficients
	triplets_cuts_initial.reserve(4 * initial_size);
	double KmeansPlusPlus_Cost = kInfinity;
	if (warm_start == 1 || warm_start == 2) {//using Llyod's method to initialize
		auto llyod_start = std::chrono::high_resolution_clock::now();
		kmeans_Xsol = runKMeans(dataPoints, num_cluster, 100000, 50, random_seed);
		Eigen::MatrixXd Kmeansres = kmeans_Xsol.cwiseProduct(dis_matrix);
		KmeansPlusPlus_Cost = Kmeansres.sum() / 2;
		auto llyod_end = std::chrono::high_resolution_clock::now();




		auto llyod_duration = std::chrono::duration_cast<std::chrono::milliseconds>(llyod_end - llyod_start);
		std::cout << std::fixed << std::setprecision(6) << "Kmeans++ obj = " << KmeansPlusPlus_Cost; std::cout.unsetf(std::ios_base::fixed);
		std::cout << " , wall_time used: " << llyod_duration.count() / 1e3;
		//dataPoints.clear();

		int added_count = 0;
		unsigned long long scaned_count = 0;
		unsigned long long act_size = 0;
		unsigned long long total_size = (N) * (N - 1) * (N - 2) / 2;


		if (warm_start == 1) {
			int size_each_i = max_init / N;// when problem size is large, we added first size_each_i triangle inequalities for each i,j 
			for (int i = 0; i < N; i++) {
				int added_count_i = 0;
				for (int j = 0; j < N; j++) {
					if (i != j) {
						for (int k = j + 1; k < N; ++k) {
							if (k != i) {
								double violation = kmeans_Xsol(i, j) + kmeans_Xsol(i, k) - kmeans_Xsol(i, i) - kmeans_Xsol(j, k);

								if (violation > -cuts_act_tol && violation < cuts_act_tol) {
									int newRow = baseRow + added_count;
									cutting_planes.emplace_back(TriangleInequality(std::list<int>{i, j, k}, -2));
									triplets_cuts_initial.emplace_back(newRow, std::min(i, j) * (2 * N - std::min(i, j) + 1) / 2 + std::max(i, j) - std::min(i, j), 1);
									triplets_cuts_initial.emplace_back(newRow, std::min(i, k) * (2 * N - std::min(i, k) + 1) / 2 + std::max(i, k) - std::min(i, k), 1);
									triplets_cuts_initial.emplace_back(newRow, i * (2 * N - i + 1) / 2, -1);
									triplets_cuts_initial.emplace_back(newRow, j * (2 * N - j + 1) / 2 + k - j, -1);
									added_count_i++;
									added_count++;
								}
							}
							if (added_count_i >= size_each_i) {
								break;
							}
						}
					}
					if (added_count_i >= size_each_i) {
						break;
					}
				}
			}
			std::cout << " ,initial inequalities: " << added_count << std::endl;
		}
		else {
			//std::random_device rd;
			//std::mt19937 gen(rd());
			std::mt19937 gen(random_seed);
			std::uniform_real_distribution<double> dis(0.0, 1.0);

			for (int i = 0; i < N; ++i) {
				for (int j = 0; j < N; ++j) {
					if (i != j) {
						for (int k = j + 1; k < N; ++k) {
							if (k != i) {
								double violation = kmeans_Xsol(i, j) + kmeans_Xsol(i, k) - kmeans_Xsol(i, i) - kmeans_Xsol(j, k);

								if (violation > -cuts_act_tol && violation < cuts_act_tol) {
									unsigned long long remaining_combinations = static_cast<unsigned long long>(N) * (N - 1) * (N - 2) / 2 - scaned_count;
									double p = double(initial_size - added_count) / double(remaining_combinations);
									act_size++;
									if (dis(gen) < p) {
										int newRow = baseRow + added_count;
										cutting_planes.emplace_back(TriangleInequality(std::list<int>{i, j, k}, -2));
										triplets_cuts_initial.emplace_back(newRow, std::min(i, j) * (2 * N - std::min(i, j) + 1) / 2 + std::max(i, j) - std::min(i, j), 1);
										triplets_cuts_initial.emplace_back(newRow, std::min(i, k) * (2 * N - std::min(i, k) + 1) / 2 + std::max(i, k) - std::min(i, k), 1);
										triplets_cuts_initial.emplace_back(newRow, i * (2 * N - i + 1) / 2, -1);
										triplets_cuts_initial.emplace_back(newRow, j * (2 * N - j + 1) / 2 + k - j, -1);
										added_count++;
									}
								}
								scaned_count++;
							}
						}
					}
				}
			}

			std::cout << " ,initial inequalities: " << added_count << " ,active size " << act_size << std::endl;
		}

		initial_size = added_count;
	}
	else {
		auto llyod_start = std::chrono::high_resolution_clock::now();
		kmeans_Xsol = runKMeans(dataPoints, num_cluster, 100000, 50, random_seed);
		Eigen::MatrixXd Kmeansres = kmeans_Xsol.cwiseProduct(dis_matrix);
		KmeansPlusPlus_Cost = Kmeansres.sum() / 2;
		auto llyod_end = std::chrono::high_resolution_clock::now();




		auto llyod_duration = std::chrono::duration_cast<std::chrono::milliseconds>(llyod_end - llyod_start);
		std::cout << std::fixed << std::setprecision(6) << "Kmeans++ obj = " << KmeansPlusPlus_Cost; std::cout.unsetf(std::ios_base::fixed);
		std::cout << " , wall_time used: " << llyod_duration.count() / 1e3;

		std::cout << "select first initial size's triangle inequality";

		int added_count = 0;

		for (int i = 0; i < N; ++i) {
			for (int j = 0; j < N; ++j) {
				if (i != j) {
					for (int k = j + 1; k < N; ++k) {
						if (k != i) {
							if (added_count < initial_size) {  // Select the first initial_size triangle inequalities
								int newRow = baseRow + added_count;
								cutting_planes.emplace_back(TriangleInequality(std::list<int>{i, j, k}, -2));
								triplets_cuts_initial.emplace_back(newRow, std::min(i, j) * (2 * N - std::min(i, j) + 1) / 2 + std::max(i, j) - std::min(i, j), 1);
								triplets_cuts_initial.emplace_back(newRow, std::min(i, k) * (2 * N - std::min(i, k) + 1) / 2 + std::max(i, k) - std::min(i, k), 1);
								triplets_cuts_initial.emplace_back(newRow, i * (2 * N - i + 1) / 2, -1);
								triplets_cuts_initial.emplace_back(newRow, j * (2 * N - j + 1) / 2 + k - j, -1);
								added_count++;
							}
							if (added_count >= initial_size) {  // Break the loops if initial_size is reached
								break;
							}
						}
					}
					if (added_count >= initial_size) {  // Break the loops if initial_size is reached
						break;
					}
				}
			}
			if (added_count >= initial_size) {  // Break the loops if initial_size is reached
				break;
			}
		}

		std::cout << " ,initial inequalities: " << added_count << std::endl;
	}

	auto time2 = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(time2 - time1);
	std::cout << "initialization time(reading data, lloyd and identify active constraints): " << duration.count() / 1e3 << std::endl;

	/*******************************
	start implementing the basic part
	**********************************/

	numVars = N * (N + 1) / 2;

	lb = std::vector<double>(numVars, 0.0); // Lower bounds
	ub = std::vector<double>(numVars, kInfinity); // Upper bounds
	obj_coef = std::vector<double>(numVars, 0); // Objective function coefficients

	Eigen::SparseMatrix<double, Eigen::ColMajor> ConsMatrix;// constraint matrix
	// index mapping logic is (i,j) j>i index = i*(2*N-i+1)/2+j-i
	int index = 0;
	for (int i = 0; i < N; ++i) {
		for (int j = i; j < N; ++j) {
			obj_coef[index++] = dis_matrix(i, j);
		}
	}

	// basic part constraint bounds
	std::vector<double> cons_lb_basic(1 + N, 1);
	std::vector<double> cons_ub_basic(1 + N, 1);
	cons_lb_basic[0] = num_cluster;
	cons_ub_basic[0] = num_cluster;

	// Reserve space for triplets based on an estimate of non-zero elements
	std::vector<Eigen::Triplet<double>> triplets_basic;
	triplets_basic.reserve((N + 1) * N);

	// Add non-zeros for the original matrix part
	for (int i = 0; i < N; ++i) {
		int col = i * (2 * N - i + 1) / 2;
		triplets_basic.emplace_back(0, col, 1); // First row, diagonal elements set to 1
	}

	for (int i = 1; i <= N; ++i) {
		for (int j = 0; j < N; ++j) {
			int col = std::min(i - 1, j) * (2 * N - std::min(i - 1, j) + 1) / 2 + std::max(i - 1, j) - std::min(i - 1, j);
			triplets_basic.emplace_back(i, col, 1); // Subsequent rows
		}
	}

	// Combine the basic part and the triangle inequalities
	std::vector<Eigen::Triplet<double>> combinedTriplets(triplets_basic);

	combinedTriplets.insert(combinedTriplets.end(), triplets_cuts_initial.begin(), triplets_cuts_initial.end());
	ConsMatrix.resize(baseRow + initial_size, N * (N + 1) / 2);
	ConsMatrix.setFromTriplets(combinedTriplets.begin(), combinedTriplets.end());

	cons_lb = cons_lb_basic;
	cons_ub = cons_ub_basic;
	//std::vector<double> cons_lb_tri(initial_size, -2);
	std::vector<double> cons_lb_tri(initial_size, -kInfinity);
	std::vector<double> cons_ub_tri(initial_size, 0);
	cons_lb.insert(cons_lb.end(), cons_lb_tri.begin(), cons_lb_tri.end());
	cons_ub.insert(cons_ub.end(), cons_ub_tri.begin(), cons_ub_tri.end());

	/**********************************************
	* Iterative Cutting Plane Part
	**********************************************/
	auto prepare_end = std::chrono::high_resolution_clock::now();
	auto prepare_time = std::chrono::duration_cast<std::chrono::milliseconds>(prepare_end - start_time);
	std::cout << "start cutting plane | prepare time: " << prepare_time.count() / 1e3 << std::endl;
	int cut_iter = 0;
	int exit_status = 0;
	int max_T = 2;
	double previous_lower_bound = 0.0;
	double best_primal_obj = 0.0;
	double upper_bound = KmeansPlusPlus_Cost;
	double lower_bound = -1e5;
	double primal_obj = 0.0;
	float tolerance = initial_pdlp_tol;
	float cut_active_tol = 1e-4;
	float time_limit = initial_time_limit;
	bool t_increased = false;
	bool tolerance_decreased = false;
	bool inaccurate_decrease = false;
	bool improvement_failed = false;
	bool time_limit_increased = false;
	std::vector<bool> t_increased_record;
	Eigen::MatrixXd partitionMatrix = kmeans_Xsol;

	double total_pdlp_time = 0.0;
	double total_post_heuristic_time = 0.0;
	double total_separation_time = 0.0;
	double total_adding_removing_cuts_time = 0.0;

	// maitain the record of added cuts
	std::vector<int> added_cuts_record(t_upper_bound - 1, 0);
	std::vector<int> retcode_record;

	while (true) {
		cut_iter++;
		double current_lb = 0;

		auto start = std::chrono::high_resolution_clock::now();
		int retcode = 1;

		if (strcmp(solver, "gpu") == 0) {
			retcode = solve_partial_lpk(tolerance, time_limit, static_cast<void*>(&Xsol), N, &current_lb, &primal_obj, static_cast<void*>(&cutting_planes), static_cast<void*>(&lb),
				static_cast<void*>(&ub), static_cast<void*>(&obj_coef),
				static_cast<void*>(&cons_lb),
				static_cast<void*>(&cons_ub), static_cast<void*>(&ConsMatrix));
		}
		else if (strcmp(solver, "cpu") == 0) {
			retcode = cpu_partial_lpk(tolerance, time_limit, static_cast<void*>(&Xsol), N, &current_lb, &primal_obj, static_cast<void*>(&cutting_planes), static_cast<void*>(&lb),
				static_cast<void*>(&ub), static_cast<void*>(&obj_coef),
				static_cast<void*>(&cons_lb),
				static_cast<void*>(&cons_ub), static_cast<void*>(&ConsMatrix));
		}
		else if (strcmp(solver, "gurobi") == 0) {
			retcode = gurobi_partial_lpk(tolerance, time_limit, static_cast<void*>(&Xsol), N, &current_lb, &primal_obj, static_cast<void*>(&cutting_planes), static_cast<void*>(&lb),
				static_cast<void*>(&ub), static_cast<void*>(&obj_coef),
				static_cast<void*>(&cons_lb),
				static_cast<void*>(&cons_ub), static_cast<void*>(&ConsMatrix));
		}
		else {
			std::cout << "solver not supported" << std::endl;
			exit_status = 1;
			break;
		}
		retcode_record.push_back(retcode);

		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		total_pdlp_time += duration.count() / 1e3;


		if (retcode > 1) {
			std::cout << "solve partial lpk failed" << std::endl;
			exit_status = 1;
			break;
		}

		cons_lb.clear();
		cons_ub.clear();
		ConsMatrix.resize(0, 0);

		std::cout << std::endl;
		if (retcode != 0) {
			std::cout << " solution does not satisfy desired tol" << std::endl;
		}
		std::cout << "iter: " << cut_iter << " ,pdlp wall time: " << duration.count() / 1e3 << " sec " << ", tolerance: " << tolerance;

		// print lower bound and primal obj
		std::cout << std::fixed << std::setprecision(6) << " ,dual obj: " << current_lb << " ,primal obj: " << primal_obj; std::cout.unsetf(std::ios_base::fixed);
		if (current_lb > lower_bound) {
			lower_bound = current_lb;
		}


		//double improvement = (lower_bound - previous_lower_bound) / (previous_lower_bound);
		double improvement = (primal_obj - best_primal_obj) / (best_primal_obj);
		std::cout << " ,primal obj improvement " << improvement << std::endl;
		if (primal_obj > best_primal_obj) {
			best_primal_obj = primal_obj;
		}
		previous_lower_bound = lower_bound;

		Eigen::MatrixXd normMatrix = Xsol * Xsol - Xsol;
		double normValue = normMatrix.norm();

		if (normValue < 1e-2 || improvement < 1e-6) {
			// close enough to optimum, run rounding
				// post heuristic and counting time
			auto post_heuristic_start = std::chrono::high_resolution_clock::now();
			Eigen::MatrixXd temp_partitionMatrix = postHeuristic(Xsol, dataPoints, num_cluster, 1e7);
			Eigen::MatrixXd cost_matrix = temp_partitionMatrix.cwiseProduct(dis_matrix);
			double temp_upper_bound = cost_matrix.sum() / 2;
			// print upper bound and post heuristic time
			std::cout << std::fixed << std::setprecision(6) << " ,post heuristic obj: " << temp_upper_bound; std::cout.unsetf(std::ios_base::fixed);
			auto post_heuristic_end = std::chrono::high_resolution_clock::now();
			auto post_heuristic_duration = std::chrono::duration_cast<std::chrono::milliseconds>(post_heuristic_end - post_heuristic_start);
			std::cout << " ,post heuristic time: " << post_heuristic_duration.count() / 1e3 << " sec" << std::endl;
			total_post_heuristic_time += post_heuristic_duration.count() / 1e3;


			if (temp_upper_bound < upper_bound) {
				upper_bound = temp_upper_bound;
				if ((temp_partitionMatrix - partitionMatrix).norm() > 0.0) {
					partitionMatrix = temp_partitionMatrix;
				}
			}
		}

		std::cout << " ,norm value " << normValue;

		double optimality_gap = (upper_bound - lower_bound) / upper_bound;
		std::cout << " ,optimality gap " << optimality_gap << std::endl;
		if (optimality_gap < opt_gap || normValue < cuts_vio_tol) {
			break;
		}

		// if the problem was not solved in the first iter, we enter the actual cuts loop
		if (cut_iter == 1) {
			tolerance = tolerance_per_iter;
			time_limit = time_limit_iter;
		}
		if (cut_iter == 2) {
			cut_active_tol = cuts_act_tol;
		}




		// check triangle inequalities violation
		start = std::chrono::high_resolution_clock::now();
		// check the partitionm matrix first, if it is then break;


		// check violation
		auto time1 = std::chrono::high_resolution_clock::now();

		int violation_size = 0;
		//omp_set_num_threads(4);
		std::vector<std::list<std::pair<int, std::vector<int>>>> violated_cuts(max_T - 1);
		while (true) {
			separation_scheme(Xsol, violated_cuts, max_T, N, 1.5e7, cuts_vio_tol);
			for (int i = 0; i < max_T - 1; ++i) {
				violation_size += violated_cuts[i].size();
			}
			if (violation_size == 0 && max_T >= 3) {
				std::cout << "start exact eumeration for t up to " << max_T << std::endl;
				exact_separation(Xsol, violated_cuts, max_T, N, 1.5e7, cuts_vio_tol);
			}
			if (violation_size != 0 || max_T == t_upper_bound) {
				break;
			}
			else {
				if (max_T < t_upper_bound) {
					std::cout << "no violated cuts found for t up to: " << max_T << " increase it" << std::endl;
					max_T++;
					violated_cuts.resize(max_T - 1);
				}
			}
		}

		auto time2 = std::chrono::high_resolution_clock::now();
		auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(time2 - time1);
		std::cout << " ,time in checking " << duration2.count() / 1e3 << " sec ";
		total_separation_time += duration2.count() / 1e3;


		auto end_time = std::chrono::high_resolution_clock::now();
		auto elpased_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
		std::cout << " ,elpased wall time: " << elpased_time.count() / 1e3;

		// Before sorting, check condition of stopping
		tolerance_decreased = false;
		if (violation_size == 0 && max_T == t_upper_bound) {
			if (tolerance < 1e-5 && retcode == 0) {// if there is no violated cuts founded and at least the solution of pdlp has accuracy 1e-6
				std::cout << std::endl;
				std::cout << " no violated ineq " << std::endl;
				exit_status = 1;
				break;
			}
			else {
				if (tolerance > ub_pdlp_tol) {
					tolerance_decreased = true;
					tolerance = tolerance * 0.01;
				}
				if (retcode == 1) {
					time_limit = time_limit_all;
				}
			}
		}

		if (elpased_time.count() > time_limit_all * 1e3) {
			std::cout << std::endl;
			std::cout << " time out " << std::endl;
			exit_status = 2;
			break;
		}
		if (cut_iter > 30 * num_cluster) {
			std::cout << std::endl;
			std::cout << " maximum iteration reached " << std::endl;
			exit_status = 3;
			break;
		}

		std::vector<TriangleInequality> tri_added; tri_added.reserve(cutting_planes.size());// buffer for new added triangle inequalities
		std::list<Eigen::Triplet<double>> new_tri; //new_tri.reserve(cutting_planes.size());// buffer for new added triplets constraint matrix
		int tri_idx = 0;// record the index of added triangle inequalities, also record the number of active inequalities
		// start with added inequalities
		//index mapping logic is(i, j) j > i index = i * (2 * N - i + 1) / 2 + j - i


		for (auto& element : cutting_planes) {
			std::list<int> subgraph = element.ineq_idx;  // Consider changing to std::vector if order isn't crucial
			auto it = subgraph.begin();
			int firstElement = *it;
			double violation = -Xsol(firstElement, firstElement);
			std::vector<int> indices(subgraph.begin(), subgraph.end());
			for (size_t j = 1; j < indices.size(); ++j) {
				int currentJ = indices[j];
				violation += Xsol(firstElement, currentJ);
				for (size_t k = j + 1; k < indices.size(); ++k) {
					int currentK = indices[k];
					violation -= Xsol(currentJ, currentK);
				}
			}

			if (violation > -cut_active_tol && violation < cut_active_tol) {

				double dual_value = element.dual_value;
				tri_added.emplace_back(TriangleInequality(subgraph, dual_value));
				int newRow = baseRow + tri_idx;
				int firstTerm = firstElement * (2 * N - firstElement + 1) / 2;
				new_tri.emplace_back(newRow, firstTerm, -1);

				for (size_t j = 1; j < indices.size(); ++j) {
					int currentJ = indices[j];
					int temp_i = std::min(firstElement, currentJ);
					int temp_j = std::max(firstElement, currentJ);
					int indexValue = temp_i * (2 * N - temp_i + 1) / 2 + temp_j - temp_i;
					new_tri.emplace_back(newRow, indexValue, 1);

					for (size_t k = j + 1; k < indices.size(); ++k) {
						int currentK = indices[k];
						// Assuming j and k are indices, and their direct use is intentional
						new_tri.emplace_back(newRow, currentJ * (2 * N - currentJ + 1) / 2 + currentK - currentJ, -1);
					}
				}

				tri_idx += 1;
			}
		}

		int kept_size = tri_idx;

		std::cout << " sec ,kept size " << tri_idx;

		// if (max_per_iter - tri_idx < max_per_iter * 0.5) {
		// 	if (max_per_iter * 2 < 5e7) {
		// 		max_per_iter *= 2;
		// 	}
		// 	else {
		// 		max_per_iter = 5e7;
		// 	}
		// }

		// resize tri_added, avoiding reallocation
		//tri_added.resize(kept_size);

		// check time for this part
		auto time3 = std::chrono::high_resolution_clock::now();
		int remaining_capacity = std::max(max_per_iter - tri_idx, 0);
		if (remaining_capacity < 100000) {
			remaining_capacity += 100000;
		}
		// int temp_capacity = remaining_capacity;

		// std::vector<int> added_capacity(max_T - 1);

		// int size_idx = 0;
		// for (auto& subgraphList : violated_cuts) {
		// 	if (subgraphList.size() < temp_capacity) {
		// 		temp_capacity = temp_capacity - subgraphList.size();
		// 		added_capacity[size_idx] = subgraphList.size();
		// 		size_idx++;
		// 	}
		// 	else if (temp_capacity != 0) {
		// 		subgraphList.sort([&Xsol](const std::pair<int, std::vector<int>>& a, const std::pair<int, std::vector<int>>& b) {
		// 			return calculateSubgraphCost(a.first, a.second, Xsol) < calculateSubgraphCost(b.first, b.second, Xsol);
		// 			});
		// 		//added_capacity[size_idx] = temp_capacity / (max_T - size_idx - 1);
		// 		added_capacity[size_idx] = temp_capacity;
		// 		temp_capacity = 0;
		// 		size_idx++;
		// 	}
		// 	else {
		// 		temp_capacity = 0;
		// 		added_capacity[size_idx] = temp_capacity;
		// 		size_idx++;
		// 	}
		// }


		if (violation_size <= remaining_capacity) {
			for (size_t i = 0; i < violated_cuts.size(); ++i) {
				std::cout << " ,violated size " << violated_cuts[i].size() << " | " << violated_cuts[i].size() << " added";
				added_cuts_record[i] += violated_cuts[i].size();
				std::list<std::pair<int, std::vector<int>>>& subgraphList = violated_cuts[i];
				size_t item = 0;
				for (auto it = subgraphList.begin(); it != subgraphList.end() && item < violated_cuts[i].size(); ++it, ++item) {
					const std::vector<int>& key = it->second;
					int source = it->first;
					std::list<int> elements = { source };
					elements.insert(elements.end(), key.begin(), key.end());

					int newRow = baseRow + tri_idx;
					int firstTerm = source * (2 * N - source + 1) / 2;
					new_tri.emplace_back(newRow, firstTerm, -1);

					// compute nagative one terms
					double l_c = -1;

					for (size_t j = 0; j < key.size(); ++j) {
						int currentJ = key[j];
						int temp_i = std::min(source, currentJ);
						int temp_j = std::max(source, currentJ);
						int indexValue = temp_i * (2 * N - temp_i + 1) / 2 + temp_j - temp_i;
						new_tri.emplace_back(newRow, indexValue, 1);

						for (size_t k = j + 1; k < key.size(); ++k) {
							int currentK = key[k];
							//int currentJ = key[j];
							new_tri.emplace_back(newRow, currentJ * (2 * N - currentJ + 1) / 2 + currentK - currentJ, -1);
							l_c -= 1;
						}
					}
					tri_added.emplace_back(TriangleInequality(elements, l_c));
					tri_idx += 1;
				}
			}
		}
		else {
			std::vector<std::pair<std::pair<int, std::vector<int>>, size_t>> all_subgraphs_with_index;
			all_subgraphs_with_index.reserve(violation_size);
			// Create a vector to record the count for each original subgraphList
			std::vector<int> violated_cuts_count(violated_cuts.size(), 0);
			std::vector<int> added_capacity(violated_cuts.size(), 0);

			for (size_t i = 0; i < violated_cuts.size(); ++i) {
				violated_cuts_count[i] = violated_cuts[i].size();
				for (const auto& subgraph : violated_cuts[i]) {
					all_subgraphs_with_index.push_back({ std::move(subgraph), i });
				}
				// Clear each sublist after moving its contents
				violated_cuts[i].clear();
			}

			// Clear the outer vector
			violated_cuts.clear();

			// Sort all_subgraphs_with_index
			std::sort(all_subgraphs_with_index.begin(), all_subgraphs_with_index.end(),
				[&Xsol](const auto& a, const auto& b) {
					return calculateSubgraphCost(a.first.first, a.first.second, Xsol)
						< calculateSubgraphCost(b.first.first, b.first.second, Xsol);
				});

			// Iterate through remaining_capacity most negative ones
			for (int i = 0; i < remaining_capacity && i < all_subgraphs_with_index.size(); ++i) {
				size_t original_index = all_subgraphs_with_index[i].second;
				added_capacity[original_index]++;

				const std::vector<int>& key = all_subgraphs_with_index[i].first.second;
				int source = all_subgraphs_with_index[i].first.first;
				std::list<int> elements = { source };
				elements.insert(elements.end(), key.begin(), key.end());

				int newRow = baseRow + tri_idx;
				int firstTerm = source * (2 * N - source + 1) / 2;
				new_tri.emplace_back(newRow, firstTerm, -1);

				// compute nagative one terms
				double l_c = -1;

				for (size_t j = 0; j < key.size(); ++j) {
					int currentJ = key[j];
					int temp_i = std::min(source, currentJ);
					int temp_j = std::max(source, currentJ);
					int indexValue = temp_i * (2 * N - temp_i + 1) / 2 + temp_j - temp_i;
					new_tri.emplace_back(newRow, indexValue, 1);

					for (size_t k = j + 1; k < key.size(); ++k) {
						int currentK = key[k];
						//int currentJ = key[j];
						new_tri.emplace_back(newRow, currentJ * (2 * N - currentJ + 1) / 2 + currentK - currentJ, -1);
						l_c -= 1;
					}
				}
				tri_added.emplace_back(TriangleInequality(elements, l_c));
				tri_idx += 1;
			}

			// print violated size and added size
			for (size_t i = 0; i < violated_cuts_count.size(); ++i) {
				std::cout << " ,violated size " << violated_cuts_count[i] << " | " << added_capacity[i] << " added";
				added_cuts_record[i] += added_capacity[i];
			}
		}

		std::cout << " , total cuts " << tri_idx;

		auto time4 = std::chrono::high_resolution_clock::now();
		auto duration3 = std::chrono::duration_cast<std::chrono::milliseconds>(time4 - time3);
		std::cout << " ,time in adding " << duration3.count() / 1e3 << " sec ";


		// consider increase the max_T
		t_increased_record.push_back(t_increased);
		//std::cout<<" t increased "<< t_increased << "max T " << max_T << "t upper bound " << t_upper_bound << std::endl;
		if (t_increased == false && max_T < t_upper_bound) {
			if (improvement < 1e-6 && best_primal_obj < upper_bound && improvement_failed == true) {
				max_T++;
				t_increased = true;
				t_increased_record[cut_iter - 1] = t_increased;
				std::cout << std::endl;
				std::cout << " t increased due to insufficent improvement " << std::endl;
			}
			else if (violation_size < 0.1 * kept_size && best_primal_obj < upper_bound && improvement < 1e-6) {
				max_T++;
				t_increased = true;
				t_increased_record[cut_iter - 1] = t_increased;
				std::cout << std::endl;
				std::cout << "t increased, insufficient violated inequalities" << std::endl;
			}
			else {
				t_increased = false;
				t_increased_record[cut_iter - 1] = t_increased;
			}
		}
		else {
			t_increased = false;
			t_increased_record[cut_iter - 1] = t_increased;
		}

		// consider decrase the tolerance
		if ((optimality_gap < opt_gap * 10 || normValue < 1e-2) && tolerance_decreased == false) {//&& improvement < 1e-6 && t_increased == false
			// if we are approaching optimum
			if (tolerance > ub_pdlp_tol) {
				tolerance = tolerance * 0.01;
				tolerance_decreased = true;
				std::cout << std::endl;
				std::cout << "tolerance decreased, close to optimum" << std::endl;
			}
		}

		if (best_primal_obj > upper_bound && tolerance_decreased == false) {
			if (tolerance > ub_pdlp_tol) {
				tolerance = tolerance * 0.01;
				tolerance_decreased = true;
				std::cout << std::endl;
				std::cout << "tolerance decreased, close to optimum" << std::endl;
			}
		}

		if (improvement_failed == true && improvement < 1e-6 && max_T == t_upper_bound && tolerance_decreased == false) {
			// if there are two consecutive iterations with no improvement, we decrease the tolerance
			if (tolerance > ub_pdlp_tol) {
				tolerance = tolerance * 0.01;
				tolerance_decreased = true;
				std::cout << std::endl;
				std::cout << "tolerance decreased, max T reached and failed in improvment" << std::endl;
			}
		}

		if (improvement < 1e-6) {
			improvement_failed = true;
		}
		else {
			improvement_failed = false;
		}

		// consider increase time limit
		if (cut_iter >= 3) {
			if (retcode_record[cut_iter - 1] == 1 && retcode_record[cut_iter - 2] == 1 && time_limit_increased == false) {
				time_limit += time_limit_iter * 0.5;
				std::cout << std::endl;
				std::cout << "time increased to: " << time_limit << std::endl;
				time_limit_increased = true;
			}
			else {
				if (retcode_record[cut_iter - 1] == 1 && (best_primal_obj > upper_bound || optimality_gap < opt_gap * 10)) {
					time_limit += time_limit_iter * 0.5;
					std::cout << std::endl;
					std::cout << "time increased to: " << time_limit << std::endl;
					time_limit_increased = true;
				}
			}
		}


		// update the dual value and cons matrix
		cutting_planes = tri_added;

		std::vector<Eigen::Triplet<double>> newTriplets(triplets_basic);
		newTriplets.insert(newTriplets.end(), new_tri.begin(), new_tri.end());
		ConsMatrix.resize(baseRow + tri_idx, N * (N + 1) / 2);
		ConsMatrix.setFromTriplets(newTriplets.begin(), newTriplets.end());
		cons_lb = cons_lb_basic;
		cons_ub = cons_ub_basic;
		std::vector<double> lb_cuts(tri_idx, -kInfinity);
		std::vector<double> ub_cuts(tri_idx, 0);

		cons_lb.insert(cons_lb.end(), lb_cuts.begin(), lb_cuts.end());
		cons_ub.insert(cons_ub.end(), ub_cuts.begin(), ub_cuts.end());
		end = std::chrono::high_resolution_clock::now();
		duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		auto adding_removing_cuts_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - end_time);
		std::cout << " ,time in updating " << duration.count() / 1e3 << " sec " << std::endl;
		total_adding_removing_cuts_time += adding_removing_cuts_time.count() / 1e3;

	}

	// Print at last

	Eigen::MatrixXd result = partitionMatrix.cwiseProduct(dis_matrix);
	double objsum = result.sum() / 2;
	auto end_time = std::chrono::high_resolution_clock::now();
	auto elpased_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
	double time_used = elpased_time.count() / 1e3;



	std::cout << std::endl;
	std::cout << "cutting plane solved the problem in " << cut_iter << " iterations with " << time_used << " seconds" << std::endl;
	std::cout << "status: " << exit_status;
	std::cout << std::fixed << std::setprecision(6) << " ,obj = " << objsum; std::cout.unsetf(std::ios_base::fixed);
	std::cout << " Kmeans++ sol same as final sol: " << (kmeans_Xsol - partitionMatrix).norm() << std::endl;
	// print the added cuts record
	std::cout << "added cuts record: ";
	for (int i = 0; i < t_upper_bound - 1; ++i) {
		std::cout << "t = " << i + 2 << ": " << added_cuts_record[i] << " ";
	}
	// print time cost for each part
	std::cout << std::endl;
	std::cout << "total pdlp time: " << total_pdlp_time << " sec" << std::endl;
	std::cout << "total post heuristic time: " << total_post_heuristic_time << " sec" << std::endl;
	std::cout << "total separation time: " << total_separation_time << " sec" << std::endl;
	std::cout << "total adding and removing cuts time: " << total_adding_removing_cuts_time << " sec" << std::endl;


	/******************
	Save Xsol
	******************/
	// std::string solfilename = "./Xsol_" + std::to_string(N) + "_" + std::to_string(num_cluster) + ".csv";

	// std::ofstream solFile(solfilename);
	// if (!solFile.is_open()) {
	// 	std::cerr << "Error opening file \"" << solfilename << "\" for writing." << std::endl;
	// 	return 1;
	// }


	// solFile << "Cutting plane solved the problem in " << cut_iter << " iterations with "
	// 	<< time_used << " seconds, "
	// 	<< "status: " << exit_status << ", obj = " << objsum << "\n"
	// 	<< "Kmeans++ sol same as final sol: " << (kmeans_Xsol - partitionMatrix).norm() << std::endl;


	// for (int i = 0; i < N; ++i) {
	// 	for (int j = 0; j < N; ++j) {
	// 		solFile << partitionMatrix(i, j);
	// 		if (j < N - 1) solFile << ",";
	// 	}
	// 	solFile << "\n";
	// }

	// solFile.close();

	// std::cout << "Data written to " << solfilename << std::endl;

	return 0;
}









// input necessary info to build partial LPK,
// Eigen::SparseMatrix<double, Eigen::ColMajor> Xsol
// int number of points; unordered_map dual_value;
// vec lb; vec ub; vec obj_coef; vec cons_lb; vec cons_ub;
// Eigen::SparseMatrix<double, Eigen::ColMajor> ConsMatrix

extern "C" int solve_partial_lpk(float tolerance, float time_limit, void* XsolPtr, int N, double* dual_obj, double* primal_obj, void* dualValuePtr,
	void* lbPtr, void* ubPtr, void* objCoefPtr,
	void* consLbPtr, void* consUbPtr, void* ConsMatrixPtr) {
	cupdlp_retcode retcode = RETCODE_OK;
	// Cast back to the original types
	auto& Xsol = *static_cast<Eigen::MatrixXd*>(XsolPtr);
	auto& dual_value = *static_cast<std::vector<TriangleInequality>*>(dualValuePtr);
	auto& lb = *static_cast<std::vector<double>*>(lbPtr);
	auto& ub = *static_cast<std::vector<double>*>(ubPtr);
	auto& obj_coef = *static_cast<std::vector<double>*>(objCoefPtr);
	auto& cons_lb = *static_cast<std::vector<double>*>(consLbPtr);
	auto& cons_ub = *static_cast<std::vector<double>*>(consUbPtr);
	auto& ConsMatrix = *static_cast<Eigen::SparseMatrix<double, Eigen::ColMajor>*>(ConsMatrixPtr);

	HighsModel highs;

	// Prepare vectors for defining model
	// Number of variables for the upper triangular part, including the diagonal
	int numVars = N * (N + 1) / 2;
	int numConstr = cons_lb.size();
	std::vector<int> start, _index;
	std::vector<double> value;

	// Reserve space for the total number of non-zero elements
	_index.reserve(ConsMatrix.nonZeros());
	value.reserve(ConsMatrix.nonZeros());

	start.push_back(0); // Start of the first column

	for (int k = 0; k < ConsMatrix.outerSize(); ++k) { // Iterate through each column
		int colStart = ConsMatrix.outerIndexPtr()[k]; // Start index of the current column in the values/indices array
		int colEnd = ConsMatrix.outerIndexPtr()[k + 1]; // End index (one past the last element) of the current column

		for (int idx = colStart; idx < colEnd; ++idx) {
			_index.push_back(ConsMatrix.innerIndexPtr()[idx]); // Row index of the non-zero element
			value.push_back(ConsMatrix.valuePtr()[idx]); // Value of the non-zero element
		}

		start.push_back(colEnd); // Start of the next column is the end of the current column
	}

	// Add variables to the model
	highs.lp_.num_col_ = numVars;
	highs.lp_.col_cost_ = obj_coef;
	highs.lp_.sense_ = ObjSense::kMinimize;
	highs.lp_.col_lower_ = lb;
	highs.lp_.col_upper_ = ub;

	highs.lp_.num_row_ = numConstr;
	highs.lp_.row_lower_ = cons_lb;
	highs.lp_.row_upper_ = cons_ub;
	highs.lp_.a_matrix_.format_ = MatrixFormat::kColwise;
	highs.lp_.a_matrix_.start_ = start;
	highs.lp_.a_matrix_.index_ = _index;
	highs.lp_.a_matrix_.value_ = value;

	Highs* model = new Highs();
	model->setOptionValue("log_to_console", false);
	model->passModel(highs);

	/*********************************
	Start PDLP part
	**********************************/
	// for cuPDLP
	int nCols_pdlp = 0;
	int nRows_pdlp = 0;
	int nEqs_pdlp = 0;
	int nnz_pdlp = 0;
	int status_pdlp = -1;



	cupdlp_float* rhs = NULL;
	cupdlp_float* cost = NULL;
	cupdlp_float* lower = NULL;
	cupdlp_float* upper = NULL;

	// -------------------------
	int* csc_beg = NULL, * csc_idx = NULL;
	double* csc_val = NULL;

	// for model to solve, need to free
	double offset =
		0.0;  // true objVal = sig * c'x + offset, sig = 1 (min) or -1 (max)
	double sense = 1;  // 1 (min) or -1 (max)
	int* constraint_new_idx = NULL;
	int* constraint_type = NULL;

	// for model to solve, need not to free
	int nCols = 0;
	cupdlp_float* col_value = cupdlp_NULL;
	cupdlp_float* col_dual = cupdlp_NULL;
	cupdlp_float* row_value = cupdlp_NULL;
	cupdlp_float* row_dual = cupdlp_NULL;

	// for original model, need to free
	int nCols_org = 0;
	int nRows_org = 0;
	cupdlp_float* col_value_org = cupdlp_NULL;
	cupdlp_float* col_dual_org = cupdlp_NULL;
	cupdlp_float* row_value_org = cupdlp_NULL;
	cupdlp_float* row_dual_org = cupdlp_NULL;

	cupdlp_int value_valid = 0;
	cupdlp_int dual_valid = 0;

	void* model2solve = NULL;

	CUPDLPscaling* scaling =
		(CUPDLPscaling*)cupdlp_malloc(sizeof(CUPDLPscaling));

	// claim solvers variables
	// prepare pointers
	CUPDLP_MATRIX_FORMAT src_matrix_format = CSC;
	CUPDLP_MATRIX_FORMAT dst_matrix_format = CSR_CSC;
	CUPDLPcsc* csc_cpu = cupdlp_NULL;
	CUPDLPproblem* prob = cupdlp_NULL;

	//
	CUPDLPwork* w = cupdlp_NULL;
	cupdlp_float alloc_matrix_time = 0.0;
	cupdlp_float cuda_prepare_time = 0.0;
	cupdlp_float copy_vec_time = 0.0;
	cupdlp_bool ifChangeIntParam[N_INT_USER_PARAM] = { false };
	cupdlp_int intParam[N_INT_USER_PARAM] = { 0 };
	cupdlp_bool ifChangeFloatParam[N_FLOAT_USER_PARAM] = { false };
	cupdlp_float floatParam[N_FLOAT_USER_PARAM] = { 0.0 };
	ifChangeFloatParam[D_TIME_LIM] = true;
	floatParam[D_TIME_LIM] = time_limit;
	ifChangeFloatParam[D_PRIMAL_TOL] = true;
	floatParam[D_PRIMAL_TOL] = tolerance;
	ifChangeFloatParam[D_DUAL_TOL] = true;
	floatParam[D_DUAL_TOL] = tolerance;
	ifChangeFloatParam[D_GAP_TOL] = true;
	floatParam[D_GAP_TOL] = 1e-4;
	char* fout = "./solution-sum.json";
	char* fout_sol = "./solution.json";


	int col_idx = 0;
	int num_tri_ineq = dual_value.size();
	int row_idx = numConstr - num_tri_ineq;
	double dual_value_sum = 0.0;
	Eigen::VectorXd r;
	Eigen::VectorXd epsilon;
	Eigen::VectorXd delta;
	Eigen::VectorXd row_dual_org_vec;

	int return_code = 0;


	getModelSize_highs(model, &nCols_org, &nRows_org, NULL);
	nCols = nCols_org;
	model2solve = model;

	CUPDLP_CALL(formulateLP_highs(model2solve, &cost, &nCols_pdlp, &nRows_pdlp,
		&nnz_pdlp, &nEqs_pdlp, &csc_beg, &csc_idx,
		&csc_val, &rhs, &lower, &upper, &offset, &sense,
		&nCols, &constraint_new_idx, &constraint_type));
	CUPDLP_CALL(Init_Scaling(scaling, nCols_pdlp, nRows_pdlp, cost, rhs));
	// the work object needs to be established first
// free inside cuPDLP
	CUPDLP_INIT_ZERO(w, 1);
#if !(CUPDLP_CPU)
	cuda_prepare_time = getTimeStamp();
	CHECK_CUSPARSE(cusparseCreate(&w->cusparsehandle));
	CHECK_CUBLAS(cublasCreate(&w->cublashandle));
	cuda_prepare_time = getTimeStamp() - cuda_prepare_time;
#endif

	CUPDLP_CALL(problem_create(&prob));

	// currently, only supprot that input matrix is CSC, and store both CSC and
	// CSR
	CUPDLP_CALL(csc_create(&csc_cpu));
	csc_cpu->nRows = nRows_pdlp;
	csc_cpu->nCols = nCols_pdlp;
	csc_cpu->nMatElem = nnz_pdlp;
	csc_cpu->colMatBeg = (int*)malloc((1 + nCols_pdlp) * sizeof(int));
	csc_cpu->colMatIdx = (int*)malloc(nnz_pdlp * sizeof(int));
	csc_cpu->colMatElem = (double*)malloc(nnz_pdlp * sizeof(double));
	memcpy(csc_cpu->colMatBeg, csc_beg, (nCols_pdlp + 1) * sizeof(int));
	memcpy(csc_cpu->colMatIdx, csc_idx, nnz_pdlp * sizeof(int));
	memcpy(csc_cpu->colMatElem, csc_val, nnz_pdlp * sizeof(double));



#if !(CUPDLP_CPU)
	csc_cpu->cuda_csc = NULL;
#endif

	CUPDLP_CALL(PDHG_Scale_Data_cuda(csc_cpu, 1, scaling, cost, lower,
		upper, rhs));



	CUPDLP_CALL(problem_alloc(prob, nRows_pdlp, nCols_pdlp, nEqs_pdlp, cost,
		offset, sense, csc_cpu, src_matrix_format,
		dst_matrix_format, rhs, lower, upper,
		&alloc_matrix_time, &copy_vec_time));

	// solve
	w->problem = prob;
	w->scaling = scaling;
	PDHG_Alloc(w);
	w->timers->dScalingTime = 0.0;
	w->timers->dPresolveTime = 0.0;
	CUPDLP_COPY_VEC(w->rowScale, scaling->rowScale, cupdlp_float, nRows_pdlp);
	CUPDLP_COPY_VEC(w->colScale, scaling->colScale, cupdlp_float, nCols_pdlp);

#if !(CUPDLP_CPU)
	w->timers->AllocMem_CopyMatToDeviceTime += alloc_matrix_time;
	w->timers->CopyVecToDeviceTime += copy_vec_time;
	w->timers->CudaPrepareTime = cuda_prepare_time;
#endif

	CUPDLP_INIT_ZERO(col_value_org, nCols_org);
	CUPDLP_INIT_ZERO(col_dual_org, nCols_org);
	CUPDLP_INIT_ZERO(row_value_org, nRows_org);
	CUPDLP_INIT_ZERO(row_dual_org, nRows_org);

	col_value = col_value_org;
	col_dual = col_dual_org;
	row_value = row_value_org;
	row_dual = row_dual_org;

	CUPDLP_CALL(LP_SolvePDHG(w, ifChangeIntParam, intParam, ifChangeFloatParam,
		floatParam, fout, nCols, col_value, col_dual,
		row_value, row_dual, &value_valid, &dual_valid, 0,
		fout_sol, constraint_new_idx, constraint_type,
		&status_pdlp));

	/*****************************************************
	Update Xsol and dual value for next lpk construction
	******************************************************/

	// Report the Xsol
	col_idx = 0;
	for (int i = 0; i < N; ++i) {
		for (int j = i; j < N; ++j) {
			double value = col_value_org[col_idx++];
			Xsol(i, j) = value;
			if (i != j) {
				Xsol(j, i) = value;  // For symmetric matrix
			}
		}
	}


	// dual constraints residual
	// compute vector r by r = ConsMatrix^T * row_dual_org(every row) - obj_coef
	//r = Eigen::VectorXd::Map(obj_coef.data(), obj_coef.size());
	//row_dual_org_vec = Eigen::VectorXd::Map(row_dual_org, numConstr);
	//r -= ConsMatrix.transpose() * row_dual_org_vec;

	r = ConsMatrix.transpose() * Eigen::VectorXd::Map(row_dual_org, numConstr) - Eigen::VectorXd::Map(obj_coef.data(), obj_coef.size());


	// Then we compute the dual_obj manually
	dual_value_sum = row_dual[0] * cons_ub[0];
	for (int i = 1; i < N + 1; i++) {
		dual_value_sum += row_dual[i];
	}

	col_idx = 0;
	for (int i = 0; i < N; ++i) {
		for (int j = i; j < N; ++j) {
			if (r[col_idx] > 0) {
				dual_value_sum -= r[col_idx] * 1;//col_value_org[col_idx]largest_x
			}
			col_idx++;
		}
	}


	// clean all local var: r and row_dual_org_vec
	r.resize(0);
	row_dual_org_vec.resize(0);



	if (w->resobj->termIterate == LAST_ITERATE) {
		//*dual_obj = w->resobj->dDualObj;
		*primal_obj = w->resobj->dPrimalObj;
	}
	else {
		//*dual_obj = w->resobj->dDualObjAverage;
		*primal_obj = w->resobj->dPrimalObjAverage;
	}

	if (w->resobj->termCode == OPTIMAL) {
		return_code = 0;
	}
	else if (w->resobj->termCode == TIMELIMIT_OR_ITERLIMIT) {
		return_code = 1;
	}

	*dual_obj = dual_value_sum;

exit_cleanup:
	// free model and solution
	deleteModel_highs(model);
	if (col_value_org != NULL) cupdlp_free(col_value_org);
	if (col_dual_org != NULL) cupdlp_free(col_dual_org);
	if (row_value_org != NULL) cupdlp_free(row_value_org);
	if (row_dual_org != NULL) cupdlp_free(row_dual_org);
	col_value = NULL;
	col_dual = NULL;
	row_value = NULL;
	row_dual = NULL;

	// free problem
	if (scaling) {
		scaling_clear(scaling);
	}

	if (cost != NULL) cupdlp_free(cost);
	if (csc_beg != NULL) cupdlp_free(csc_beg);
	if (csc_idx != NULL) cupdlp_free(csc_idx);
	if (csc_val != NULL) cupdlp_free(csc_val);
	if (rhs != NULL) cupdlp_free(rhs);
	if (lower != NULL) cupdlp_free(lower);
	if (upper != NULL) cupdlp_free(upper);
	if (constraint_new_idx != NULL) cupdlp_free(constraint_new_idx);
	if (constraint_type != NULL) cupdlp_free(constraint_type);

	// free memory
	csc_clear(csc_cpu);
	problem_clear(prob);

	return return_code;
}


extern "C" int cpu_partial_lpk(float tolerance, float time_limit, void* XsolPtr, int N, double* dual_obj, double* primal_obj, void* dualValuePtr,
	void* lbPtr, void* ubPtr, void* objCoefPtr,
	void* consLbPtr, void* consUbPtr, void* ConsMatrixPtr) {
	auto& Xsol = *static_cast<Eigen::MatrixXd*>(XsolPtr);
	auto& dual_value = *static_cast<std::vector<TriangleInequality>*>(dualValuePtr);
	auto& lb = *static_cast<std::vector<double>*>(lbPtr);
	auto& ub = *static_cast<std::vector<double>*>(ubPtr);
	auto& obj_coef = *static_cast<std::vector<double>*>(objCoefPtr);
	auto& cons_lb = *static_cast<std::vector<double>*>(consLbPtr);
	auto& cons_ub = *static_cast<std::vector<double>*>(consUbPtr);
	auto& ConsMatrix = *static_cast<Eigen::SparseMatrix<double, Eigen::ColMajor>*>(ConsMatrixPtr);


	int numConstr = cons_lb.size();


	pdlp::QuadraticProgram lp(lb.size(), cons_lb.size());
	lp.constraint_lower_bounds = Eigen::VectorXd::Map(cons_lb.data(), cons_lb.size());
	lp.constraint_upper_bounds = Eigen::VectorXd::Map(cons_ub.data(), cons_ub.size());
	lp.variable_lower_bounds = Eigen::VectorXd::Map(lb.data(), lb.size());
	lp.variable_upper_bounds = Eigen::VectorXd::Map(ub.data(), ub.size());
	lp.objective_vector = Eigen::VectorXd::Map(obj_coef.data(), obj_coef.size());
	lp.constraint_matrix = ConsMatrix;

	pdlp::PrimalDualHybridGradientParams params;
	params.mutable_termination_criteria()
		->mutable_simple_optimality_criteria()
		->set_eps_optimal_relative(tolerance);
	params.mutable_termination_criteria()
		->mutable_simple_optimality_criteria()
		->set_eps_optimal_absolute(tolerance);
	params.mutable_termination_criteria()->set_time_sec_limit(time_limit);
	//params.set_handle_some_primal_gradients_on_finite_bounds_as_residuals(false);
	params.set_num_shards(16);
	params.set_num_threads(4);
	//params.mutable_presolve_options()->set_use_glop(true);
	//params.set_verbosity_level(3);

	const pdlp::SolverResult result =
		pdlp::PrimalDualHybridGradient(lp, params);

	const pdlp::SolveLog& solve_log = result.solve_log;

	if (solve_log.termination_reason() == pdlp::TERMINATION_REASON_OPTIMAL) {
		int col_idx = 0;
		for (int i = 0; i < N; ++i) {
			for (int j = i; j < N; ++j) {
				double value = result.primal_solution(col_idx++);
				Xsol(i, j) = value;
				if (i != j) {
					Xsol(j, i) = value;  // For symmetric matrix
				}
			}
		}


		// print primal and dual objective value
		const pdlp::PointType solution_type = solve_log.solution_type();
		const std::optional<pdlp::ConvergenceInformation> ci =
			pdlp::GetConvergenceInformation(solve_log.solution_stats(),
				solution_type);
		if (ci.has_value()) {

			// now compute the corrected dual objective
			double corrected_dual_obj = result.dual_solution[0] * cons_ub[0];
			for (int i = 1; i < N + 1; i++) {
				corrected_dual_obj += result.dual_solution[i];
			}

			//auto r = result.reduced_costs;

			// according to ortools r = c-A^T*y
			// Eigen::VectorXd residual = -Eigen::VectorXd::Map(obj_coef.data(), obj_coef.size());
			// Eigen::VectorXd row_dual_org_vec = Eigen::VectorXd::Map(result.dual_solution.data(), cons_lb.size());
			// residual += ConsMatrix.transpose() * row_dual_org_vec;

			Eigen::VectorXd residual = ConsMatrix.transpose() * result.dual_solution - Eigen::VectorXd::Map(obj_coef.data(), obj_coef.size());


			// compare with pdlp's residual
			//std::cout << "violation: ||min(c-(A^T*y+Q^T*z), 0)||" << residual.cwiseMin(0).norm() << std::endl;
			//std::cout << "violation: ||max(c-(A^T*y+Q^T*z), 0)||" << residual.cwiseMax(0).norm() << std::endl;

			for (int i = 0; i < residual.size(); ++i) {
				if (residual[i] > 0) {
					corrected_dual_obj -= residual[i];
				}
			}


			//std::cout << std::left << std::setw(27) << "dual value sum:" << std::right << std::setw(15) << std::setprecision(8) << std::scientific << corrected_dual_obj << '\n';


			*dual_obj = corrected_dual_obj;
			*primal_obj = ci->primal_objective();

		}

		return 0;
	}
	else if (solve_log.termination_reason() == pdlp::TERMINATION_REASON_TIME_LIMIT) {
		// std::cout << "Time limit reached";
		// print Final solution stats
		auto primal_res = result.primal_solution;
		auto dual_res = result.dual_solution;
		pdlp::PrimalAndDualSolution initial_solution;
		initial_solution.primal_solution = primal_res;
		initial_solution.dual_solution = dual_res;

		pdlp::PrimalDualHybridGradientParams params_restart;
		params_restart.mutable_termination_criteria()
			->mutable_simple_optimality_criteria()
			->set_eps_optimal_relative(1e-3);
		params_restart.mutable_termination_criteria()
			->mutable_simple_optimality_criteria()
			->set_eps_optimal_absolute(1e-3);
		params_restart.mutable_termination_criteria()->set_time_sec_limit(time_limit);
		params_restart.set_num_shards(16);
		params_restart.set_num_threads(4);
		//params_restart.set_verbosity_level(3);

		const pdlp::SolverResult result_restart =
			pdlp::PrimalDualHybridGradient(lp, params_restart, initial_solution);
		const pdlp::SolveLog& solve_log_restart = result_restart.solve_log;


		if (solve_log_restart.termination_reason() == pdlp::TERMINATION_REASON_OPTIMAL) {
			std::cout << " and 1e-3 tolerance is reached\n";
		}
		else {
			std::cout << " and 1e-3 tolerance is not reached\n";
		}
		int col_idx = 0;
		for (int i = 0; i < N; ++i) {
			for (int j = i; j < N; ++j) {
				double value = result_restart.primal_solution(col_idx++);
				Xsol(i, j) = value;
				if (i != j) {
					Xsol(j, i) = value;  // For symmetric matrix
				}
			}
		}

		//int row_idx = cons_lb.size() - dual_value.size();
		//for (auto& item : dual_value) {
		//	item.dual_value = result_restart.dual_solution(row_idx);
		//	row_idx++;
		//}

		// print primal and dual objective value
		const pdlp::PointType solution_type_restart = solve_log_restart.solution_type();
		const std::optional<pdlp::ConvergenceInformation> ci_restart =
			pdlp::GetConvergenceInformation(solve_log_restart.solution_stats(),
				solution_type_restart);
		if (ci_restart.has_value()) {

			// now compute the corrected dual objective
			double corrected_dual_obj = result_restart.dual_solution[0] * cons_ub[0];
			for (int i = 1; i < N + 1; i++) {
				corrected_dual_obj += result_restart.dual_solution[i];
			}
			// compare with pdlp's residual
			//auto dual_res = result_restart.reduced_costs;


			Eigen::VectorXd residual = ConsMatrix.transpose() * result.dual_solution - Eigen::VectorXd::Map(obj_coef.data(), obj_coef.size());

			for (int i = 0; i < residual.size(); ++i) {
				if (residual[i] > 0) {
					corrected_dual_obj -= residual[i];
				}
			}


			//std::cout << std::left << std::setw(27) << "dual value sum:" << std::right << std::setw(15) << std::setprecision(8) << std::scientific << corrected_dual_obj << '\n';


			*dual_obj = corrected_dual_obj;
			*primal_obj = ci_restart->primal_objective();

		}

		return 1;

	}
	else {
		std::cout << "Solve not successful. Status: "
			<< pdlp::TerminationReason_Name(solve_log.termination_reason())
			<< '\n';
		return 2;
	}



	// result.primal_solution contains the solution to the primal problem
	// result.dual_solution contains the solution to the dual problem

	return 0;

}

extern "C" int gurobi_partial_lpk(float tolerance, float time_limit, void* XsolPtr, int N, double* dual_obj, double* primal_obj, void* dualValuePtr,
	void* lbPtr, void* ubPtr, void* objCoefPtr,
	void* consLbPtr, void* consUbPtr, void* ConsMatrixPtr) {
	try {
		// Create an environment and set WLS parameters
		GRBEnv env = GRBEnv(true);
		env.set(GRB_IntParam_OutputFlag, 0);
		env.set("WLSACCESSID", "257b1c4f-526d-40dc-a072-bbc50d5ffda8");
		env.set("WLSSECRET", "7b6bd99e-108c-4c55-9526-0b808993313f");
		env.set("LICENSEID", "2502162");

		env.start();


		// Create a model
		GRBModel model = GRBModel(env);
		model.set(GRB_IntParam_LogToConsole, 0);
		model.set(GRB_DoubleParam_TimeLimit, 3600);
		model.set(GRB_DoubleParam_BarConvTol, tolerance);
		model.set(GRB_IntParam_Method, 2);
		model.set(GRB_IntParam_Crossover, 0);

		auto& Xsol = *static_cast<Eigen::MatrixXd*>(XsolPtr);
		auto& dual_value = *static_cast<std::vector<TriangleInequality>*>(dualValuePtr);
		auto& lb = *static_cast<std::vector<double>*>(lbPtr);
		auto& ub = *static_cast<std::vector<double>*>(ubPtr);
		auto& obj_coef = *static_cast<std::vector<double>*>(objCoefPtr);
		auto& cons_lb = *static_cast<std::vector<double>*>(consLbPtr);
		auto& cons_ub = *static_cast<std::vector<double>*>(consUbPtr);
		auto& ConsMatrix = *static_cast<Eigen::SparseMatrix<double, Eigen::ColMajor>*>(ConsMatrixPtr);



		// Number of variables and constraints
		int num_vars = lb.size();
		int num_cons = cons_lb.size();

		// Create variables with bounds
		std::vector<GRBVar> vars(num_vars);
		for (int i = 0; i < num_vars; ++i) {
			vars[i] = model.addVar(lb[i], ub[i], obj_coef[i], GRB_CONTINUOUS);
		}

		std::vector<GRBConstr> constraints(num_cons);

		// Convert to row-major format for efficient row-wise access
		Eigen::SparseMatrix<double, Eigen::RowMajor> RowMajorMatrix(ConsMatrix);

		for (int row = 0; row < RowMajorMatrix.rows(); ++row) {
			GRBLinExpr expr = 0.0;

			for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(RowMajorMatrix, row); it; ++it) {
				int col = it.col();
				double value = it.value();

				if (col >= 0 && col < RowMajorMatrix.cols()) {
					expr += value * vars[col];
				}
				else {
					std::cerr << "Warning: Column index out of bounds: " << col << " at row " << row << std::endl;
				}
			}

			// Add constraint for this row
			if (row < N + 1) {
				constraints[row] = model.addConstr(expr == cons_ub[row]);
			}
			else {
				constraints[row] = model.addConstr(expr <= cons_ub[row]);
			}
		}
		// clear RowMajorMatrix
		RowMajorMatrix.resize(0, 0);

		// Optimize the model
		model.optimize();

		// Get the solution
		int col_idx = 0;
		for (int i = 0; i < N; ++i) {
			for (int j = i; j < N; ++j) {
				double value = vars[col_idx].get(GRB_DoubleAttr_X);
				Xsol(i, j) = value;
				if (i != j) {
					Xsol(j, i) = value;  // For symmetric matrix
				}
				col_idx++;
			}
		}

		// get the dual value as a VectorXd
		Eigen::VectorXd dual_values(num_cons);
		for (int i = 0; i < num_cons; ++i) {
			dual_values[i] = constraints[i].get(GRB_DoubleAttr_Pi);
		}

		// now compute the corrected dual objective
		double corrected_dual_obj = constraints[0].get(GRB_DoubleAttr_Pi) * cons_ub[0];
		for (int i = 1; i < N + 1; i++) {
			corrected_dual_obj += constraints[i].get(GRB_DoubleAttr_Pi);
		}

		Eigen::VectorXd residual = ConsMatrix.transpose() * dual_values - Eigen::VectorXd::Map(obj_coef.data(), obj_coef.size());
		for (int i = 0; i < residual.size(); ++i) {
			if (residual[i] > 0) {
				corrected_dual_obj -= residual[i];
			}
		}

		*dual_obj = corrected_dual_obj;
		*primal_obj = model.get(GRB_DoubleAttr_ObjVal);
	}
	catch (GRBException& e) {
		std::cout << "Error code = " << e.getErrorCode() << std::endl;
		std::cout << e.getMessage() << std::endl;
	}

	return 0;
}