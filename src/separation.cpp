#include "separation.h"
#include <limits>
#include <algorithm>
#include <omp.h>
#include <unordered_set>
#include <chrono>
#include <unordered_map>
#include <iostream>

std::string getInequalityKey(const std::vector<int>& ineq_idx) {
    std::stringstream ss;
    for (int idx : ineq_idx) {
        ss << idx << "_";
    }
    return ss.str();
}

void separation_scheme(
    const Eigen::MatrixXd& Xsol, 
    std::vector<std::list<validInequality>>& violated_cuts, 
    int max_T, 
    int N, 
    int maxSize, 
    double cuts_vio_tol
) {
	int max_list_size = 0;
#pragma omp parallel for shared(violated_cuts, max_list_size)
	for (int source = 0; source < N; ++source) {
		for (int j = 0; j < N; ++j) {
			if (j != source) {
				std::vector<int> chain = { j };
				int current_node = j;
				double current_cost = -Xsol(source, source) + Xsol(source, j);

				for (int size = 2; size <= max_T; ++size) {
					if (max_list_size >= maxSize) break; // Early exit check

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

					if (best_next_node != -1 && max_list_size < maxSize) {
						if (max_next_cost > cuts_vio_tol && potential_chain.size() > 1) {
#pragma omp critical
							{
								// create a list of int start with source and then potential_chain
                                std::vector<int> cut = potential_chain;
                                cut.insert(cut.begin(), source);
								violated_cuts[size - 2].push_back(validInequality(cut, max_next_cost));
#pragma omp atomic
								max_list_size++;
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

void extend_chain(
    int source, const Eigen::MatrixXd& Xsol, std::vector<int>& chain, double current_cost,
    const int N, const int max_T, double cuts_vio_tol,
    std::vector<std::list<validInequality>>& violated_cuts, int& max_list_size, const int max_init,
    const std::chrono::steady_clock::time_point& start_time, double time_limit_seconds,
    int current_depth, int search_depth 
) {
    // Check time limit first
    auto now = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = now - start_time;
    if (elapsed.count() >= time_limit_seconds) {
        return; // stop recursion if time limit exceeded
    }

    if (chain.size() >= max_T || max_list_size >= max_init) {
        return;
    }

    int last_node = chain.back();
    
    // If we're still within the search depth, use full enumeration
    if (current_depth < search_depth) {
        for (int next = last_node + 1; next < N; ++next) {
            if (next == source) continue;

            double additional_cost = 0.0;
            for (int k : chain) {
                if (k < next) {
                    additional_cost += Xsol(k, next);
                }
            }
            double next_cost = current_cost + Xsol(source, next) - additional_cost;

            if (next_cost > cuts_vio_tol) {
                #pragma omp critical
                {
                    if (max_list_size < max_init) {
                        std::vector<int> recording_chain(chain.begin(), chain.end());
                        recording_chain.push_back(next);
                        recording_chain.insert(recording_chain.begin(), source);
                        violated_cuts[recording_chain.size() - 3].push_back(validInequality(recording_chain, next_cost));
                        max_list_size++;
                    }
                }
            }

            chain.push_back(next);
            extend_chain(source, Xsol, chain, next_cost, N, max_T, cuts_vio_tol, 
                         violated_cuts, max_list_size, max_init, start_time, 
                         time_limit_seconds, current_depth + 1, search_depth);
            chain.pop_back();
        }
    }
    // Once we hit search depth, switch to greedy approach
    else {
        // Find best next node
        int best_next = -1;
        double best_cost = -std::numeric_limits<double>::infinity();
        
        for (int next = last_node + 1; next < N; ++next) {
            if (next == source || std::find(chain.begin(), chain.end(), next) != chain.end()) continue;
            
            double additional_cost = 0.0;
            for (int k : chain) {
                if (k < next) {
                    additional_cost += Xsol(k, next);
                }
            }
            double next_cost = current_cost + Xsol(source, next) - additional_cost;
            
            if (next_cost > best_cost) {
                best_cost = next_cost;
                best_next = next;
            }
        }
        
        // If we found a good next node with positive violation
        if (best_next != -1) {
            // Only add violated cut if best_cost > cuts_vio_tol
            if (best_cost > cuts_vio_tol) {
                #pragma omp critical
                {
                    if (max_list_size < max_init) {
                        std::vector<int> recording_chain(chain.begin(), chain.end());
                        recording_chain.push_back(best_next);
                        recording_chain.insert(recording_chain.begin(), source);
                        violated_cuts[recording_chain.size() - 3].push_back(validInequality(recording_chain, best_cost));
                        max_list_size++;
                    }
                }
            }

            // Always extend the chain to best_next
            chain.push_back(best_next);
            extend_chain(source, Xsol, chain, best_cost, N, max_T, cuts_vio_tol, 
                        violated_cuts, max_list_size, max_init, start_time, 
                        time_limit_seconds, current_depth + 1, search_depth);
            chain.pop_back();
        }
    }
}

void exact_separation_scheme(
    const Eigen::MatrixXd& Xsol, std::vector<std::list<validInequality>>& violated_cuts,
    int max_T, int N, int max_init, double cuts_vio_tol,
    double time_limit_seconds, 
    int search_depth   
) {
    // Validate search_depth to be between 1 and max_T
    search_depth = std::min(std::max(1, search_depth), max_T);
    
    int max_list_size = 0;
    std::vector<int> chain;
    chain.reserve(max_T);

    auto start_time = std::chrono::steady_clock::now(); // Start timer

    #pragma omp parallel for shared(violated_cuts, max_list_size) firstprivate(chain)
    for (int source = 0; source < N; ++source) {
        for (int j = 0; j < N; ++j) {
            if (j != source) {
                chain.clear();
                chain.push_back(j);
                double initial_cost = -Xsol(source, source) + Xsol(source, j);
                extend_chain(source, Xsol, chain, initial_cost, N, max_T, cuts_vio_tol, 
                            violated_cuts, max_list_size, max_init, start_time, 
                            time_limit_seconds, 1, search_depth);
            }
        }
    }
}

void extend_chain(
    int source, 
    const Eigen::MatrixXd& Xsol, 
    std::vector<int>& chain, 
    double current_cost, 
    const int N, 
    const int max_T, 
    double cuts_vio_tol, 
    std::vector<std::list<validInequality>>& violated_cuts, 
    std::atomic<int>& max_list_size, 
    const int max_init,
    std::vector<double>& node_contributions
) {
    // Base case checks
    if (chain.size() >= max_T || max_list_size.load(std::memory_order_relaxed) >= max_init) {
        return;
    }
    
    const int last_node = chain.back();
    
    // Reset and update node contributions for efficient cost calculation
    std::fill(node_contributions.begin(), node_contributions.end(), 0.0);
    for (int node_idx : chain) {
        for (int next = std::max(node_idx + 1, last_node + 1); next < N; ++next) {
            if (next != source) {
                node_contributions[next] += Xsol(node_idx, next);
            }
        }
    }
    
    // Try adding each possible next node
    for (int next = last_node + 1; next < N; ++next) {
        if (next == source) continue;
        
        // Use pre-computed node contributions
        double next_cost = current_cost + Xsol(source, next) - node_contributions[next];
        
        // If this is a violated inequality, add it to our cuts
        if (next_cost > cuts_vio_tol) {
            int local_max = max_list_size.load(std::memory_order_acquire);
            
            if (local_max < max_init) {
                std::vector<int> recording_chain;
                recording_chain.reserve(chain.size() + 2);
                recording_chain.push_back(source);
                recording_chain.insert(recording_chain.end(), chain.begin(), chain.end());
                recording_chain.push_back(next);
                
                #pragma omp critical
                {
                    if (max_list_size < max_init) {
                        violated_cuts[recording_chain.size() - 3].push_back(
                            validInequality(recording_chain, next_cost)
                        );
                        max_list_size.fetch_add(1, std::memory_order_release);
                    }
                }
            }
        }
        
        // Recursively extend the chain
        chain.push_back(next);
        extend_chain(source, Xsol, chain, next_cost, N, max_T, 
                     cuts_vio_tol, violated_cuts, max_list_size, max_init, node_contributions);
        chain.pop_back();
    }
}

void exact_separation_scheme(
    const Eigen::MatrixXd& Xsol, 
    std::vector<std::list<validInequality>>& violated_cuts, 
    int max_T, 
    int N, 
    int max_init, 
    double cuts_vio_tol
) {
    std::atomic<int> max_list_size{0};
    
    #pragma omp parallel
    {
        // Thread-local chain
        std::vector<int> chain;
        chain.reserve(max_T);
        
        // Pre-allocate node contributions array
        std::vector<double> node_contributions(N, 0.0);
        
        #pragma omp for schedule(dynamic, 4) nowait
        for (int source = 0; source < N; ++source) {
            // Periodically check global limit
            int local_max = max_list_size.load(std::memory_order_relaxed);
            if (local_max >= max_init) continue;
            
            for (int j = 0; j < N; ++j) {
                if (j == source) continue;
                
                // Check occasionally to reduce atomic operations
                if (j % 5 == 0) {
                    local_max = max_list_size.load(std::memory_order_relaxed);
                    if (local_max >= max_init) break;
                }
                
                chain.clear();
                chain.push_back(j);
                double initial_cost = -Xsol(source, source) + Xsol(source, j);
                
                // Use thread-local chain and node_contributions
                extend_chain(source, Xsol, chain, initial_cost, N, max_T, 
                             cuts_vio_tol, violated_cuts, max_list_size, max_init, node_contributions);
            }
        }
    }
}

int update_cuts(LPK& lp, const parameters& params, const Eigen::MatrixXd& Xsol, int& max_T, 
    int N, std::vector<char>& signs, std::vector<validInequality>& cutting_planes, int& violation_size, int& active_size){
        auto separation_start = std::chrono::high_resolution_clock::now();
        std::vector<std::list<validInequality>> violated_cuts(max_T - 1);
        violation_size = 0;

        while (true) {
            omp_set_num_threads(2);

            for (auto& cuts_list : violated_cuts) {
                cuts_list.clear();
            }
            violation_size = 0;

            separation_scheme(Xsol, violated_cuts, max_T, N, params.max_separation_size, params.cuts_vio_tol);

            for (int i = 0; i < max_T - 1; ++i) {
                violation_size += violated_cuts[i].size();
            }

            if (violation_size > 0) { 
                break;
            }

            if (max_T < params.t_upper_bound) {
                // Increase max_T if possible
                max_T++;
                signs.push_back('+');
                violated_cuts.resize(max_T - 1);
            } else {
                // do exact separation
                exact_separation_scheme(Xsol, violated_cuts, max_T, N, params.max_separation_size, params.cuts_vio_tol, params.max_separation_time, max_T);
                for (int i = 0; i < max_T - 1; ++i) {
                    violation_size += violated_cuts[i].size();
                }
                break;
            }
        }
        auto separation_end = std::chrono::high_resolution_clock::now();
        auto separation_time = std::chrono::duration_cast<std::chrono::milliseconds>(separation_end - separation_start);
        //std::cout<<"separation time: "<< separation_time.count()/1e3<<std::endl;

        if (violation_size == 0) {
            return 1; // No violated cuts found
        }

        cut_selection_active(N, max_T, Xsol, cutting_planes, lp, params.cuts_act_tol);

        active_size = cutting_planes.size();
        int cuts_idx = lp.cons_lb_cuts.size();
        if (active_size != cuts_idx) {
            std::cout << "Warning: Active cuts size (" << active_size 
                      << ") does not match cuts index (" << cuts_idx << ")." << std::endl;
                      return -1;
        }
        int cuts_idx_start = lp.cons_lb_basic.size();

        int remaining_capacity = std::max(params.max_cuts_per_iter - cuts_idx, 0);// add as many as possible but control the size of LP
        remaining_capacity = std::min(remaining_capacity, params.max_cuts_added_iter);
        if (remaining_capacity == 0) {
            remaining_capacity += 100000;
        }

        std::vector<Eigen::Triplet<int>> violated_cuts_triplets;
        double maximum_violation = -1.0;
        if (violation_size <= remaining_capacity) {
            cutting_planes.reserve(cutting_planes.size() + violation_size);
            violated_cuts_triplets.reserve(violation_size * (max_T + 2)); // Estimate max size


            for (int i = 0; i < violated_cuts.size(); i++) {
                for (const auto& element : violated_cuts[i]) {
                    cutting_planes.push_back(element);
                    if (element.violation > maximum_violation){
                      maximum_violation = element.violation;
                    } 
                    std::string key = getInequalityKey(element.ineq_idx);

                    int newRow = cuts_idx_start + cuts_idx;
                    auto it = element.ineq_idx.begin();
                    int firstElement = *it;
                    int firstTerm = firstElement * (2 * N - firstElement + 1) / 2;
                    violated_cuts_triplets.emplace_back(newRow, firstTerm, -1);

                    for (auto j = std::next(it); j != element.ineq_idx.end(); ++j) {
                        int currentJ = *j;
                        int temp_i = std::min(firstElement, currentJ);
                        int temp_j = std::max(firstElement, currentJ);
                        int indexValue = temp_i * (2 * N - temp_i + 1) / 2 + temp_j - temp_i;
                        violated_cuts_triplets.emplace_back(newRow, indexValue, 1);

                        for (auto k = std::next(j); k != element.ineq_idx.end(); ++k) {
                            int currentK = *k;
                            violated_cuts_triplets.emplace_back(newRow, currentJ * (2 * N - currentJ + 1) / 2 + currentK - currentJ, -1);
                        }
                    }
                    ++cuts_idx;
                }
            }
        }
        else {
            cutting_planes.reserve(cutting_planes.size() + remaining_capacity);
            violated_cuts_triplets.reserve(remaining_capacity * (max_T + 2)); // Estimate max size

            std::vector<validInequality> violated_cuts_sorted;
            violated_cuts_sorted.reserve(violation_size);
            for (int i = 0; i < violated_cuts.size(); i++) {
                for (const auto& element : violated_cuts[i]) {
                    violated_cuts_sorted.push_back(element);
                }
            }
            // sorting based on violation
            std::sort(violated_cuts_sorted.begin(), violated_cuts_sorted.end(), [](const validInequality& a, const validInequality& b) {
                return a.violation > b.violation;
                });
            maximum_violation =  violated_cuts_sorted[0].violation;

            for (int i = 0; i < remaining_capacity; i++) {
                cutting_planes.push_back(violated_cuts_sorted[i]);

                std::string key = getInequalityKey(violated_cuts_sorted[i].ineq_idx);

                int newRow = cuts_idx_start + cuts_idx;
                auto it = violated_cuts_sorted[i].ineq_idx.begin();
                int firstElement = *it;
                int firstTerm = firstElement * (2 * N - firstElement + 1) / 2;
                violated_cuts_triplets.emplace_back(newRow, firstTerm, -1);

                for (auto j = std::next(it); j != violated_cuts_sorted[i].ineq_idx.end(); ++j) {
                    int currentJ = *j;
                    int temp_i = std::min(firstElement, currentJ);
                    int temp_j = std::max(firstElement, currentJ);
                    int indexValue = temp_i * (2 * N - temp_i + 1) / 2 + temp_j - temp_i;
                    violated_cuts_triplets.emplace_back(newRow, indexValue, 1);

                    for (auto k = std::next(j); k != violated_cuts_sorted[i].ineq_idx.end(); ++k) {
                        int currentK = *k;
                        violated_cuts_triplets.emplace_back(newRow, currentJ * (2 * N - currentJ + 1) / 2 + currentK - currentJ, -1);
                    }
                }
                ++cuts_idx;
            }
        }
        violated_cuts_triplets.shrink_to_fit();

        // append violated_cuts_triplets to lp.triplets_cuts
        lp.triplets_cuts.insert(lp.triplets_cuts.end(), violated_cuts_triplets.begin(), violated_cuts_triplets.end());
        lp.cons_lb_cuts.resize(cutting_planes.size(), -kInfinity);
        lp.cons_ub_cuts.resize(cutting_planes.size(), 0.0);

        return 0;
}



void cut_selection_active(const int N, const int max_T, const Eigen::MatrixXd& Xsol, 
    std::vector<validInequality>& cutting_planes, LPK& lp, double tolerance){
    
        auto cut_selection_start = std::chrono::high_resolution_clock::now();
    
        // Free previous triplets and allocate new ones
        std::vector<Eigen::Triplet<int>> kept_cuts_triplets;
        kept_cuts_triplets.reserve(cutting_planes.size() * (max_T + 2)); // Estimate max size
        
        int cuts_idx = 0;
        int active_idx = 0;
        int cuts_idx_start = lp.cons_lb_basic.size() + lp.cons_lb_cuts.size();
        
        // Process all cuts, keeping active ones at the front of the vector
        for (size_t i = 0; i < cutting_planes.size(); ++i) {
            const auto& element = cutting_planes[i];
            std::string key = getInequalityKey(element.ineq_idx);
            
            if (std::abs(element.violation) < tolerance) {// || inequality_add_counts[key] >= 2
                // If we're not already at this position, move the active cut to the front
                if (active_idx != i) {
                    cutting_planes[active_idx] = element;
                }
                
                // Create triplets for this cut
                int newRow = cuts_idx_start + cuts_idx;
                
                // Now using vector instead of list, so we can access elements directly
                int firstElement = element.ineq_idx[0];
                int firstTerm = firstElement * (2 * N - firstElement + 1) / 2;
                kept_cuts_triplets.emplace_back(newRow, firstTerm, -1);
                
                // Iterate through the remaining elements
                for (size_t j = 1; j < element.ineq_idx.size(); ++j) {
                    int currentJ = element.ineq_idx[j];
                    int temp_i = std::min(firstElement, currentJ);
                    int temp_j = std::max(firstElement, currentJ);
                    int indexValue = temp_i * (2 * N - temp_i + 1) / 2 + temp_j - temp_i;
                    kept_cuts_triplets.emplace_back(newRow, indexValue, 1);
                    
                    // Generate triplets for pairs of non-first elements
                    for (size_t k = j + 1; k < element.ineq_idx.size(); ++k) {
                        int currentK = element.ineq_idx[k];
                        kept_cuts_triplets.emplace_back(newRow, currentJ * (2 * N - currentJ + 1) / 2 + currentK - currentJ, -1);
                    }
                }
                
                ++active_idx;
                ++cuts_idx;
            }
        }
        
        // Resize the cutting_planes vector to contain only active cuts
        cutting_planes.resize(active_idx);
        cutting_planes.shrink_to_fit();
        kept_cuts_triplets.shrink_to_fit();

        lp.triplets_cuts = kept_cuts_triplets;
        lp.cons_lb_cuts.insert(lp.cons_lb_cuts.end(), cutting_planes.size(), -kInfinity);
        lp.cons_ub_cuts.insert(lp.cons_ub_cuts.end(), cutting_planes.size(), 0.0);
        
        auto act_cut_selection_end = std::chrono::high_resolution_clock::now();
        auto act_cut_time = std::chrono::duration_cast<std::chrono::milliseconds>(act_cut_selection_end - cut_selection_start);
        //std::cout << "active cut selection time: " << act_cut_time.count() / 1e3 << " seconds" << std::endl;
    }