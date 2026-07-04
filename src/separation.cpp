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
    double cuts_vio_tol,
    double time_limit_seconds
) {
	int max_list_size = 0;
    auto start_time = std::chrono::steady_clock::now();
#pragma omp parallel for shared(violated_cuts, max_list_size, start_time, time_limit_seconds)
	for (int source = 0; source < N; ++source) {
        // Check time limit at source level — skip remaining sources if time expired
        {
            auto now = std::chrono::steady_clock::now();
            std::chrono::duration<double> elapsed = now - start_time;
            if (elapsed.count() >= time_limit_seconds) continue;
        }
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

            //separation_scheme_top_k(Xsol, violated_cuts, max_T, N, params.cutting_plane_max_cuts_separation_size, params.cutting_plane_cuts_vio_tol, 2);
            separation_scheme(Xsol, violated_cuts, max_T, N, params.cutting_plane_max_cuts_separation_size, params.cutting_plane_cuts_vio_tol, params.cutting_plane_max_separation_time);

            for (int i = 0; i < max_T - 1; ++i) {
                violation_size += violated_cuts[i].size();
            }

            if (violation_size > 0) { 
                break;
            }

            if (max_T < params.cutting_plane_t_upper_bound) {
                // Increase max_T if possible
                max_T++;
                signs.push_back('+');
                violated_cuts.resize(max_T - 1);
            } else if(params.cutting_plane_exact_separation) {
                // do exact separation
                // std::cout<<"No violated cuts found with current max_T = "<<max_T<<". start search top K."<<std::endl;
                separation_scheme_top_k(Xsol, violated_cuts, max_T, N, params.cutting_plane_max_cuts_separation_size, params.cutting_plane_cuts_vio_tol, max_T + 1, params.cutting_plane_max_separation_time);
                //exact_separation_scheme(Xsol, violated_cuts, max_T, N, params.cutting_plane_max_cuts_separation_size, params.cutting_plane_cuts_vio_tol, params.cutting_plane_max_separation_time, max_T);
                for (int i = 0; i < max_T - 1; ++i) {
                    violation_size += violated_cuts[i].size();
                }
                break;
            }
            else{
                break;
            }
        }
        auto separation_end = std::chrono::high_resolution_clock::now();
        auto separation_time = std::chrono::duration_cast<std::chrono::milliseconds>(separation_end - separation_start);
        //std::cout<<"separation time: "<< separation_time.count()/1e3<<std::endl;

        if (violation_size == 0) {
            return 1; // No violated cuts found
        }

        cut_selection_active(N, max_T, Xsol, cutting_planes, lp, params.cutting_plane_cuts_act_tol, params.cutting_plane_remove_inactive_cuts);

        active_size = cutting_planes.size();
        int cuts_idx = lp.cons_lb_cuts.size();
        if (active_size != cuts_idx) {
            std::cout << "Warning: Active cuts size (" << active_size 
                      << ") does not match cuts index (" << cuts_idx << ")." << std::endl;
                      return -1;
        }
        int cuts_idx_start = lp.cons_lb_basic.size() + lp.cons_lb_branch.size();

        int remaining_capacity = std::max(params.cutting_plane_max_cuts_per_iter - cuts_idx, 0);// add as many as possible but control the size of LP
        remaining_capacity = std::min(remaining_capacity, params.cutting_plane_max_cuts_added_iter);
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
                        int indexValue = getPairIndex(firstElement, currentJ, N);
                        violated_cuts_triplets.emplace_back(newRow, indexValue, 1);

                        for (auto k = std::next(j); k != element.ineq_idx.end(); ++k) {
                            int currentK = *k;
                            violated_cuts_triplets.emplace_back(newRow, getPairIndex(currentJ, currentK, N), -1);
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
                int firstTerm = getPairIndex(firstElement, firstElement, N);
                violated_cuts_triplets.emplace_back(newRow, firstTerm, -1);

                for (auto j = std::next(it); j != violated_cuts_sorted[i].ineq_idx.end(); ++j) {
                    int currentJ = *j;
                    int indexValue = getPairIndex(firstElement, currentJ, N);
                    violated_cuts_triplets.emplace_back(newRow, indexValue, 1);

                    for (auto k = std::next(j); k != violated_cuts_sorted[i].ineq_idx.end(); ++k) {
                        int currentK = *k;
                        violated_cuts_triplets.emplace_back(newRow, getPairIndex(currentJ, currentK, N), -1);
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
    std::vector<validInequality>& cutting_planes, LPK& lp, double tolerance, bool remove_inactive) {
    
        auto cut_selection_start = std::chrono::high_resolution_clock::now();
    
        // Free previous triplets and allocate new ones
        std::vector<Eigen::Triplet<int>> kept_cuts_triplets;
        kept_cuts_triplets.reserve(cutting_planes.size() * (max_T + 2)); // Estimate max size
        
        int cuts_idx = 0;
        int active_idx = 0;
        int cuts_idx_start = lp.cons_lb_basic.size() + lp.cons_lb_branch.size() + lp.cons_lb_cuts.size();
        
        // Process all cuts, keeping active ones at the front of the vector
        for (size_t i = 0; i < cutting_planes.size(); ++i) {
            const auto& element = cutting_planes[i];
            std::string key = getInequalityKey(element.ineq_idx);

            if (std::abs(element.violation) < tolerance || (!remove_inactive)) {// || inequality_add_counts[key] >= 2
                // If we're not already at this position, move the active cut to the front
                if (active_idx != i) {
                    cutting_planes[active_idx] = element;
                }
                
                // Create triplets for this cut
                int newRow = cuts_idx_start + cuts_idx;
                
                // Now using vector instead of list, so we can access elements directly
                int firstElement = element.ineq_idx[0];
                int firstTerm = getPairIndex(firstElement, firstElement, N);
                kept_cuts_triplets.emplace_back(newRow, firstTerm, -1);
                
                // Iterate through the remaining elements
                for (size_t j = 1; j < element.ineq_idx.size(); ++j) {
                    int currentJ = element.ineq_idx[j];
                    int indexValue = getPairIndex(firstElement, currentJ, N);
                    kept_cuts_triplets.emplace_back(newRow, indexValue, 1);
                    
                    // Generate triplets for pairs of non-first elements
                    for (size_t k = j + 1; k < element.ineq_idx.size(); ++k) {
                        int currentK = element.ineq_idx[k];
                        kept_cuts_triplets.emplace_back(newRow, getPairIndex(currentJ, currentK, N), -1);
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

void separation_scheme_top_k(
    const Eigen::MatrixXd& Xsol, 
    std::vector<std::list<validInequality>>& violated_cuts, 
    int max_T, 
    int N, 
    int maxSize, 
    double cuts_vio_tol,
    int k_branching,  // Number of top nodes to consider at each step
    double time_limit_seconds
) {
    int max_list_size = 0;
    auto start_time = std::chrono::steady_clock::now();
    
    // Structure to hold potential extensions with their costs
    struct NodeExtension {
        int node;
        double cost;
        std::vector<int> chain;
        
        bool operator<(const NodeExtension& other) const {
            return cost > other.cost; // For max-heap (highest cost first)
        }
    };

    // use up to max available threads minus 2 threads
    int available_threads = omp_get_max_threads();
    int threads_to_use = std::max(1, available_threads - 2);
    omp_set_num_threads(threads_to_use);

#pragma omp parallel for shared(violated_cuts, max_list_size, start_time, time_limit_seconds)
    for (int source = 0; source < N; ++source) {
        // Check time limit at source level — skip remaining sources if time expired
        {
            auto now = std::chrono::steady_clock::now();
            std::chrono::duration<double> elapsed = now - start_time;
            if (elapsed.count() >= time_limit_seconds) continue;
        }
        for (int j = 0; j < N; ++j) {
            if (j != source) {
                // Initialize with single node chain
                std::vector<NodeExtension> current_level;
                current_level.push_back({j, -Xsol(source, source) + Xsol(source, j), {j}});
                
                for (int size = 2; size <= max_T; ++size) {
                    if (max_list_size >= maxSize) break;
                    
                    std::vector<NodeExtension> next_level;
                    
                    // For each chain in current level
                    for (const auto& current : current_level) {
                        std::vector<std::pair<int, double>> candidates;
                        
                        // Find all possible extensions
                        int current_node = current.chain.back();
                        for (int next = current_node + 1; next < N; ++next) {
                            if (next != source) {
                                double additional_cost = 0.0;
                                for (int k : current.chain) {
                                    if (k < next) {
                                        additional_cost += Xsol(k, next);
                                    }
                                }
                                double next_cost = current.cost + Xsol(source, next) - additional_cost;
                                candidates.push_back({next, next_cost});
                            }
                        }
                        
                        // Sort candidates by cost (descending) and take top k
                        std::sort(candidates.begin(), candidates.end(), 
                                [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                                    return a.second > b.second;
                                });
                        
                        int num_to_take = std::min(k_branching, (int)candidates.size());
                        for (int i = 0; i < num_to_take; ++i) {
                            int next_node = candidates[i].first;
                            double next_cost = candidates[i].second;
                            
                            // Create extended chain
                            std::vector<int> extended_chain = current.chain;
                            extended_chain.push_back(next_node);
                            
                            // Add to next level
                            next_level.push_back({next_node, next_cost, extended_chain});
                            
                            // Check if this is a violated cut
                            if (next_cost > cuts_vio_tol && extended_chain.size() > 1) {
#pragma omp critical
                                {
                                    if (max_list_size < maxSize) {
                                        std::vector<int> cut = extended_chain;
                                        cut.insert(cut.begin(), source);
                                        violated_cuts[size - 2].push_back(validInequality(cut, next_cost));
#pragma omp atomic
                                        max_list_size++;
                                    }
                                }
#pragma omp flush(max_list_size)
                            }
                        }
                    }
                    
                    // Move to next level, but limit the number of chains to keep memory manageable
                    current_level = std::move(next_level);
                    
                    // limit the number of chains per level to control memory
                    if (current_level.size() > maxSize / 10) {
                        std::sort(current_level.begin(), current_level.end());
                        current_level.resize(maxSize / 10);
                    }
                    
                    if (current_level.empty()) break;
                }
            }
        }
    }
}