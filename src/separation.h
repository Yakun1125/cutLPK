#pragma once
#include <vector>
#include <list>
#include <vector>
#include "Utils_Struct.h"
#include <chrono>

// Per-(source,j) greedy chain-walk state, persisted across successive separation_scheme
// calls within a single update_cuts() ladder (i.e. while max_T is incremented but Xsol is
// unchanged) so each call can resume extending from where the previous one left off instead
// of re-walking chains from scratch.
struct ChainWalkState {
    std::vector<int> chain;      // greedy chain built so far, excluding `source` (chain[0] == j)
    int current_node = -1;
    double current_cost = 0.0;
    bool initialized = false;    // false => must be seeded fresh (chain = {j})
    bool exhausted = false;      // true => no further extension exists; deterministic, skip on resume
};

void separation_scheme(
    const Eigen::MatrixXd& Xsol,
    std::vector<std::list<validInequality>>& violated_cuts,
    int max_T,
    int N,
    int maxSize,
    double cuts_vio_tol,
    double time_limit_seconds,
    std::vector<ChainWalkState>& walk_states
);

void separation_scheme_top_k(
    const Eigen::MatrixXd& Xsol, 
    std::vector<std::list<validInequality>>& violated_cuts, 
    int max_T, 
    int N, 
    int maxSize, 
    double cuts_vio_tol,
    int k_branching,
    double time_limit_seconds
);


void extend_chain(
    int source, const Eigen::MatrixXd& Xsol, std::vector<int>& chain, double current_cost,
    const int N, const int max_T, double cuts_vio_tol,
    std::vector<std::list<validInequality>>& violated_cuts, int& max_list_size, const int max_init,
    const std::chrono::steady_clock::time_point& start_time, double time_limit_seconds,
    int current_depth, int search_depth 
);


void exact_separation_scheme(
    const Eigen::MatrixXd& Xsol, std::vector<std::list<validInequality>>& violated_cuts,
    int max_T, int N, int max_init, double cuts_vio_tol,
    double time_limit_seconds, // Time limit parameter 
    int search_depth  
) ;


void cut_selection_active(const int N, const int max_T, const Eigen::MatrixXd& Xsol, 
    std::vector<validInequality>& cutting_planes,  LPK& lp, double tolerance, bool remove_inactive_cuts);


int update_cuts(LPK& lp, const parameters& params, const Eigen::MatrixXd& Xsol, int& max_T, 
    int N, std::vector<char>& signs, std::vector<validInequality>& cutting_planes, int& violation_size, int& active_size);

// Test function to compare separation schemes
void test_separation_schemes(const Eigen::MatrixXd& Xsol, int max_T, int N, int maxSize, double cuts_vio_tol);