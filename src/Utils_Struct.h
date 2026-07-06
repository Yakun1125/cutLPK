#pragma once
#include <vector>
#include <list>
#include <string>
#include <limits>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
#include <Eigen/StdVector>
using VectorXdList = std::vector<Eigen::VectorXd, Eigen::aligned_allocator<Eigen::VectorXd>>;

#include <iostream>
#include <iomanip>

#include <queue>
#include <memory>

constexpr double kInfinity = std::numeric_limits<double>::infinity();

struct validInequality {
	std::vector<int> ineq_idx;// index start with i and then subset S
	double violation;
	double dual_value;
    // Default constructor
    validInequality() : violation(kInfinity), dual_value(0.0) {}

    // Parameterized constructor
    validInequality(std::vector<int> ineq_idx, double violation = kInfinity)
        : ineq_idx(std::move(ineq_idx)), violation(violation), dual_value(0.0) {}
};

// struct LPK {
// 	 int N; 
// 	std::vector<double> varLb; std::vector<double> varUb; std::vector<double> objCoef;
// 	std::vector<double> consLb; std::vector<double> consUb; Eigen::SparseMatrix<double, Eigen::ColMajor> ConsMatrix;
// 	std::vector<double> cons_lb_basic; std::vector<double> cons_ub_basic;
// 	std::vector<double> cons_lb_cuts; std::vector<double> cons_ub_cuts;
// 	std::vector<Eigen::Triplet<int>> triplets_basic;
// 	std::vector<Eigen::Triplet<int>> triplets_cuts;
// };

class LPK{
	public:
		 int N; 
	std::vector<double> varLb; std::vector<double> varUb; std::vector<double> objCoef;
	std::vector<double> consLb; std::vector<double> consUb; Eigen::SparseMatrix<double, Eigen::ColMajor> ConsMatrix;
    std::vector<double> cons_lb_basic; std::vector<double> cons_ub_basic;
	std::vector<double> cons_lb_cuts; std::vector<double> cons_ub_cuts;
	std::vector<Eigen::Triplet<int>> triplets_basic;
	std::vector<Eigen::Triplet<int>> triplets_cuts;
	std::vector<double> cons_lb_branch; std::vector<double> cons_ub_branch;
	std::vector<Eigen::Triplet<int>> triplets_branch;

	//methods to combine sparsematrix and cons bounds
	void setupLPK(bool preserve_cuts = false){
		std::vector<Eigen::Triplet<int>> combinedTriplets;
		combinedTriplets.reserve(triplets_basic.size() + triplets_branch.size() + triplets_cuts.size());
		combinedTriplets.insert(combinedTriplets.end(), triplets_basic.begin(), triplets_basic.end());
		combinedTriplets.insert(combinedTriplets.end(), triplets_branch.begin(), triplets_branch.end());
		combinedTriplets.insert(combinedTriplets.end(), triplets_cuts.begin(), triplets_cuts.end());
		ConsMatrix.resize(cons_lb_basic.size() + cons_lb_branch.size() + cons_lb_cuts.size(), N * (N + 1) / 2);
		ConsMatrix.setFromTriplets(combinedTriplets.begin(), combinedTriplets.end());
		if (!preserve_cuts) {
			triplets_cuts.clear();
		}

		consLb.assign(cons_lb_basic.begin(), cons_lb_basic.end());
		consLb.insert(consLb.end(), cons_lb_branch.begin(), cons_lb_branch.end());
		consLb.insert(consLb.end(), cons_lb_cuts.begin(), cons_lb_cuts.end());

		// Combine upper bounds
		consUb.assign(cons_ub_basic.begin(), cons_ub_basic.end());
		consUb.insert(consUb.end(), cons_ub_branch.begin(), cons_ub_branch.end());
		consUb.insert(consUb.end(), cons_ub_cuts.begin(), cons_ub_cuts.end());

		if (!preserve_cuts) {
			cons_lb_cuts.clear();
			cons_ub_cuts.clear();
		}
	};

    void printDebugInfo() const {
        std::cout << "\n--- LPK Debug Info ---" << std::endl;
        std::cout << std::left << std::setw(25) << "N:" << N << std::endl;
        
        std::cout << "\n--- Variable Vectors ---" << std::endl;
        std::cout << std::left << std::setw(25) << "varLb.size():" << varLb.size() << std::endl;
        std::cout << std::left << std::setw(25) << "varUb.size():" << varUb.size() << std::endl;
        std::cout << std::left << std::setw(25) << "objCoef.size():" << objCoef.size() << std::endl;
        
        std::cout << "\n--- Final Constraint Vectors ---" << std::endl;
        std::cout << std::left << std::setw(25) << "consLb.size():" << consLb.size() << std::endl;
        std::cout << std::left << std::setw(25) << "consUb.size():" << consUb.size() << std::endl;
        
        std::cout << "\n--- Basic Constraint Parts ---" << std::endl;
        std::cout << std::left << std::setw(25) << "cons_lb_basic.size():" << cons_lb_basic.size() << std::endl;
        std::cout << std::left << std::setw(25) << "cons_ub_basic.size():" << cons_ub_basic.size() << std::endl;
        std::cout << std::left << std::setw(25) << "triplets_basic.size():" << triplets_basic.size() << std::endl;

        std::cout << "\n--- Cut Constraint Parts ---" << std::endl;
        std::cout << std::left << std::setw(25) << "cons_lb_cuts.size():" << cons_lb_cuts.size() << std::endl;
        std::cout << std::left << std::setw(25) << "cons_ub_cuts.size():" << cons_ub_cuts.size() << std::endl;
        std::cout << std::left << std::setw(25) << "triplets_cuts.size():" << triplets_cuts.size() << std::endl;

        std::cout << "\n--- Final Constraint Matrix ---" << std::endl;
        std::cout << std::left << std::setw(25) << "ConsMatrix.rows():" << ConsMatrix.rows() << std::endl;
        std::cout << std::left << std::setw(25) << "ConsMatrix.cols():" << ConsMatrix.cols() << std::endl;
        std::cout << std::left << std::setw(25) << "ConsMatrix.nonZeros():" << ConsMatrix.nonZeros() << std::endl;
        std::cout << "------------------------\n" << std::endl;
    }

};

struct parameters {
	int random_seed;
	std::string solver;
	bool solver_warm_start;

	std::string cutting_plane_output_file;
	bool cutting_plane_exact_separation;
    bool cutting_plane_remove_inactive_cuts;
	int cutting_plane_verbose;
	int cutting_plane_output_level;
	int cutting_plane_max_cuts_firstLP;
	int cutting_plane_max_cuts_per_iter;
	int cutting_plane_max_cuts_added_iter;
	int cutting_plane_max_cuts_separation_size;
	int cutting_plane_max_active_cuts_size;
	int cutting_plane_warm_start;
	int cutting_plane_max_iter;
	int cutting_plane_t_upper_bound;
	int cutting_plane_num_iter_no_improve;
	double cutting_plane_firstLP_time_limit;
	double cutting_plane_LP_time_limit;
	double cutting_plane_time_limit;
	double cutting_plane_firstLP_solver_tol;
	double cutting_plane_solver_tol;
	double cutting_plane_lb_solver_tol;
	double cutting_plane_cuts_vio_tol;
	double cutting_plane_cuts_act_tol;
	double cutting_plane_opt_gap;
	double cutting_plane_max_separation_time;

	int bnb_node_limit;
	double bnb_time_limit;
	double bnb_gap_tol;
	double bnb_global_ub;
	int bnb_cut_iter_limit;
	std::string bnb_output_file;
	int bnb_verbose;
	int bnb_output_level;

    int lloyd_num_random_starts;
	std::string fair_clustering_fairness_type;
	double fair_clustering_fairness_param;
	std::string fair_clustering_group_file;
	std::string fair_assignment_solver;  // "highs" (default) or "gurobi"
	bool is_spectral_clustering;
    bool heuristic_only;
	    // Constructor with default values
		parameters() : 
		random_seed(42),
        solver("cupdlpx"),
		solver_warm_start(true),

		cutting_plane_exact_separation(true),
        cutting_plane_remove_inactive_cuts(true),
        cutting_plane_output_file(""),
		cutting_plane_verbose(1),
        cutting_plane_output_level(3),
        cutting_plane_max_cuts_firstLP(1.5e7),
		cutting_plane_max_cuts_per_iter(1e8),
		cutting_plane_max_cuts_added_iter(1e7),
		cutting_plane_max_cuts_separation_size(1.5e7),
		cutting_plane_max_active_cuts_size(1e8),
		cutting_plane_warm_start(1),
		cutting_plane_max_iter(3000),
		cutting_plane_t_upper_bound(2),
		cutting_plane_num_iter_no_improve(2),
		cutting_plane_firstLP_time_limit(360.0),
		cutting_plane_LP_time_limit(180.0),
		cutting_plane_time_limit(7200.0),
		cutting_plane_firstLP_solver_tol(1e-6),
		cutting_plane_solver_tol(1e-6),
		cutting_plane_lb_solver_tol(1e-6),
		cutting_plane_cuts_vio_tol(1e-4),
		cutting_plane_cuts_act_tol(1e-4),
		cutting_plane_opt_gap(1e-4),
		cutting_plane_max_separation_time(300.0),

		bnb_node_limit(0), // by default, no branch and bound will be executed
		bnb_time_limit(3600.0),
		bnb_gap_tol(1e-4),
		bnb_global_ub(kInfinity),
		bnb_cut_iter_limit(30),
		bnb_output_file(""),
		bnb_verbose(1),
		bnb_output_level(2),

		lloyd_num_random_starts(100),
		fair_clustering_fairness_type(""),
		fair_clustering_fairness_param(1.0),
		fair_clustering_group_file(""),
		fair_assignment_solver("highs"),
        is_spectral_clustering(false),
        heuristic_only(false)
    {}
};

// Return status for iterative cutting plane solver
enum class ICPStatus : int {
    SUCCESS = 0,  
    NO_VIOLATED_CUTS = 1,
    NO_IMPROVEMENT = 2,
    TIME_OR_LIMIT = 3,
    ERROR = 4,
    MAX_ITER = 5,
    INFEASIBLE = 6,
    HEURISTIC_ONLY = 7
};

struct cutLPKSolveInfo {
	double total_solver_time;
	double total_post_heuristic_time;
	double total_separation_time;
	double lower_bound;
	double upper_bound;
	double optimality_gap;
	int retcode; // 0 optimal; 1 no violated cuts; 2 no improvement; 3 time or iter limit reached; 4 error;
	Eigen::MatrixXd best_upper_bound_solution;
    Eigen::MatrixXd final_lp_Xsol;
    // Primal and dual solutions for warm starting
    std::vector<double> primal_solution;
    std::vector<double> dual_solution;
};

enum class BnBStatus : int {
	OPTIMAL = 0,        // Found optimal integer solution
	NODE_LIMIT = 1,     // Node is infeasible
	TIME_LIMIT = 2,     // Time limit reached
	ALL_NODES_EXPLORED = 3, // All nodes explored without finding better solution
	ERROR = 4           // Error during processing
};

enum class BranchType {
    SAME_CLUSTER,    // X(i,j) = X(i,i) = X(j,j) (i and j must be in same cluster)
    DIFF_CLUSTER     // X(i,j) = 0 (i and j must be in different clusters)
};

struct BranchConstraint {
    int i;  // First point index
    int j;  // Second point index (j > i)
    BranchType type;
    
    BranchConstraint(int i_, int j_, BranchType type_) 
        : i(i_), j(j_), type(type_) {}
};

struct BBNode {
    int node_id;
    int parent_id;
    int depth;
    
    // Branching constraints accumulated from root to this node
    std::vector<BranchConstraint> branch_constraints;
    
    // Bounds at this node
    double lower_bound;
    double upper_bound;
    
    // LP solution at this node
    Eigen::MatrixXd lp_solution;
    
    // Primal and dual solutions for warm starting
    std::vector<double> primal_solution;
    std::vector<double> dual_solution;
    
    // Status flags
    bool is_explored;
    bool is_pruned;
    bool is_integer;
    
    // Constructor
    BBNode(int id, int parent, int d) 
        : node_id(id), parent_id(parent), depth(d),
          lower_bound(-kInfinity), upper_bound(kInfinity),
          is_explored(false), is_pruned(false), is_integer(false) {}
};

struct BBNodeComparator {
    bool operator()(const std::shared_ptr<BBNode>& a, const std::shared_ptr<BBNode>& b) const {
        // Lower bound is better when it's higher
        return a->lower_bound > b->lower_bound;  // min-heap: explore nodes with smallest lower bound first
    }
};

struct BBTree {
    std::priority_queue<std::shared_ptr<BBNode>, 
                       std::vector<std::shared_ptr<BBNode>>, 
                       BBNodeComparator> open_nodes;
    
    std::vector<std::shared_ptr<BBNode>> all_nodes;
    
    int next_node_id;
    double global_lower_bound;
    double global_upper_bound;
    Eigen::MatrixXd best_solution;
    
    // Statistics
    int nodes_explored;
    int nodes_pruned;
    double total_time;
    
    // Constructor
    BBTree() : next_node_id(0), global_lower_bound(-kInfinity), 
               global_upper_bound(kInfinity), nodes_explored(0), 
               nodes_pruned(0), total_time(0.0) {}
    
    // Add a new node to the tree
    void addNode(std::shared_ptr<BBNode> node) {
        all_nodes.push_back(node);
        if (!node->is_pruned) {
            open_nodes.push(node);
        }
    }
    
    // Get the next node to explore
    std::shared_ptr<BBNode> getNextNode() {
        if (open_nodes.empty()) {
            return nullptr;
        }
        auto node = open_nodes.top();
        open_nodes.pop();
        return node;
    }
    
    // Update global bounds
    void updateGlobalBounds(double lb, double ub, const Eigen::MatrixXd& solution) {
        if (lb > global_lower_bound) {
            global_lower_bound = lb;
        }
        if (ub < global_upper_bound) {
            global_upper_bound = ub;
            best_solution = solution;
        }
    }

	void updateGlobalLowerBound() {
     // if open node is empty do nothing
	 // else since we are using min-heap, the top node has the smallest lower bound
	 if (!open_nodes.empty()) {
		 global_lower_bound = open_nodes.top()->lower_bound;
	 }
}
    
    // Prune nodes based on global upper bound
    void pruneNodes() {
        std::priority_queue<std::shared_ptr<BBNode>, 
                           std::vector<std::shared_ptr<BBNode>>, 
                           BBNodeComparator> new_queue;
        
        while (!open_nodes.empty()) {
            auto node = open_nodes.top();
            open_nodes.pop();
            
            // Prune if lower bound is worse than global upper bound
            if (node->lower_bound >= global_upper_bound - 1e-6) {
                std::cout << "*** Node " << node->node_id << " PRUNED by bound ***" << std::endl;
                std::cout << "    Node LB: " << node->lower_bound << std::endl;
                std::cout << "    Global UB: " << global_upper_bound << std::endl;
                std::cout << "    Difference: " << (node->lower_bound - global_upper_bound) << std::endl;
                node->is_pruned = true;
                nodes_pruned++;
            } else {
                new_queue.push(node);
            }
        }
        
        open_nodes = std::move(new_queue);
    }
};

inline int getPairIndex(int i, int j, int N) {
    if (j < i) std::swap(i, j);  // Ensure j >= i
    return i * (2 * N - i + 1) / 2 + j - i;
}

