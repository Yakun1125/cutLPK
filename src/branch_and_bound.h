#include "Utils_Struct.h"
#include "Rounding_heuristic.h"

// Printing functions for BnB progress
void printBnBHeader(std::ofstream* log_file = nullptr, int verbose = 1);
void printBnBStatus(
    const BBTree& tree,
    const std::shared_ptr<BBNode>& current_node,
    double elapsed_time,
    double node_solve_time,
    const std::string& node_status,
    const std::string& cp_status,
    bool found_better_ub,
    std::ofstream* log_file = nullptr,
    int verbose = 1,
    int output_level = 2
);
void printBnBSummary(const BBTree& tree, double total_time, BnBStatus status, std::ofstream* log_file = nullptr, int verbose = 1);

std::pair<int, int> selectBranchingVariable(const Eigen::MatrixXd& Xsol, int N);

// Function to check if LP solution is integer (all X(i,j) are 0 or 1)
bool isIntegerSolution(const Eigen::MatrixXd& Xsol, double tolerance = 1e-4);

// Function to apply branching constraints to LPK problem
void applyBranchConstraints(LPK& lp, const std::vector<BranchConstraint>& constraints, int N);

// Function to create two child nodes with branching constraints
std::pair<std::shared_ptr<BBNode>, std::shared_ptr<BBNode>> 
createBranchNodes(std::shared_ptr<BBNode> parent, int i, int j, BBTree& tree);

// Main branch and bound solver
BnBStatus branch_and_bound_solver(
    int N,
    int K,
    cutLPKSolveInfo& cutLPKInfo,
    std::vector<validInequality>& cutting_planes,
    LPK& lp,
    RoundingHeuristic& roundingHeuristic,
    const parameters& params
);