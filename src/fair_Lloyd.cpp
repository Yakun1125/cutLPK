#include "fair_Lloyd.h"
#ifdef ENABLE_GUROBI
#include "fair_assignment_gurobi.h"
#endif
#ifdef ENABLE_HIGHS
#include "fair_assignment_highs.h"
#endif
#include <iostream>

int gcd(int a, int b) {
    while (b != 0) {
        int temp = a % b;
        a = b;
        b = temp;
    }
    return a;
}

// ---------------------------------------------------------------
// Simplified tau parameter for fair assignment model.
//
// The tau-ratio constraint requires each cluster k to get at least
//   ceil(tau_g * |X_g|)  points from group g, where tau_g = rho / K.
//
// Since the LHS (a point count) is integral, we ceil the fractional
// RHS without changing the feasible set.  This function computes
//   f_g = target_num / |X_g|,   target_num integer,
// such that:
//   1. f_g >= tau_g  (ceil — never weakens fairness),
//   2. K * target_num <= |X_g|  (K clusters can simultaneously
//      satisfy the constraint without exceeding group total),
//   3. The model constraint  sum_{i in X_g} x_{i,k} >= |X_g| * f_g
//      has integral RHS (= target_num), preserving total unimodularity
//      so the LP optimum is integer-valued.
//
// When the tightest setting would be infeasible (K * ceil > |X_g|),
// we decrement to the largest feasible integer — this is the tightest
// constraint still admitting a K-cluster assignment.
// ---------------------------------------------------------------
double find_simplified_fraction_Tau(
    int numerator,
    int K,
    double target_factor){
    // numerator * target_factor rounded up, but must satisfy K * target_num <= numerator
    double scaled_num = numerator * target_factor;
    double integral_part;
    double fractional_part = std::modf(scaled_num, &integral_part);
    double adjusted_factor = target_factor;
    // Case 1: scaled_num is (nearly) integer — use directly if feasible
    if (fractional_part < 0.000001 || fractional_part > 0.999999)
    {
        int target_num = static_cast<int>(std::round(scaled_num));
        adjusted_factor = double(target_num) / double(numerator);
        if (target_num * K <= numerator)
        {
            std::cout << "Found direct simplification with exact factor:\n";
            std::cout << "Original group number: " << numerator << "\n";
            std::cout << "Original minimum group number: " << scaled_num << "\n";
            std::cout << "Target factor: " << target_factor << "\n";
            std::cout << "Adjusted factor: " << adjusted_factor << "\n";
            std::cout << "Minimum group number required: " << target_num << "\n";
            return adjusted_factor;
        }
    }
    else{
        // Case 2: ceil and check feasibility for K clusters
        int target_num = static_cast<int>(std::ceil(scaled_num));
        adjusted_factor = double(target_num) / double(numerator);
        if (target_num * K <= numerator)
        {
            std::cout << "Found increased factor:\n";
            std::cout << "Original group number: " << numerator << "\n";
            std::cout << "Original minimum group number: " << scaled_num << "\n";
            std::cout << "Target factor: " << target_factor << "\n";
            std::cout << "Adjusted factor: " << adjusted_factor << "\n";
            std::cout << "Minimum group number required: " << target_num << "\n";
            return adjusted_factor;
        }
        else{
            // Case 3: ceil too tight — decrement to largest feasible integer
            while (target_num > 0)
            {
                target_num--;
                adjusted_factor = double(target_num) / double(numerator);
                if (target_num * K <= numerator)
                {
                    std::cout << "Found simplification (minimum group number decreased):\n";
                    std::cout << "Original group number: " << numerator << "\n";
                    std::cout << "Original minimum group number: " << scaled_num << "\n";
                    std::cout << "Target factor: " << target_factor << "\n";
                    std::cout << "Adjusted factor: " << adjusted_factor << "\n";
                    std::cout << "Minimum group number required: " << target_num << "\n";
                    return adjusted_factor;
                }
            }
        }
    }
    std::cout << "No valid tau simplification found, returning original factor.\n";
    return target_factor; 
}

// Compute integral-feasible tau parameters per group (see find_simplified_fraction_Tau above).
std::vector<double> tau_fairParam_adjustment(const std::vector<int> &groupRatio, double fairness_param, int N, int K)
{
    std::vector<double> adjusted_factors(groupRatio.size(), 1.0);
    for (int g = 0; g < static_cast<int>(groupRatio.size()); g++)
    {
        int numerator = groupRatio[g];
        double target_factor = fairness_param;
        double adjusted_factor = find_simplified_fraction_Tau(numerator, K, target_factor);
        adjusted_factors[g] = adjusted_factor;
    }
    return adjusted_factors;
}

// ---------------------------------------------------------------------------
// Solver-agnostic fair Lloyd heuristic
// ---------------------------------------------------------------------------
std::pair<double, std::vector<int>> runFairKMeans(
    const VectorXdList& dataPoints, int k, int maxIterations,
    int random_seed, FairAssignmentSolver& solver)
{
    int n = static_cast<int>(dataPoints.size());
    // Initialize centroids with k-means++
    VectorXdList centroids = initializeCentroidsPlusPlus(dataPoints, k, random_seed);
    std::vector<int> assignment(n, -1);
    
    // Start with a FAIR assignment
    FairAssignStatus init_status = solver.solve(dataPoints, centroids, assignment);
    if (init_status == FairAssignStatus::INFEASIBLE || init_status == FairAssignStatus::UNBOUNDED) {
        std::cerr << "Initial fair assignment infeasible or unbounded. Stopping." << std::endl;
        return std::make_pair(kInfinity, assignment);
    }
    double currentWCSS = computeWCSS(dataPoints, centroids, assignment);
    double prevWCSS = currentWCSS;

    for (int iter = 0; iter < maxIterations; ++iter)
    {
        // Update centroids from current assignment
        updateCentroids(dataPoints, centroids, assignment, k);
        // Fair assignment with new centroids
        FairAssignStatus assign_status = solver.solve(dataPoints, centroids, assignment);
        if (assign_status == FairAssignStatus::INFEASIBLE || assign_status == FairAssignStatus::UNBOUNDED) {
            std::cerr << "Fair assignment became infeasible during Lloyd iteration. Stopping early." << std::endl;
            break;
        }
        if (assign_status != FairAssignStatus::SUCCESS) {
            // CONVERGED: assignment unchanged
            break;
        }
        currentWCSS = computeWCSS(dataPoints, centroids, assignment);
        // Converge when objective stops decreasing
        if (currentWCSS >= prevWCSS)
        {
            break;
        }
        prevWCSS = currentWCSS;
    }

    return std::make_pair(currentWCSS, assignment);
}

// ---------------------------------------------------------------------------
// Factory: create the configured fair assignment solver
// ---------------------------------------------------------------------------
std::unique_ptr<FairAssignmentSolver> createFairAssignmentSolver(
    const std::string& solverName)
{
    if (solverName == "highs") {
#ifdef ENABLE_HIGHS
        return std::make_unique<HiGHSFairAssignmentSolver>();
#else
        throw std::runtime_error(
            "HiGHS solver requested but cutLPK was built without HiGHS support. "
            "Rebuild with -DENABLE_HIGHS=ON or set fair_assignment_solver=\"gurobi\".");
#endif
    } else if (solverName == "gurobi") {
#ifdef ENABLE_GUROBI
        return std::make_unique<GurobiFairAssignmentSolver>();
#else
        throw std::runtime_error(
            "Gurobi solver requested but cutLPK was built without Gurobi support. "
            "Rebuild with -DENABLE_GUROBI=ON or set fair_assignment_solver=\"highs\".");
#endif
    } else {
        throw std::runtime_error(
            "Unknown fair assignment solver: '" + solverName +
            "'. Supported: 'highs', 'gurobi'.");
    }
}
