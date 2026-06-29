#pragma once

#include "Lloyd.h"
#include "Utils_Struct.h"
#include <vector>
#include <memory>
#include <string>

// Status for fair assignment operations
enum class FairAssignStatus {
    SUCCESS = 0,           // Assignment successful and changed
    CONVERGED = 1,         // Assignment successful but no changes (converged)
    INFEASIBLE = 2,        // Problem is infeasible
    UNBOUNDED = 3,         // Problem is unbounded
    ERROR = 4              // Other error
};

// ---------------------------------------------------------------------------
// Abstract interface for fair assignment MIP/LP solvers.
//
// The model is built once (buildAlphaModel / buildTauModel) and then
// re-solved many times with updated objective coefficients (solve).
// This matches the iterative fair Lloyd pattern.
// ---------------------------------------------------------------------------
class FairAssignmentSolver {
public:
    FairAssignmentSolver() = default;
    virtual ~FairAssignmentSolver() = default;

    // --- Model construction (call exactly one) ---

    // Alpha-fairness: ratio-based constraints, binary variables
    virtual void buildAlphaModel(
        int N, int K, int numGroups,
        const std::vector<std::vector<bool>>& dataGroups,
        const std::vector<int>& groupRatio,
        const std::vector<double>& fairness_param) = 0;

    // Tau-fairness: absolute-count constraints, continuous variables (TU)
    virtual void buildTauModel(
        int N, int K, int numGroups,
        const std::vector<std::vector<bool>>& dataGroups,
        const std::vector<int>& groupRatio,
        const std::vector<double>& fairness_param) = 0;

    // --- Solve with current centroids (updates objective, re-solves) ---
    virtual FairAssignStatus solve(
        const VectorXdList& dataPoints,
        const VectorXdList& centroids,
        std::vector<int>& assignment) = 0;

    // --- Branch constraint management (for BnB) ---
    virtual void addSameClusterConstraint(int i, int j) = 0;
    virtual void addDiffClusterConstraint(int i, int j) = 0;
    virtual void removeBranchConstraints() = 0;

    // --- Introspection ---
    virtual int getNumConstraints() const = 0;
    virtual std::string solverName() const = 0;
};
