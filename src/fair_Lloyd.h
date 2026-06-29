#pragma once
#include "Lloyd.h"
#include "Utils_Struct.h"
#include "fair_assignment_solver.h"
#include <memory>
#include <numeric>
#include <vector>

// ---------------------------------------------------------------------------
// Tau simplification helpers (shared across all solvers)
// ---------------------------------------------------------------------------
double find_simplified_fraction_Tau(int numerator, int K, double target_factor);

std::vector<double> tau_fairParam_adjustment(
    const std::vector<int>& groupRatio, double fairness_param, int N, int K);

// ---------------------------------------------------------------------------
// Solver-agnostic fair K-means (Lloyd-style heuristic)
//
// Uses the provided FairAssignmentSolver to iteratively assign points to
// clusters while respecting fairness constraints.  The solver must already
// have its model built (buildAlphaModel / buildTauModel).
// ---------------------------------------------------------------------------
std::pair<double, std::vector<int>> runFairKMeans(
    const VectorXdList& dataPoints, int k, int maxIterations,
    int random_seed, FairAssignmentSolver& solver);

// ---------------------------------------------------------------------------
// Factory: create the configured fair assignment solver
// ---------------------------------------------------------------------------
std::unique_ptr<FairAssignmentSolver> createFairAssignmentSolver(
    const std::string& solverName);