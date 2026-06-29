#include "fair_assignment_highs.h"
#include <cstdint>   // for uint8_t used by HiGHS enums
#include "Highs.h"

#include <iostream>
#include <cstring>
#include <algorithm>
#include <stdexcept>

// ---------------------------------------------------------------------------
// Pimpl struct holds all HiGHS-specific state
// ---------------------------------------------------------------------------
struct HiGHSFairAssignmentSolver::Impl {
    // Use unique_ptr because Highs::operator= is deleted (const timer member)
    std::unique_ptr<Highs> highs;

    // Problem dimensions
    int N = 0;
    int K = 0;
    int numGroups = 0;
    bool isAlpha = false;
    bool modelBuilt = false;
    bool firstSolveDone = false;

    // Row tracking for branch constraints (indices of added rows)
    int baseRows = 0;                       // rows before branch constraints
    std::vector<int> branchRowIndices;       // row indices added by branch

    // Warm start storage
    bool hasBasis = false;
    HighsBasis prevBasis;
    bool hasSolution = false;
    HighsSolution prevSolution;

    // Cached group membership for quick access
    std::vector<std::vector<bool>> dataGroups;
    std::vector<int> groupRatio;
    std::vector<double> fairnessParam;

    Impl() : highs(std::make_unique<Highs>()) {
        highs->setOptionValue("output_flag", false);
    }
};

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------
HiGHSFairAssignmentSolver::HiGHSFairAssignmentSolver()
    : pImpl(std::make_unique<Impl>())
{
    // Silence HiGHS by default
    pImpl->highs->setOptionValue("output_flag", false);
}

HiGHSFairAssignmentSolver::~HiGHSFairAssignmentSolver() = default;

// ---------------------------------------------------------------------------
// Build alpha-fairness model (ratio constraints, binary variables)
// ---------------------------------------------------------------------------
void HiGHSFairAssignmentSolver::buildAlphaModel(
    int N, int K, int numGroups,
    const std::vector<std::vector<bool>>& dataGroups,
    const std::vector<int>& groupRatio,
    const std::vector<double>& fairness_param)
{
    buildBaseModel(N, K, numGroups, dataGroups, groupRatio, fairness_param, true);
}

// ---------------------------------------------------------------------------
// Build tau-fairness model (absolute-count constraints, continuous vars)
// ---------------------------------------------------------------------------
void HiGHSFairAssignmentSolver::buildTauModel(
    int N, int K, int numGroups,
    const std::vector<std::vector<bool>>& dataGroups,
    const std::vector<int>& groupRatio,
    const std::vector<double>& fairness_param)
{
    buildBaseModel(N, K, numGroups, dataGroups, groupRatio, fairness_param, false);
}

// ---------------------------------------------------------------------------
// Core model building (shared by alpha and tau)
// ---------------------------------------------------------------------------
void HiGHSFairAssignmentSolver::buildBaseModel(
    int N, int K, int numGroups,
    const std::vector<std::vector<bool>>& dataGroups,
    const std::vector<int>& groupRatio,
    const std::vector<double>& fairness_param,
    bool isAlpha)
{
    auto& impl = *pImpl;
    impl.N = N;
    impl.K = K;
    impl.numGroups = numGroups;
    impl.isAlpha = isAlpha;
    impl.dataGroups = dataGroups;
    impl.groupRatio = groupRatio;
    impl.fairnessParam = fairness_param;
    impl.modelBuilt = false;
    impl.firstSolveDone = false;
    impl.hasBasis = false;
    impl.hasSolution = false;
    impl.branchRowIndices.clear();

    // Clear any previous model by creating a fresh Highs instance
    impl.highs = std::make_unique<Highs>();
    impl.highs->setOptionValue("output_flag", false);

    int numCols = N * K;
    double inf = kHighsInf;

    // -------------------------------------------------------------------
    // Step 1: Add variables x[i][k]
    // For alpha: binary {0,1}
    // For tau: continuous [0,1]
    // -------------------------------------------------------------------
    // We add columns one by one with zero objective (set later)
    // Each column appears in:
    //   - 1 assignment row (row i)
    //   - 1 min-cluster-size row (row N + k)
    //   - up to 2*numGroups fairness rows per cluster
    
    // We'll add columns first, then rows later
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < K; ++k) {
            impl.highs->addCol(0.0,            // cost (set later in solve())
                         0.0,            // lower bound
                         1.0,            // upper bound
                         0, nullptr, nullptr);  // no matrix entries yet
        }
    }

    // Set integrality for alpha model
    if (isAlpha) {
        for (int col = 0; col < numCols; ++col) {
            impl.highs->changeColIntegrality(col, HighsVarType::kInteger);
        }
    }

    // -------------------------------------------------------------------
    // Step 2: Add assignment constraints (each point to exactly 1 cluster)
    //   sum_k x[i][k] == 1   for i = 0..N-1
    // -------------------------------------------------------------------
    std::vector<int> indices(K);
    std::vector<double> values(K, 1.0);
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < K; ++k) {
            indices[k] = i * K + k;
        }
        impl.highs->addRow(1.0, 1.0, K, indices.data(), values.data());
    }

    // -------------------------------------------------------------------
    // Step 3: Minimum cluster size (each cluster gets at least 1 point)
    //   sum_i x[i][k] >= 1   for k = 0..K-1
    // -------------------------------------------------------------------
    std::vector<int> clusterIndices(N);
    std::vector<double> clusterValues(N, 1.0);
    for (int k = 0; k < K; ++k) {
        for (int i = 0; i < N; ++i) {
            clusterIndices[i] = i * K + k;
        }
        impl.highs->addRow(1.0, inf, N, clusterIndices.data(), clusterValues.data());
    }

    // -------------------------------------------------------------------
    // Step 4: Fairness constraints
    // -------------------------------------------------------------------
    if (isAlpha) {
        // Alpha: ratio-based fairness
        // For each cluster k, each group g:
        //   LOWER: sum_{i in g} x[i][k] >= factor_lb * sum_i x[i][k]
        //   where factor_lb = (|g|/N) * fairness_param
        //   => sum_{i in g} x[i][k] - factor_lb * sum_i x[i][k] >= 0
        //
        //   UPPER (if fairness_param != 1):
        //   sum_{i in g} x[i][k] <= factor_ub * sum_i x[i][k]
        //   where factor_ub = (|g|/N) / fairness_param
        //   => factor_ub * sum_i x[i][k] - sum_{i in g} x[i][k] >= 0
        //
        // Coefficient for x[j][k]:
        //   LOWER: j in g ? (1 - factor_lb) : (-factor_lb)
        //   UPPER: j in g ? (factor_ub - 1) : (factor_ub)

        // Precompute factors
        std::vector<double> factor_lb(numGroups), factor_ub(numGroups);
        for (int g = 0; g < numGroups; ++g) {
            double ratio = static_cast<double>(groupRatio[g]) / static_cast<double>(N);
            factor_lb[g] = ratio * fairness_param[g];
            factor_ub[g] = ratio / fairness_param[g];
        }

        // Build per-cluster, per-group constraints
        // We pre-build index/value arrays (dense: all N points for every
        // fairness constraint because of the sum_i x[i][k] term)
        std::vector<int> fairIndices(N);
        std::vector<double> fairLowerVals(N);
        std::vector<double> fairUpperVals(N);

        for (int k = 0; k < K; ++k) {
            for (int g = 0; g < numGroups; ++g) {
                // Column indices for all x[i][k]
                for (int i = 0; i < N; ++i) {
                    fairIndices[i] = i * K + k;
                }

                // Lower bound constraint
                for (int i = 0; i < N; ++i) {
                    fairLowerVals[i] = dataGroups[i][g] ? (1.0 - factor_lb[g])
                                                        : (-factor_lb[g]);
                }
                impl.highs->addRow(0.0, inf, N, fairIndices.data(), fairLowerVals.data());

                // Upper bound constraint (if fairness_param != 1)
                if (std::abs(fairness_param[g] - 1.0) > 1e-6) {
                    for (int i = 0; i < N; ++i) {
                        fairUpperVals[i] = dataGroups[i][g] ? (factor_ub[g] - 1.0)
                                                            : (factor_ub[g]);
                    }
                    impl.highs->addRow(0.0, inf, N, fairIndices.data(), fairUpperVals.data());
                }
            }
        }
    } else {
        // Tau: absolute-count fairness
        // For each cluster k, each group g:
        //   LOWER: sum_{i in g} x[i][k] >= groupRatio[g] * fairness_param[g]
        //   UPPER: sum_{i in g} x[i][k] <= groupRatio[g]
        //
        // Only involves x[i][k] for i in group g (sparse)

        for (int k = 0; k < K; ++k) {
            for (int g = 0; g < numGroups; ++g) {
                int groupSize = groupRatio[g];
                
                // Collect indices of points in group g
                std::vector<int> groupIndices;
                groupIndices.reserve(groupSize);
                for (int i = 0; i < N; ++i) {
                    if (dataGroups[i][g]) {
                        groupIndices.push_back(i * K + k);
                    }
                }

                std::vector<double> groupValues(groupIndices.size(), 1.0);

                // Lower bound
                double lb = static_cast<double>(groupSize) * fairness_param[g];
                impl.highs->addRow(lb, inf,
                             static_cast<int>(groupIndices.size()),
                             groupIndices.data(), groupValues.data());

                // Upper bound
                double ub = static_cast<double>(groupSize);
                impl.highs->addRow(-inf, ub,
                             static_cast<int>(groupIndices.size()),
                             groupIndices.data(), groupValues.data());
            }
        }
    }

    // Record number of base rows (before any branch constraints)
    impl.baseRows = impl.highs->getNumRow();
    impl.modelBuilt = true;
}

// ---------------------------------------------------------------------------
// Solve: update objective with current centroid distances and re-optimize
// ---------------------------------------------------------------------------
FairAssignStatus HiGHSFairAssignmentSolver::solve(
    const VectorXdList& dataPoints,
    const VectorXdList& centroids,
    std::vector<int>& assignment)
{
    auto& impl = *pImpl;
    if (!impl.modelBuilt) {
        std::cerr << "HiGHSFairAssignmentSolver: model not built" << std::endl;
        return FairAssignStatus::ERROR;
    }

    Highs& highs = *impl.highs;
    int N = impl.N;
    int K = impl.K;

    // -------------------------------------------------------------------
    // Update objective: cost[i][k] = ||dataPoints[i] - centroids[k]||^2
    // -------------------------------------------------------------------
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < K; ++k) {
            double dist = (dataPoints[i] - centroids[k]).squaredNorm();
            int col = i * K + k;
            impl.highs->changeColCost(col, dist);
        }
    }

    // -------------------------------------------------------------------
    // Apply warm start
    // -------------------------------------------------------------------
    applyWarmStart();

    // -------------------------------------------------------------------
    // Solve
    // -------------------------------------------------------------------
    HighsStatus status = impl.highs->run();
    if (status != HighsStatus::kOk && status != HighsStatus::kWarning) {
        std::cerr << "HiGHS solve returned status: " 
                  << static_cast<int>(status) << std::endl;
        return FairAssignStatus::ERROR;
    }

    // -------------------------------------------------------------------
    // Store warm start info for next call
    // -------------------------------------------------------------------
    HighsModelStatus modelStatus = impl.highs->getModelStatus();
    if (modelStatus == HighsModelStatus::kOptimal) {
        if (!impl.isAlpha) {
            // LP: store basis for next warm start
            impl.prevBasis = impl.highs->getBasis();
            impl.hasBasis = true;
        }
        // Also store solution (useful for MIP start)
        impl.prevSolution = impl.highs->getSolution();
        impl.hasSolution = true;
        impl.firstSolveDone = true;
    }

    // -------------------------------------------------------------------
    // Extract solution into assignment vector
    // -------------------------------------------------------------------
    return extractSolution(assignment);
}

// ---------------------------------------------------------------------------
// Apply warm start (previous basis or solution)
// ---------------------------------------------------------------------------
void HiGHSFairAssignmentSolver::applyWarmStart() {
    auto& impl = *pImpl;

    if (!impl.firstSolveDone) return;

    if (!impl.isAlpha && impl.hasBasis) {
        // LP: use basis warm start (most effective for LP re-optimization)
        impl.highs->setBasis(impl.prevBasis);
    } else if (impl.hasSolution) {
        // MIP or fallback: use previous solution
        impl.highs->setSolution(impl.prevSolution);
    }
}

// ---------------------------------------------------------------------------
// Extract integer assignment from solver solution
// ---------------------------------------------------------------------------
FairAssignStatus HiGHSFairAssignmentSolver::extractSolution(
    std::vector<int>& assignment)
{
    auto& impl = *pImpl;
    int N = impl.N;
    int K = impl.K;

    HighsModelStatus modelStatus = impl.highs->getModelStatus();

    if (modelStatus == HighsModelStatus::kOptimal) {
        const HighsSolution& sol = impl.highs->getSolution();
        
        bool changed = false;
        for (int i = 0; i < N; ++i) {
            int bestK = -1;
            double bestVal = -1.0;
            for (int k = 0; k < K; ++k) {
                double val = sol.col_value[i * K + k];
                if (val > bestVal) {
                    bestVal = val;
                    bestK = k;
                }
            }
            // Threshold: require value > 0.5 for assignment
            if (bestVal > 0.5) {
                if (assignment[i] != bestK) {
                    assignment[i] = bestK;
                    changed = true;
                }
            } else {
                // No clear assignment — pick argmax anyway
                if (assignment[i] != bestK) {
                    assignment[i] = bestK;
                    changed = true;
                }
            }
        }
        return changed ? FairAssignStatus::SUCCESS : FairAssignStatus::CONVERGED;
        
    } else if (modelStatus == HighsModelStatus::kInfeasible) {
        std::cout << "Fair assignment model is infeasible." << std::endl;
        return FairAssignStatus::INFEASIBLE;
    } else if (modelStatus == HighsModelStatus::kUnbounded ||
               modelStatus == HighsModelStatus::kUnboundedOrInfeasible) {
        std::cout << "Fair assignment model is unbounded or infeasible." << std::endl;
        return FairAssignStatus::UNBOUNDED;
    } else {
        std::cout << "HiGHS solve status: " << static_cast<int>(modelStatus) << std::endl;
        return FairAssignStatus::ERROR;
    }
}

// ---------------------------------------------------------------------------
// Branch constraint: x[i][k] == x[j][k] for all k  (same cluster)
// ---------------------------------------------------------------------------
void HiGHSFairAssignmentSolver::addSameClusterConstraint(int i, int j) {
    auto& impl = *pImpl;
    int K = impl.K;

    for (int k = 0; k < K; ++k) {
        int indices[2] = { i * K + k, j * K + k };
        double values[2] = { 1.0, -1.0 };
        impl.highs->addRow(0.0, 0.0, 2, indices, values);
        impl.branchRowIndices.push_back(impl.highs->getNumRow() - 1);
    }
}

// ---------------------------------------------------------------------------
// Branch constraint: x[i][k] + x[j][k] <= 1 for all k  (different cluster)
// ---------------------------------------------------------------------------
void HiGHSFairAssignmentSolver::addDiffClusterConstraint(int i, int j) {
    auto& impl = *pImpl;
    int K = impl.K;

    for (int k = 0; k < K; ++k) {
        int indices[2] = { i * K + k, j * K + k };
        double values[2] = { 1.0, 1.0 };
        impl.highs->addRow(-kHighsInf, 1.0, 2, indices, values);
        impl.branchRowIndices.push_back(impl.highs->getNumRow() - 1);
    }
}

// ---------------------------------------------------------------------------
// Remove all previously added branch constraints
// ---------------------------------------------------------------------------
void HiGHSFairAssignmentSolver::removeBranchConstraints() {
    auto& impl = *pImpl;

    if (impl.branchRowIndices.empty()) return;

    // Delete rows in reverse order to preserve indices
    std::sort(impl.branchRowIndices.begin(), impl.branchRowIndices.end(),
              std::greater<int>());
    for (int rowIdx : impl.branchRowIndices) {
        impl.highs->deleteRows(rowIdx, rowIdx + 1);
    }
    impl.branchRowIndices.clear();
    impl.firstSolveDone = false;  // invalidate warm start after structural change
    impl.hasBasis = false;
    impl.hasSolution = false;
}

// ---------------------------------------------------------------------------
// Introspection
// ---------------------------------------------------------------------------
int HiGHSFairAssignmentSolver::getNumConstraints() const {
    return pImpl->highs->getNumRow();
}

