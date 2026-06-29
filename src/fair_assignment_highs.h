#pragma once

#include "fair_assignment_solver.h"
#include <memory>

// Forward declare HiGHS types (avoid header dependency for clients)
class Highs;
struct HighsSolution;

class HiGHSFairAssignmentSolver : public FairAssignmentSolver {
public:
    HiGHSFairAssignmentSolver();
    ~HiGHSFairAssignmentSolver() override;

    void buildAlphaModel(int N, int K, int numGroups,
        const std::vector<std::vector<bool>>& dataGroups,
        const std::vector<int>& groupRatio,
        const std::vector<double>& fairness_param) override;

    void buildTauModel(int N, int K, int numGroups,
        const std::vector<std::vector<bool>>& dataGroups,
        const std::vector<int>& groupRatio,
        const std::vector<double>& fairness_param) override;

    FairAssignStatus solve(const VectorXdList& dataPoints,
        const VectorXdList& centroids,
        std::vector<int>& assignment) override;

    void addSameClusterConstraint(int i, int j) override;
    void addDiffClusterConstraint(int i, int j) override;
    void removeBranchConstraints() override;

    int getNumConstraints() const override;
    std::string solverName() const override { return "HiGHS"; }

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;

    // Internal helpers
    void buildBaseModel(int N, int K, int numGroups,
        const std::vector<std::vector<bool>>& dataGroups,
        const std::vector<int>& groupRatio,
        const std::vector<double>& fairness_param,
        bool isAlpha);
    FairAssignStatus extractSolution(std::vector<int>& assignment);
    void applyWarmStart();
};
