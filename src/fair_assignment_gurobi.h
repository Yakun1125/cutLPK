#pragma once

#include "fair_assignment_solver.h"
#include "gurobi_c++.h"
#include <memory>

class GurobiFairAssignmentSolver : public FairAssignmentSolver {
public:
    GurobiFairAssignmentSolver();
    ~GurobiFairAssignmentSolver() override;

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
    std::string solverName() const override { return "Gurobi"; }

    // Access to internal Gurobi objects (for backward compat)
    GRBModel* getModel() const;
    std::vector<std::vector<GRBVar>>* getVars() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};
