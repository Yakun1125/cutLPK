#include "fair_assignment_gurobi.h"
#include <iostream>
#include <stdexcept>

struct GurobiFairAssignmentSolver::Impl {
    GRBEnv env;
    std::unique_ptr<GRBModel> model;
    std::vector<std::vector<GRBVar>> x;
    int N = 0, K = 0, numGroups = 0;
    bool isAlpha = false;
    
    // Tracked row names for branch constraint removal
    std::vector<std::string> branchRowNames;
    
    Impl() : env(true) {
        env.set(GRB_IntParam_OutputFlag, 0);
        try {
            env.start();
        } catch (GRBException& e) {
            throw std::runtime_error("Failed to start Gurobi environment: " + 
                                     std::string(e.getMessage()));
        }
    }
};

GurobiFairAssignmentSolver::GurobiFairAssignmentSolver()
    : pImpl(std::make_unique<Impl>()) {}

GurobiFairAssignmentSolver::~GurobiFairAssignmentSolver() = default;

// ---------------------------------------------------------------------------
// Helpers: find_simplified_fraction_Tau and tau_fairParam_adjustment
// (moved from fair_Lloyd.cpp — identical logic)
// ---------------------------------------------------------------------------
namespace {

int gcd(int a, int b) {
    while (b != 0) { int t = a % b; a = b; b = t; }
    return a;
}

double find_simplified_fraction_Tau(int numerator, int K, double target_factor) {
    double scaled_num = numerator * target_factor;
    double integral_part, fractional_part;
    fractional_part = std::modf(scaled_num, &integral_part);
    double adjusted_factor = target_factor;
    
    if (fractional_part < 0.000001 || fractional_part > 0.999999) {
        int target_num = static_cast<int>(std::round(scaled_num));
        adjusted_factor = double(target_num) / double(numerator);
        if (target_num * K <= numerator) return adjusted_factor;
    } else {
        int target_num = static_cast<int>(std::ceil(scaled_num));
        adjusted_factor = double(target_num) / double(numerator);
        if (target_num * K <= numerator) return adjusted_factor;
        else {
            while (target_num > 0) {
                target_num--;
                adjusted_factor = double(target_num) / double(numerator);
                if (target_num * K <= numerator) return adjusted_factor;
            }
        }
    }
    return target_factor;
}

std::vector<double> tau_fairParam_adjustment(
    const std::vector<int>& groupRatio, double fairness_param, int N, int K)
{
    std::vector<double> adjusted(groupRatio.size(), 1.0);
    for (size_t g = 0; g < groupRatio.size(); ++g)
        adjusted[g] = find_simplified_fraction_Tau(groupRatio[g], K, fairness_param);
    return adjusted;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// buildAlphaModel
// ---------------------------------------------------------------------------
void GurobiFairAssignmentSolver::buildAlphaModel(
    int N, int K, int numGroups,
    const std::vector<std::vector<bool>>& dataGroups,
    const std::vector<int>& groupRatio,
    const std::vector<double>& fairness_param)
{
    auto& impl = *pImpl;
    impl.N = N; impl.K = K; impl.numGroups = numGroups; impl.isAlpha = true;
    impl.branchRowNames.clear();

    impl.model = std::make_unique<GRBModel>(impl.env);
    impl.x.resize(N, std::vector<GRBVar>(K));

    try {
        // Variables
        for (int i = 0; i < N; ++i) {
            for (int k = 0; k < K; ++k) {
                impl.x[i][k] = impl.model->addVar(0.0, 1.0, 0.0, GRB_BINARY,
                    "x_" + std::to_string(i) + "_" + std::to_string(k));
            }
        }

        // Assignment constraints
        for (int i = 0; i < N; ++i) {
            GRBLinExpr sum_xik = 0.0;
            for (int k = 0; k < K; ++k) sum_xik += impl.x[i][k];
            impl.model->addConstr(sum_xik == 1, "assign_" + std::to_string(i));
        }

        // Fairness constraints
        std::vector<double> normRatio(numGroups, 0.0);
        for (int g = 0; g < numGroups; ++g)
            normRatio[g] = double(groupRatio[g]) / double(N);

        for (int k = 0; k < K; ++k) {
            GRBLinExpr sum_xik = 0.0;
            for (int i = 0; i < N; ++i) sum_xik += impl.x[i][k];
            impl.model->addConstr(sum_xik >= 1);

            for (int g = 0; g < numGroups; ++g) {
                GRBLinExpr sum_xikg = 0.0;
                for (int i = 0; i < N; ++i)
                    if (dataGroups[i][g]) sum_xikg += impl.x[i][k];

                if (std::abs(fairness_param[g] - 1.0) <= 1e-6) {
                    impl.model->addConstr(sum_xikg == (normRatio[g] * fairness_param[g]) * sum_xik,
                        "fair_lb_k" + std::to_string(k) + "_g" + std::to_string(g));
                } else {
                    impl.model->addConstr(sum_xikg >= (normRatio[g] * fairness_param[g]) * sum_xik,
                        "fair_lb_k" + std::to_string(k) + "_g" + std::to_string(g));
                    impl.model->addConstr(sum_xikg <= (normRatio[g] / fairness_param[g]) * sum_xik,
                        "fair_ub_k" + std::to_string(k) + "_g" + std::to_string(g));
                }
            }
        }
    } catch (GRBException& e) {
        std::cerr << "Gurobi error: " << e.getErrorCode() << " " << e.getMessage() << std::endl;
        throw;
    }
}

// ---------------------------------------------------------------------------
// buildTauModel
// ---------------------------------------------------------------------------
void GurobiFairAssignmentSolver::buildTauModel(
    int N, int K, int numGroups,
    const std::vector<std::vector<bool>>& dataGroups,
    const std::vector<int>& groupRatio,
    const std::vector<double>& fairness_param)
{
    auto& impl = *pImpl;
    impl.N = N; impl.K = K; impl.numGroups = numGroups; impl.isAlpha = false;
    impl.branchRowNames.clear();

    impl.model = std::make_unique<GRBModel>(impl.env);
    impl.x.resize(N, std::vector<GRBVar>(K));

    try {
        // Variables (continuous — TU guarantees integrality)
        for (int i = 0; i < N; ++i) {
            for (int k = 0; k < K; ++k) {
                impl.x[i][k] = impl.model->addVar(0.0, 1.0, 0.0, GRB_CONTINUOUS,
                    "x_" + std::to_string(i) + "_" + std::to_string(k));
            }
        }

        // Assignment constraints
        for (int i = 0; i < N; ++i) {
            GRBLinExpr sum_xik = 0.0;
            for (int k = 0; k < K; ++k) sum_xik += impl.x[i][k];
            impl.model->addConstr(sum_xik == 1, "assign_" + std::to_string(i));
        }

        // Fairness constraints
        for (int k = 0; k < K; ++k) {
            GRBLinExpr sum_xik = 0.0;
            for (int i = 0; i < N; ++i) sum_xik += impl.x[i][k];
            impl.model->addConstr(sum_xik >= 1);

            for (int g = 0; g < numGroups; ++g) {
                GRBLinExpr sum_xikg = 0.0;
                for (int i = 0; i < N; ++i)
                    if (dataGroups[i][g]) sum_xikg += impl.x[i][k];

                impl.model->addConstr(sum_xikg >= groupRatio[g] * fairness_param[g],
                    "fair_lb_k" + std::to_string(k) + "_g" + std::to_string(g));
                impl.model->addConstr(sum_xikg <= groupRatio[g],
                    "fair_ub_k" + std::to_string(k) + "_g" + std::to_string(g));
            }
        }
    } catch (GRBException& e) {
        std::cerr << "Gurobi error: " << e.getErrorCode() << " " << e.getMessage() << std::endl;
        throw;
    }
}

// ---------------------------------------------------------------------------
// solve
// ---------------------------------------------------------------------------
FairAssignStatus GurobiFairAssignmentSolver::solve(
    const VectorXdList& dataPoints,
    const VectorXdList& centroids,
    std::vector<int>& assignment)
{
    auto& impl = *pImpl;
    if (!impl.model) return FairAssignStatus::ERROR;

    int N = impl.N, K = impl.K;

    try {
        // Set objective: sum dist(i,k) * x[i][k]
        GRBLinExpr obj = 0.0;
        for (int i = 0; i < N; ++i) {
            for (int k = 0; k < K; ++k) {
                double dist = (dataPoints[i] - centroids[k]).squaredNorm();
                obj += dist * impl.x[i][k];
            }
        }
        impl.model->setObjective(obj, GRB_MINIMIZE);
        impl.model->optimize();

        int status = impl.model->get(GRB_IntAttr_Status);
        if (status == GRB_OPTIMAL) {
            bool changed = false;
            for (int i = 0; i < N; ++i) {
                for (int k = 0; k < K; ++k) {
                    if (impl.x[i][k].get(GRB_DoubleAttr_X) > 0.5) {
                        if (assignment[i] != k) { assignment[i] = k; changed = true; }
                        break;
                    }
                }
            }
            return changed ? FairAssignStatus::SUCCESS : FairAssignStatus::CONVERGED;
        } else if (status == GRB_INF_OR_UNBD) {
            return FairAssignStatus::UNBOUNDED;
        } else if (status == GRB_INFEASIBLE) {
            return FairAssignStatus::INFEASIBLE;
        } else {
            return FairAssignStatus::ERROR;
        }
    } catch (GRBException& e) {
        std::cerr << "Gurobi error: " << e.getErrorCode() << " " << e.getMessage() << std::endl;
        return FairAssignStatus::ERROR;
    }
}

// ---------------------------------------------------------------------------
// Branch constraints
// ---------------------------------------------------------------------------
void GurobiFairAssignmentSolver::addSameClusterConstraint(int i, int j) {
    auto& impl = *pImpl;
    if (!impl.model) return;
    for (int k = 0; k < impl.K; ++k) {
        std::string name = "same_" + std::to_string(i) + "_" + std::to_string(j) + "_k" + std::to_string(k);
        impl.model->addConstr(impl.x[i][k] - impl.x[j][k] == 0, name);
        impl.branchRowNames.push_back(name);
    }
    impl.model->update();
}

void GurobiFairAssignmentSolver::addDiffClusterConstraint(int i, int j) {
    auto& impl = *pImpl;
    if (!impl.model) return;
    for (int k = 0; k < impl.K; ++k) {
        std::string name = "diff_" + std::to_string(i) + "_" + std::to_string(j) + "_k" + std::to_string(k);
        impl.model->addConstr(impl.x[i][k] + impl.x[j][k] <= 1, name);
        impl.branchRowNames.push_back(name);
    }
    impl.model->update();
}

void GurobiFairAssignmentSolver::removeBranchConstraints() {
    auto& impl = *pImpl;
    if (!impl.model || impl.branchRowNames.empty()) return;
    
    GRBConstr* constrs = impl.model->getConstrs();
    int numConstrs = impl.model->get(GRB_IntAttr_NumConstrs);
    for (int i = 0; i < numConstrs; ++i) {
        std::string name = constrs[i].get(GRB_StringAttr_ConstrName);
        for (const auto& branchName : impl.branchRowNames) {
            if (name == branchName) {
                impl.model->remove(constrs[i]);
                break;
            }
        }
    }
    delete[] constrs;
    impl.branchRowNames.clear();
    impl.model->update();
}

int GurobiFairAssignmentSolver::getNumConstraints() const {
    if (!pImpl->model) return 0;
    return pImpl->model->get(GRB_IntAttr_NumConstrs);
}

GRBModel* GurobiFairAssignmentSolver::getModel() const {
    return pImpl->model.get();
}

std::vector<std::vector<GRBVar>>* GurobiFairAssignmentSolver::getVars() const {
    return &pImpl->x;
}
