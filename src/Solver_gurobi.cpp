#include "Solver_gurobi.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "fair_Lloyd.h"
#include "gurobi_c++.h"

namespace {
double toGRBLowerBound(double value) {
    if (std::isinf(value) && value < 0) {
        return -GRB_INFINITY;
    }
    return value;
}

double toGRBUpperBound(double value) {
    if (std::isinf(value) && value > 0) {
        return GRB_INFINITY;
    }
    return value;
}
}  // namespace

int solver_gurobi(double& dual_obj, double& primal_obj,
                  Eigen::MatrixXd& Xsol, std::vector<validInequality>& cuts,
                  LPK& lp, float tolerance, float time_limit) {
    int return_code = 0;

    try {
        // --- Create and configure Gurobi environment (C++ API) ---
        GRBEnv env = GRBEnv(true);
        //silence gurobi output
        env.set(GRB_IntParam_OutputFlag, 0);
        //env.set(GRB_StringParam_LogFile, "gurobi.log");

        // Configure WLS credentials (if provided)
        setupGurobiWLS(env);

        env.start();

        env.set(GRB_IntParam_OutputFlag, 0);
        env.set(GRB_DoubleParam_TimeLimit, time_limit);
        env.set(GRB_IntParam_PDHGGPU, 1);
        env.set(GRB_IntParam_Method, 6);
        env.set(GRB_IntParam_Crossover, 0);

        // --- Build model ---
        GRBModel model(env);
        model.set(GRB_IntAttr_ModelSense, 1);  // Minimize

        const int numVars = lp.N * (lp.N + 1) / 2;
        // std::cout << "numVars = " << numVars
        //           << ", lp.objCoef.size() = " << lp.objCoef.size() << std::endl;
        // double objCoefSum = 0.0;
        // for (int i = 0; i < std::min(10, static_cast<int>(lp.objCoef.size())); ++i) {
        //     std::cout << "objCoef[" << i << "] = " << lp.objCoef[i] << std::endl;
        //     objCoefSum += std::abs(lp.objCoef[i]);
        // }
        // std::cout << "Sum of |objCoef| (first 10): " << objCoefSum << std::endl;

        std::vector<GRBVar> vars;
        vars.reserve(numVars);

        for (int idx = 0; idx < numVars; ++idx) {
            const double lb = toGRBLowerBound(lp.varLb[idx]);
            const double ub = toGRBUpperBound(lp.varUb[idx]);
            vars.emplace_back(model.addVar(lb, ub, lp.objCoef[idx], GRB_CONTINUOUS));
        }

        model.update();

        Eigen::SparseMatrix<double, Eigen::RowMajor> consMatrixRow = lp.ConsMatrix;
        const int numConstr = consMatrixRow.rows();
        std::vector<GRBConstr> constrs;
        constrs.reserve(numConstr);

        for (int row = 0; row < numConstr; ++row) {
            GRBLinExpr expr = 0.0;
            for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(consMatrixRow, row); it; ++it) {
                const int colIdx = it.col();
                if (colIdx < 0 || colIdx >= numVars) {
                    std::cerr << "Constraint column index out of bounds: " << colIdx
                              << " for numVars = " << numVars << std::endl;
                    throw std::runtime_error("Invalid constraint column index");
                }
                expr += it.value() * vars[colIdx];
            }

            double lb = toGRBLowerBound(lp.consLb[row]);
            double ub = toGRBUpperBound(lp.consUb[row]);

            if (row <= lp.N) {
                // Force equality for the first N+1 constraints, matching original logic
                lb = ub = lp.consUb[row];
            }

            constrs.emplace_back(model.addRange(expr, lb, ub));
        }

        model.update();

        // --- Optimize ---
        model.optimize();

        const int optimStatus = model.get(GRB_IntAttr_Status);
        if (optimStatus != GRB_OPTIMAL && optimStatus != GRB_SUBOPTIMAL &&
            optimStatus != GRB_TIME_LIMIT) {
            std::cerr << "Optimization finished with status: " << optimStatus << std::endl;
            return_code = 3;
            return return_code;
        }

        if (model.get(GRB_IntAttr_SolCount) == 0) {
            std::cerr << "No solution available from Gurobi." << std::endl;
            return_code = 3;
            return return_code;
        }

        primal_obj = model.get(GRB_DoubleAttr_ObjVal);

        std::vector<double> xVec(numVars, 0.0);
        for (int idx = 0; idx < numVars; ++idx) {
            xVec[idx] = vars[idx].get(GRB_DoubleAttr_X);
        }

        // Debug information about primal variables
        // std::cout << "Solution preview (first 10 variables):" << std::endl;
        // double xSum = 0.0;
        // for (int i = 0; i < std::min(10, numVars); ++i) {
        //     std::cout << "x[" << i << "] = " << xVec[i] << std::endl;
        //     xSum += std::abs(xVec[i]);
        // }
        // std::cout << "Sum of |x| (first 10): " << xSum << std::endl;

        // Populate Xsol (symmetric matrix)
        int colIdx = 0;
        for (int i = 0; i < lp.N; ++i) {
            for (int j = i; j < lp.N; ++j) {
                const double value = xVec[colIdx++];
                Xsol(i, j) = value;
                if (i != j) {
                    Xsol(j, i) = value;
                }
            }
        }

        // Extract dual solution
        std::vector<double> dualVec(numConstr, 0.0);
        for (int idx = 0; idx < numConstr; ++idx) {
            dualVec[idx] = constrs[idx].get(GRB_DoubleAttr_Pi);
        }

        // Manual primal objective check
        // double manualPrimal = 0.0;
        // for (int i = 0; i < numVars; ++i) {
        //     manualPrimal += lp.objCoef[i] * xVec[i];
        // }
        // std::cout << "primal obj check = " << manualPrimal << std::endl;
        // std::cout << "Gurobi reported primal obj = " << primal_obj << std::endl;

        // Calculate dual objective components
        if (!dualVec.empty()) {
            double dualManual = dualVec[0] * lp.consUb[0];
            for (int i = 1; i < lp.N + 1 && i < static_cast<int>(dualVec.size()); ++i) {
                dualManual += dualVec[i];
            }
            // std::cout << "temp dual obj = " << dualManual << " " << dualVec[0]
            //           << " " << lp.consUb[0] << std::endl;
        }

        Eigen::Map<const Eigen::VectorXd> xEigen(xVec.data(), numVars);
        Eigen::VectorXd r = lp.ConsMatrix * xEigen;

        const int numCuts = static_cast<int>(cuts.size());
        const int cutsIdxStart = std::max(0, numConstr - numCuts);
        for (int cutIdx = 0; cutIdx < numCuts; ++cutIdx) {
            const int rowIdx = cutsIdxStart + cutIdx;
            if (rowIdx >= 0 && rowIdx < r.size() && rowIdx < static_cast<int>(dualVec.size())) {
                cuts[cutIdx].violation = r[rowIdx];
                cuts[cutIdx].dual_value = dualVec[rowIdx];
            }
        }

        Eigen::Map<const Eigen::VectorXd> dualEigen(dualVec.data(), numConstr);
        Eigen::VectorXd rDual = lp.ConsMatrix.transpose() * dualEigen -
                                Eigen::Map<const Eigen::VectorXd>(lp.objCoef.data(), lp.objCoef.size());

        // const double dualNorm = rDual.norm();
        // std::cout << "dual norm = " << dualNorm << std::endl;

        // for (int i = 0; i < cutsIdxStart && i < static_cast<int>(dualVec.size()); ++i) {
        //     if (std::abs(dualVec[i]) > 1e-5) {
        //         std::cout << "nonzero dual before cuts at " << i
        //                   << ", value = " << dualVec[i] << std::endl;
        //     }
        // }

        double dualValueSum = 0.0;
        if (!dualVec.empty()) {
            dualValueSum = dualVec[0] * lp.consUb[0];
            for (int i = 1; i < lp.N + 1 && i < static_cast<int>(dualVec.size()); ++i) {
                dualValueSum += dualVec[i];
            }

            for (int i = lp.N + 1; i < cutsIdxStart && i < static_cast<int>(dualVec.size()); ++i) {
                if (dualVec[i] > 0) {
                    dualValueSum += dualVec[i] * lp.consLb[i];
                } else {
                    dualValueSum += dualVec[i] * lp.consUb[i];
                }
            }

            int dualColIdx = 0;
            for (int i = 0; i < lp.N; ++i) {
                for (int j = i; j < lp.N; ++j) {
                    if (dualColIdx < rDual.size() && rDual[dualColIdx] > 0) {
                        dualValueSum -= rDual[dualColIdx];
                    }
                    ++dualColIdx;
                }
            }
        }

        dual_obj = dualValueSum;

        //std::cout << "dual obj = " << dual_obj << ", primal obj = " << primal_obj << std::endl;

    } catch (GRBException& e) {
        std::cerr << "Gurobi exception: code " << e.getErrorCode()
                  << ", message: " << e.getMessage() << std::endl;
        return_code = 3;
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return_code = 3;
    }

    return return_code;
}