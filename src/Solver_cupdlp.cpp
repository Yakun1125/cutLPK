#include "Solver_cupdlp.h"
#include <iostream>
#include <chrono>
#include <stdexcept> // For std::runtime_error
#include <cstring>

#ifdef ENABLE_CUPDLPX
#include "cupdlpx.h"  
int solver_cupdlpx(
    double& dual_obj, double& primal_obj, Eigen::MatrixXd& Xsol,
    std::vector<validInequality>& cuts, LPK& lp,
    float tolerance, float time_limit,
    const std::vector<double>* primal_init, // nullptr if not provided
    const std::vector<double>* dual_init,   // nullptr if not provided
    std::vector<double>* primal_out,        // output: can be nullptr
    std::vector<double>* dual_out           // output: can be nullptr
) {
    int numVars = lp.N * (lp.N + 1) / 2;
    int numConstr = lp.consLb.size();
    int solver_retcode = 0;

	int numCuts = cuts.size();
    int cuts_idx_start = numConstr - numCuts;
    // --- BEGIN: Use cuPDLPx API ---
    matrix_desc_t A_desc;
    A_desc.m = lp.consLb.size();
    A_desc.n = lp.objCoef.size();
    A_desc.fmt = matrix_csc;
    A_desc.zero_tolerance = 0.0;
    A_desc.data.csc.nnz = lp.ConsMatrix.nonZeros();
    A_desc.data.csc.col_ptr = const_cast<int*>(lp.ConsMatrix.outerIndexPtr());
    A_desc.data.csc.row_ind = const_cast<int*>(lp.ConsMatrix.innerIndexPtr());
    A_desc.data.csc.vals = const_cast<double*>(lp.ConsMatrix.valuePtr());

    // Objective, bounds
    const double* c = lp.objCoef.data();
    const double* l = lp.consLb.data();
    const double* u = lp.consUb.data();

        // Create problem
    lp_problem_t* prob = create_lp_problem(
        c,        // c
        &A_desc,  // A
        l,        // con_lb
        u,         // con_ub
        lp.varLb.data(), // var_lb 
        lp.varUb.data(), // var_ub 
        NULL     // objective_constant
    );
        if (!prob) {
        std::cerr << "[solver_cupdlp] create_lp_problem failed." << std::endl;
        return 2;
    }

    double* primal = nullptr;
    double* dual = nullptr;
    if (primal_init && primal_init->size() == A_desc.n) {
        primal = (double*)malloc(A_desc.n * sizeof(double));
        memcpy(primal, primal_init->data(), A_desc.n * sizeof(double));
    } else {
        primal = (double*)calloc(A_desc.n, sizeof(double));
    }
    if (dual_init && dual_init->size() == A_desc.m) {
        dual = (double*)malloc(A_desc.m * sizeof(double));
        memcpy(dual, dual_init->data(), A_desc.m * sizeof(double));
    } else {
        dual = (double*)calloc(A_desc.m, sizeof(double));
    }
    set_start_values(prob, primal, dual);


    // Solve
    pdhg_parameters_t params;
    set_default_parameters(&params);

    // Set your custom tolerance and time limit
    params.termination_criteria.eps_optimal_relative = tolerance; 
    params.termination_criteria.eps_feasible_relative = tolerance;  
    params.termination_criteria.time_sec_limit = time_limit;
    cupdlpx_result_t* res = solve_lp_problem(prob, &params);
    lp_problem_free(prob);
    if (!res) {
        std::cerr << "[solver_cupdlp] solve_lp_problem failed." << std::endl;
        return 2;
    }
    if (res->termination_reason == TERMINATION_REASON_TIME_LIMIT){
        solver_retcode = 1;
    }

    // Extract solution
    primal_obj = res->primal_objective_value;
    dual_obj = res->dual_objective_value;

    // Fill Xsol (assuming Xsol is square and symmetric)
    int col_idx = 0;
    for (int i = 0; i < lp.N; ++i) {
        for (int j = i; j < lp.N; ++j) {
            double value = res->primal_solution[col_idx++];
            Xsol(i, j) = value;
            if (i != j) Xsol(j, i) = value;
        }
    }

        // Compute constraint residuals: r = A x
    Eigen::VectorXd x_vec = Eigen::Map<const Eigen::VectorXd>(res->primal_solution, numVars);
    Eigen::VectorXd r = lp.ConsMatrix * x_vec;

    // Update cuts info
    for (int cut_idx = 0; cut_idx < numCuts; ++cut_idx) {
        cuts[cut_idx].violation = r[cuts_idx_start + cut_idx];
        cuts[cut_idx].dual_value = res->dual_solution[cuts_idx_start + cut_idx];
    }

    // --- Dual objective calculation ---
    // Recompute r for dual objective
    Eigen::VectorXd dual_vec = Eigen::Map<const Eigen::VectorXd>(res->dual_solution, numConstr);
    r = lp.ConsMatrix.transpose() * dual_vec - Eigen::VectorXd::Map(lp.objCoef.data(), lp.objCoef.size());

    double dual_value_sum = 0.0;
    // The following matches your original dual_obj logic
    if (numConstr > 0) {
        dual_value_sum = dual_vec[0] * lp.consUb[0];
        for (int i = 1; i < lp.N + 1; i++) {
            dual_value_sum += dual_vec[i];
        }
        for (int i = lp.N + 1; i < cuts_idx_start; i++) {
            if (dual_vec[i] > 0) {
                dual_value_sum += dual_vec[i] * lp.consLb[i];
            } else {
                dual_value_sum += dual_vec[i] * lp.consUb[i];
            }
        }
        col_idx = 0;
        for (int i = 0; i < lp.N; ++i) {
            for (int j = i; j < lp.N; ++j) {
                if (r[col_idx] > 0) {
                    dual_value_sum -= r[col_idx]*lp.varUb[col_idx]; // use varUb
                }
                col_idx++;
            }
        }
    }
    dual_obj = dual_value_sum;

    if (primal_out) {
        primal_out->assign(res->primal_solution, res->primal_solution + A_desc.n);
    }
    if (dual_out) {
        dual_out->assign(res->dual_solution, res->dual_solution + A_desc.m);
    }

    free(primal);
    free(dual);
    // Clean up
    cupdlpx_result_free(res);

    return solver_retcode;
    // --- END: Use cuPDLPx API ---
}
#endif

#ifndef ENABLE_CUPDLPX
int solver_cupdlpx(
    double&, double&, Eigen::MatrixXd&,
    std::vector<validInequality>&, LPK&,
    float, float,
    const std::vector<double>*, const std::vector<double>*,
    std::vector<double>*, std::vector<double>*)
{
    std::cerr << "[solver_cupdlpx] cuPDLPx support not enabled in this build." << std::endl;
    return -1;
}
#endif