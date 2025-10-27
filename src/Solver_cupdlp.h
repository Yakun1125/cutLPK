#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
#include <vector>
#include "Utils_Struct.h"

int solver_cupdlpx(double& dual_obj, double& primal_obj, Eigen::MatrixXd& Xsol, std::vector<validInequality>& cuts, LPK& lp, float tolerance, float time_limit,    const std::vector<double>* primal_init, // nullptr if not provided
    const std::vector<double>* dual_init,   // nullptr if not provided
    std::vector<double>* primal_out,        // output: can be nullptr
    std::vector<double>* dual_out);           // output: can be nullptr