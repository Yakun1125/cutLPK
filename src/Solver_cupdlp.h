#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
#include <vector>
#include "Utils_Struct.h"
#include "../cupdlp/cupdlp.h"
// Define a simple structure for LPK (assuming necessary fields exist)
// struct LPK {
//     int N;  // Number of variables (assuming symmetric matrix case)
//     Eigen::SparseMatrix<double> ConsMatrix; // Constraint matrix in sparse format
//     std::vector<double> objCoef;  // Objective function coefficients
//     std::vector<double> varLb;  // Variable lower bounds
//     std::vector<double> varUb;  // Variable upper bounds
//     std::vector<double> consLb;  // Constraint lower bounds
//     std::vector<double> consUb;  // Constraint upper bounds
// };

// // Define a struct for valid inequalities (for testing)
// struct validInequality {
//     double violation;
//     double dual_value;
// };

int solver_cupdlp(double& dual_obj, double& primal_obj, Eigen::MatrixXd& Xsol, std::vector<validInequality>& cuts, LPK& lp, float tolerance, float time_limit);