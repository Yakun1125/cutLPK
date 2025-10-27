//#include "gurobi_c++.h"
#include <Eigen/Dense>
#include "Utils_Struct.h"

int solver_gurobi(double& dual_obj, double& primal_obj, Eigen::MatrixXd& Xsol, std::vector<validInequality>& cuts, LPK& lp, float tolerance, float time_limit);