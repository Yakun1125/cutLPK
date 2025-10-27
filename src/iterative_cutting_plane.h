#pragma once
#include "Utils_Struct.h"
#include "Rounding_heuristic.h"

ICPStatus iterative_cutting_plane_solver(
    int N,
    int K,
    cutLPKSolveInfo& cutLPKInfor, 
    std::vector<validInequality>& cutting_planes, 
    LPK& lp, 
    RoundingHeuristic& roundingHeuristic,
    const parameters& params,
    const std::vector<double>* primal_init = nullptr,  // Initial primal solution for warm start
    const std::vector<double>* dual_init = nullptr     // Initial dual solution for warm start
);