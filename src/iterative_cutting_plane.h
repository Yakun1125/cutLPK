#pragma once
#include "Utils_Struct.h"
#include "Rounding_heuristic.h"

// build a class of cutting plane solver, it get's an initial LP and then performs solve the LP and separate the cuts. Final info returns final LP, lower bound, upper bound, gap, and best upper bound solution

int iterative_cutting_plane_solver(
    int N,
    int K,
    cutLPKSolveInfo& cutLPKInfor, 
    std::vector<validInequality>& cutting_planes, 
    LPK& lp, 
    RoundingHeuristic& roundingHeuristic,
    const parameters& params
);