#include "fair_Lloyd.h"
#include <iostream>

int gcd(int a, int b) {
    while (b != 0) {
        int temp = a % b;
        a = b;
        b = temp;
    }
    return a;
}

// ---------------------------------------------------------------
// Simplified tau parameter for fair assignment model.
//
// The tau-ratio constraint requires each cluster k to get at least
//   ceil(tau_g * |X_g|)  points from group g, where tau_g = rho / K.
//
// Since the LHS (a point count) is integral, we ceil the fractional
// RHS without changing the feasible set.  This function computes
//   f_g = target_num / |X_g|,   target_num integer,
// such that:
//   1. f_g >= tau_g  (ceil — never weakens fairness),
//   2. K * target_num <= |X_g|  (K clusters can simultaneously
//      satisfy the constraint without exceeding group total),
//   3. The model constraint  sum_{i in X_g} x_{i,k} >= |X_g| * f_g
//      has integral RHS (= target_num), preserving total unimodularity
//      so the LP optimum is integer-valued.
//
// When the tightest setting would be infeasible (K * ceil > |X_g|),
// we decrement to the largest feasible integer — this is the tightest
// constraint still admitting a K-cluster assignment.
// ---------------------------------------------------------------
double find_simplified_fraction_Tau(
    int numerator,
    int K,
    double target_factor){
    // numerator * target_factor rounded up, but must satisfy K * target_num <= numerator
    double scaled_num = numerator * target_factor;
    double integral_part;
    double fractional_part = std::modf(scaled_num, &integral_part);
    double adjusted_factor = target_factor;
    // Case 1: scaled_num is (nearly) integer — use directly if feasible
    if (fractional_part < 0.000001 || fractional_part > 0.999999)
    {
        int target_num = static_cast<int>(std::round(scaled_num));
        adjusted_factor = double(target_num) / double(numerator);
        if (target_num * K <= numerator)
        {
            std::cout << "Found direct simplification with exact factor:\n";
            std::cout << "Original group number: " << numerator << "\n";
            std::cout << "Original minimum group number: " << scaled_num << "\n";
            std::cout << "Target factor: " << target_factor << "\n";
            std::cout << "Adjusted factor: " << adjusted_factor << "\n";
            std::cout << "Minimum group number required: " << target_num << "\n";
            return adjusted_factor;
        }
    }
    else{
        // Case 2: ceil and check feasibility for K clusters
        int target_num = static_cast<int>(std::ceil(scaled_num));
        adjusted_factor = double(target_num) / double(numerator);
        if (target_num * K <= numerator)
        {
            std::cout << "Found increased factor:\n";
            std::cout << "Original group number: " << numerator << "\n";
            std::cout << "Original minimum group number: " << scaled_num << "\n";
            std::cout << "Target factor: " << target_factor << "\n";
            std::cout << "Adjusted factor: " << adjusted_factor << "\n";
            std::cout << "Minimum group number required: " << target_num << "\n";
            return adjusted_factor;
        }
        else{
            // Case 3: ceil too tight — decrement to largest feasible integer
            while (target_num > 0)
            {
                target_num--;
                adjusted_factor = double(target_num) / double(numerator);
                if (target_num * K <= numerator)
                {
                    std::cout << "Found simplification (minimum group number decreased):\n";
                    std::cout << "Original group number: " << numerator << "\n";
                    std::cout << "Original minimum group number: " << scaled_num << "\n";
                    std::cout << "Target factor: " << target_factor << "\n";
                    std::cout << "Adjusted factor: " << adjusted_factor << "\n";
                    std::cout << "Minimum group number required: " << target_num << "\n";
                    return adjusted_factor;
                }
            }
        }
    }
    std::cout << "No valid tau simplification found, returning original factor.\n";
    return target_factor; 
}

// Compute integral-feasible tau parameters per group (see find_simplified_fraction_Tau above).
std::vector<double> tau_fairParam_adjustment(const std::vector<int> &groupRatio, double fairness_param, int N, int K)
{
    std::vector<double> adjusted_factors(groupRatio.size(), 1.0);
    for (int g = 0; g < groupRatio.size(); g++)
    {
        int numerator = groupRatio[g];
        double target_factor = fairness_param;
        double adjusted_factor = find_simplified_fraction_Tau(numerator, K, target_factor);
        adjusted_factors[g] = adjusted_factor;
    }
    return adjusted_factors;
}

std::pair<std::unique_ptr<GRBModel>, std::vector<std::vector<GRBVar>>> GRB_buildFairAssignmentModel(GRBEnv &env, int numClusters,
                                                                                                    int numGroups,
                                                                                                    const std::vector<std::vector<bool>> &dataGroups,
                                                                                                    const std::vector<int> &groupRatio,
                                                                                                    std::vector<double> fairness_param)
{
    int N = dataGroups.size();
    // Create an empty model
    std::unique_ptr<GRBModel> model = std::make_unique<GRBModel>(env);
    // Create variables x_{i,k}
    std::vector<std::vector<GRBVar>> x(N, std::vector<GRBVar>(numClusters));

    try
    {

        for (int i = 0; i < N; ++i)
        {
            for (int k = 0; k < numClusters; ++k)
            {
                x[i][k] = model->addVar(0.0, 1.0, 0.0, GRB_BINARY,
                                        "x_" + std::to_string(i) + "_" + std::to_string(k));
            }
        }

        // Assignment constraints: each data point assigned to exactly one cluster
        for (int i = 0; i < N; ++i)
        {
            GRBLinExpr sum_xik = 0.0;
            for (int k = 0; k < numClusters; ++k)
            {
                sum_xik += x[i][k];
            }
            model->addConstr(sum_xik == 1, "assign_" + std::to_string(i));
        }

        std::vector<double> normalized_groupRatio(numGroups, 0.0);

        for (int g = 0; g < numGroups; g++)
        {
            normalized_groupRatio[g] = double(groupRatio[g]) / double(N);
            //std::cout<<"ratio: "<<normalized_groupRatio[g]<<std::endl;
        }

        // Fairness constraints
        for (int k = 0; k < numClusters; ++k)
        {
            // Compute sum_{i=1}^N x_{i,k}
            GRBLinExpr sum_xik = 0.0;
            for (int i = 0; i < N; ++i)
            {
                sum_xik += x[i][k];
            }

            model->addConstr(sum_xik >= 1); // make sure we have k clusters

            for (int g = 0; g < numGroups; ++g)
            {
                // sum_{i in group g} x_{i,k}
                GRBLinExpr sum_xikg = 0.0;
                for (int i = 0; i < N; ++i)
                {
                    if (dataGroups[i][g])
                    {
                        sum_xikg += x[i][k];
                    }
                }

                if (std::abs(fairness_param[g] - 1.0) <= 1e-6)
                {
                    model->addConstr(sum_xikg == (normalized_groupRatio[g] * fairness_param[g]) * sum_xik,
                                     "fair_lb_k" + std::to_string(k) + "_g" + std::to_string(g));
                }
                else
                {
                    // Lower bound constraint
                    model->addConstr(sum_xikg >= (normalized_groupRatio[g] * fairness_param[g]) * sum_xik,
                                     "fair_lb_k" + std::to_string(k) + "_g" + std::to_string(g));
                    // Upper bound constraint
                    model->addConstr(sum_xikg <= (normalized_groupRatio[g] / fairness_param[g]) * sum_xik,
                                     "fair_ub_k" + std::to_string(k) + "_g" + std::to_string(g));
                }
            }
        }
        // Feasibility will be checked on first assignment solve
    }
    catch (GRBException e)
    {
        std::cout << "Gurobi error code: " << e.getErrorCode() << std::endl;
        std::cout << e.getMessage() << std::endl;
    }
    catch (...)
    {
        std::cout << "Unknown exception during optimization." << std::endl;
    }

    return std::make_pair(std::move(model), x);
}

std::pair<std::unique_ptr<GRBModel>, std::vector<std::vector<GRBVar>>> GRB_buildTauFairAssignmentModel(GRBEnv &env, int numClusters,
                                                                                                       int numGroups,
                                                                                                       const std::vector<std::vector<bool>> &dataGroups,
                                                                                                       const std::vector<int> &groupRatio,
                                                                                                       std::vector<double> fairness_param)
{
    int N = dataGroups.size();
    // Create an empty model
    std::unique_ptr<GRBModel> model = std::make_unique<GRBModel>(env);
    // Create variables x_{i,k}
    std::vector<std::vector<GRBVar>> x(N, std::vector<GRBVar>(numClusters));

    try
    {

        for (int i = 0; i < N; ++i)
        {
            for (int k = 0; k < numClusters; ++k)
            {
                x[i][k] = model->addVar(0.0, 1.0, 0.0, GRB_CONTINUOUS,
                                        "x_" + std::to_string(i) + "_" + std::to_string(k));
            }
        }

        // Assignment constraints: each data point assigned to exactly one cluster
        for (int i = 0; i < N; ++i)
        {
            GRBLinExpr sum_xik = 0.0;
            for (int k = 0; k < numClusters; ++k)
            {
                sum_xik += x[i][k];
            }
            model->addConstr(sum_xik == 1, "assign_" + std::to_string(i));
        }

        std::vector<double> normalized_groupRatio(numGroups, 0.0);

        for (int g = 0; g < numGroups; g++)
        {
            normalized_groupRatio[g] = double(groupRatio[g]) / double(N);
            // std::cout<<"ratio: "<<normalized_groupRatio[g]<<std::endl;
        }

        // Fairness constraints
        for (int k = 0; k < numClusters; ++k)
        {
            // Compute sum_{i=1}^N x_{i,k}
            GRBLinExpr sum_xik = 0.0;
            for (int i = 0; i < N; ++i)
            {
                sum_xik += x[i][k];
            }

            model->addConstr(sum_xik >= 1); // make sure we have k clusters

            for (int g = 0; g < numGroups; ++g)
            {
                // sum_{i in group g} x_{i,k}
                GRBLinExpr sum_xikg = 0.0;
                for (int i = 0; i < N; ++i)
                {
                    if (dataGroups[i][g])
                    {
                        sum_xikg += x[i][k];
                    }
                }

                model->addConstr(sum_xikg >= groupRatio[g] * fairness_param[g],
                                 "fair_lb_k" + std::to_string(k) + "_g" + std::to_string(g));
                // Upper bound: redundant for IP, but required for TU integrality of LP relaxation
                model->addConstr(sum_xikg <= groupRatio[g],
                                 "fair_ub_k" + std::to_string(k) + "_g" + std::to_string(g));
            }
        }
        // Feasibility verified by constraint structure (totally unimodular LP)
    }
    catch (GRBException e)
    {
        std::cout << "Gurobi error code: " << e.getErrorCode() << std::endl;
        std::cout << e.getMessage() << std::endl;
    }
    catch (...)
    {
        std::cout << "Unknown exception during optimization." << std::endl;
    }

    return std::make_pair(std::move(model), x);
}

FairAssignStatus GRB_fairAssignClusters(const VectorXdList &dataPoints, VectorXdList &centroids, std::vector<int> &assignment,
                            GRBModel &model, std::vector<std::vector<GRBVar>> &x)
{
    // Number of data points
    int N = dataPoints.size();

    // Number of clusters
    int numClusters = centroids.size();

    try
    {

        // Set objective function: minimize total clustering cost
        GRBLinExpr obj = 0.0;
        for (int i = 0; i < N; ++i)
        {
            for (int k = 0; k < numClusters; ++k)
            {
                double dist = (dataPoints[i] - centroids[k]).squaredNorm();
                obj += dist * x[i][k];
            }
        }
        model.setObjective(obj, GRB_MINIMIZE);

        // Optimize model
        model.optimize();

        // Check if optimal solution was found
        if (model.get(GRB_IntAttr_Status) == GRB_OPTIMAL)
        {
            bool changed = false;
            // Update assignment
            for (int i = 0; i < N; ++i)
            {
                for (int k = 0; k < numClusters; ++k)
                {
                    if (x[i][k].get(GRB_DoubleAttr_X) > 0.5)
                    {
                        if (assignment[i] != k)
                        {
                            assignment[i] = k;
                            changed = true;
                        }
                        break;
                    }
                }
            }
            return changed ? FairAssignStatus::SUCCESS : FairAssignStatus::CONVERGED;
        }
        else if (model.get(GRB_IntAttr_Status) == GRB_INF_OR_UNBD)
        {
            std::cout << "Model is infeasible or unbounded." << std::endl;
            return FairAssignStatus::UNBOUNDED;
        }
        else if (model.get(GRB_IntAttr_Status) == GRB_INFEASIBLE)
        {
            std::cout << "Model is infeasible." << std::endl;
            return FairAssignStatus::INFEASIBLE;
        }
        else
        {
            // No optimal solution found
            std::cout << "No optimal solution found." << std::endl;
            return FairAssignStatus::ERROR;
        }
    }
    catch (GRBException e)
    {
        std::cout << "Gurobi error code: " << e.getErrorCode() << std::endl;
        std::cout << e.getMessage() << std::endl;
        return FairAssignStatus::ERROR;
    }
    catch (...)
    {
        std::cout << "Unknown exception during optimization." << std::endl;
        return FairAssignStatus::ERROR;
    }
}

std::pair<double, std::vector<int>> runFairKMeans(const VectorXdList &dataPoints, int k, int maxIterations,
                                                  int random_seed, GRBModel &model, std::vector<std::vector<GRBVar>> &x)
{
    int n = dataPoints.size();
    // Initialize centroids with k-means++
    VectorXdList centroids = initializeCentroidsPlusPlus(dataPoints, k, random_seed);
    std::vector<int> assignment(n, -1);
    
    // Start with a FAIR assignment
    FairAssignStatus init_status = GRB_fairAssignClusters(dataPoints, centroids, assignment, model, x);
    if (init_status == FairAssignStatus::INFEASIBLE || init_status == FairAssignStatus::UNBOUNDED) {
        std::cerr << "Initial fair assignment infeasible or unbounded. Stopping." << std::endl;
        return std::make_pair(kInfinity, assignment);
    }
    double currentWCSS = computeWCSS(dataPoints, centroids, assignment);
    double prevWCSS = currentWCSS;
    int lloyd_iters = 0;

    for (int iter = 0; iter < maxIterations; ++iter)
    {
        // Update centroids from current assignment
        updateCentroids(dataPoints, centroids, assignment, k);
        // Fair assignment with new centroids
        FairAssignStatus assign_status = GRB_fairAssignClusters(dataPoints, centroids, assignment, model, x);
        if (assign_status == FairAssignStatus::INFEASIBLE || assign_status == FairAssignStatus::UNBOUNDED) {
            std::cerr << "Fair assignment became infeasible during Lloyd iteration. Stopping early." << std::endl;
            break;
        }
        if (assign_status != FairAssignStatus::SUCCESS) {
            // CONVERGED: assignment unchanged
            break;
        }
        lloyd_iters++;
        currentWCSS = computeWCSS(dataPoints, centroids, assignment);
        // Converge when objective stops decreasing
        if (currentWCSS >= prevWCSS)
        {
            break;
        }
        prevWCSS = currentWCSS;
    }

    std::cout << "  Fair Lloyd converged after " << lloyd_iters << " iterations (seed " << random_seed << "), objective " << currentWCSS << std::endl;

    return std::make_pair(currentWCSS, assignment);
}
