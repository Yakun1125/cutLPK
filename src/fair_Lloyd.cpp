#include "fair_Lloyd.h"
#include <fstream>
#include <iostream>

void setupGurobiWLS(GRBEnv& env) {
    const std::string filename = "gurobiWLS.txt";
    std::ifstream wls_file(filename);

    if (wls_file.is_open()) {
        std::string line;
        std::string accessID, secret, licenseID;

        while (std::getline(wls_file, line)) {
            auto pos = line.find('=');
            if (pos == std::string::npos) continue;

            std::string key = line.substr(0, pos);
            std::string value = line.substr(pos + 1);

            if (key == "WLSACCESSID") accessID = value;
            else if (key == "WLSSECRET") secret = value;
            else if (key == "LICENSEID") licenseID = value;
        }

        if (!accessID.empty() && !secret.empty() && !licenseID.empty()) {
            try {
                env.set("WLSACCESSID", accessID);
                env.set("WLSSECRET", secret);
                env.set("LICENSEID", licenseID);
            } catch (GRBException& e) {
                std::cerr << "Error setting Gurobi WLS parameters: " << e.getMessage() << std::endl;
            }
        } 
    } else {
        std::cout << filename << " not found. Proceeding without WLS setup from file." << std::endl;
    }
}

int gcd(int a, int b) {
    while (b != 0) {
        int temp = a % b;
        a = b;
        b = temp;
    }
    return a;
}

double find_simplified_fraction(
    int numerator,
    int denominator,
    double target_factor)
{

    // First try direct simplification
    // We need to handle floating point multiplication by scaling up to integers

    // Determine if the target numerator is (nearly) an integer
    double scaled_num = numerator * target_factor;
    double integral_part;
    double fractional_part = std::modf(scaled_num, &integral_part);
    double adjusted_factor = target_factor;

    // Need to scale up to work with integers
    int scale = 10;
    while (std::fmod(scaled_num * scale, 1.0) > 0.000001 && scale < 1000000)
    {
        scale *= 10;
    }

    int scaled_num_int = static_cast<int>(std::round(scaled_num * scale));
    int scaled_denom = denominator * scale;

    // Find GCD and simplify
    int common_div = gcd(scaled_num_int, scaled_denom);
    int simplified_n = scaled_num_int / common_div;
    int simplified_d = scaled_denom / common_div;

    adjusted_factor = double(simplified_n) / double(simplified_d);
    adjusted_factor = adjusted_factor/(double(numerator) / double(denominator));

    // Check if this simplification meets our criteria
    if (simplified_d < denominator)
    {
        std::cout << "Found direct simplification with exact factor:\n";
        std::cout << "Original fraction: " << numerator << "/" << denominator << "\n";
        std::cout << "Target factor: " << target_factor << "\n";
        std::cout << "Adjusted factor: " << adjusted_factor << "\n";
        std::cout << "Scaled to integers: " << scaled_num_int << "/" << scaled_denom << "\n";
        std::cout << "Simplified to: " << simplified_n << "/" << simplified_d << "\n";
        return adjusted_factor;
    }
    // Replace the while loop section (lines 58-79) with this:
    else
    {
        // Parameters for tolerance-based search
        double denominator_tolerance = 0.2;  // Allow 20% change in denominator
        double factor_tolerance = 0.1;       // Allow 10% change in factor
        
        double target_value = (numerator * target_factor) / denominator;
        
        int min_denominator = static_cast<int>(denominator * (1 - denominator_tolerance));
        int max_denominator = static_cast<int>(denominator * (1 + denominator_tolerance));
        
        double min_factor = target_factor;  // Must be >= target_factor for "larger" preference
        double max_factor = target_factor * (1 + factor_tolerance);
        
        double best_diff = std::numeric_limits<double>::infinity();
        double best_adjusted_factor = target_factor;
        bool found = false;
        
        // Search through denominator range
        for (int d = min_denominator; d <= max_denominator; ++d) {
            if (d <= 0) continue;  // Skip invalid denominators
            
            double target_numerator = target_value * d;
            
            // Try both floor and ceiling of target numerator
            for (int n : {static_cast<int>(target_numerator), static_cast<int>(target_numerator) + 1}) {
                if (n <= 0) continue;  // Skip invalid numerators
                
                // Calculate what this fraction represents in terms of the original
                double fraction_value = static_cast<double>(n) / d;
                double actual_factor = fraction_value / (static_cast<double>(numerator) / denominator);
                
                // Check if factor is in acceptable range (must be >= target_factor for "larger")
                if (actual_factor < min_factor || actual_factor > max_factor) continue;
                
                // Simplify the fraction
                int common_divisor = gcd(n, d);
                int simplified_n = n / common_divisor;
                int simplified_d = d / common_divisor;
                
                // Ensure simplified denominator is smaller than original
                if (simplified_d >= denominator) continue;
                
                // For "larger" preference: prefer values >= target, minimize difference
                double current_value = static_cast<double>(simplified_n) / simplified_d;
                double original_value = static_cast<double>(numerator) / denominator;
                double scaled_current = current_value / original_value;  // This is the actual scaling factor
                
                if (scaled_current >= target_factor) {  // "larger" preference
                    double diff = scaled_current - target_factor;  // How much larger than target
                    
                    if (diff < best_diff) {
                        best_diff = diff;
                        best_adjusted_factor = scaled_current;
                        found = true;
                        
                        std::cout << "Found better simplification:\n";
                        std::cout << "Original fraction: " << numerator << "/" << denominator << "\n";
                        std::cout << "Target factor: " << target_factor << "\n";
                        std::cout << "Adjusted factor: " << best_adjusted_factor << "\n";
                        std::cout << "Fraction found: " << n << "/" << d << "\n";
                        std::cout << "Simplified to: " << simplified_n << "/" << simplified_d << "\n";
                        std::cout << "Difference from target: " << diff << "\n";
                    }
                }
            }
        }
        
        if (found) {
            return best_adjusted_factor;
        }
        
        // Fallback: if no solution found in tolerance range, try the original approach
        // (your existing while loop as backup)
        scaled_denom = denominator * scale;  // Reset to original values
        while (scaled_denom > scaled_num_int && scaled_denom > 1) {
            scaled_denom--;
            int common_div = gcd(scaled_num_int, scaled_denom);
            int simplified_n = scaled_num_int / common_div;
            int simplified_d = scaled_denom / common_div;
            
            if (simplified_d < denominator) {
                adjusted_factor = double(simplified_n) / double(simplified_d);
                adjusted_factor = adjusted_factor / (double(numerator) / double(denominator));
                
                if (adjusted_factor >= target_factor) {  // Ensure it's "larger"
                    std::cout << "Found fallback simplification:\n";
                    std::cout << "Original fraction: " << numerator << "/" << denominator << "\n";
                    std::cout << "Target factor: " << target_factor << "\n";
                    std::cout << "Adjusted factor: " << adjusted_factor << "\n";
                    std::cout << "Simplified to: " << simplified_n << "/" << simplified_d << "\n";
                    return adjusted_factor;
                }
            }
        }
    }
    std::cout << "No valid tau simplification found, returning original factor.\n";
    return adjusted_factor; 
}

double find_simplified_fraction_Tau(
    int numerator,
    int K,
    double target_factor){
        // check numerator * target_factor is an integer or not. And (numerator * target_factor)*K should be less or equal to numerator
    double scaled_num = numerator * target_factor;
    double integral_part;
    double fractional_part = std::modf(scaled_num, &integral_part);
    double adjusted_factor = target_factor;
    // if scaled_num is an integer, we can use it directly
    if (fractional_part < 0.000001 || fractional_part > 0.999999)
    {
        int target_num = static_cast<int>(std::round(scaled_num));
                // recompute actual factor
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
        // if it is not an integer, we upper round it to to see if it is feasible
        int target_num = static_cast<int>(std::ceil(scaled_num));
        // recompute actual factor
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
            // decrease target_num 1 at a time to find a simplification meeting the criteria
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

std::vector<double> alpha_fairParam_adjustment(const std::vector<int> &groupRatio, double fairness_param, int N, int K)
{
    std::vector<double> adjusted_factors(groupRatio.size(), 1.0);
    
    for (int g = 0; g < groupRatio.size(); g++)
    {
        int C = groupRatio[g];  // numerator (group size)
        int D = N;              // denominator (total size)
        double alpha = fairness_param;
        
        // Check if the problem is feasible at all
        // We need A*K <= C and B*K <= D and A/B >= (C*alpha)/D
        // The maximum possible A/B is (C/K)/(1) = C/K (when A = C/K, B = 1)
        // The minimum required A/B is (C*alpha)/D
        double max_possible_ratio = static_cast<double>(C) / K;
        double min_required_ratio = (static_cast<double>(C) * alpha) / D;
        
        if (max_possible_ratio < min_required_ratio) {
            std::cerr << "ERROR: Infeasible fairness parameter for group " << g << "!" << std::endl;
            std::cerr << "Group size: " << C << ", Total size: " << D << ", K: " << K << std::endl;
            std::cerr << "Fairness parameter (alpha): " << alpha << std::endl;
            std::cerr << "Maximum achievable ratio: " << max_possible_ratio << std::endl;
            std::cerr << "Minimum required ratio: " << min_required_ratio << std::endl;
            std::cerr << "Suggestion: Reduce alpha to at most " << (max_possible_ratio * D) / C << std::endl;
            
            // Use the maximum achievable alpha as fallback
            double max_achievable_alpha = (max_possible_ratio * D) / C;
            std::cerr << "Using fallback alpha = " << max_achievable_alpha << " for group " << g << std::endl;
            alpha = max_achievable_alpha;
            min_required_ratio = (static_cast<double>(C) * alpha) / D;
        }
        
        // Now find the optimal A/B
        double target_ratio = (C * alpha) / D;
        
        int best_A = -1, best_B = -1;
        double min_diff = std::numeric_limits<double>::infinity();
        
        // Iterate through all possible values of B
        int max_B = D / K;  // From constraint B*K <= D
        
        bool found_solution = false;
        for (int B = 1; B <= max_B; ++B) {
            // From constraint A/B >= (C*alpha)/D, we get A >= B * (C*alpha)/D
            double min_A_exact = B * target_ratio;
            int min_A = static_cast<int>(std::ceil(min_A_exact));
            
            // From constraint A*K <= C, we get A <= C/K
            int max_A = C / K;
            
            // Check if there's a valid A for this B
            if (min_A <= max_A) {
                found_solution = true;
                
                // Choose the A that minimizes |A/B - target_ratio|
                double ratio_min_A = static_cast<double>(min_A) / B;
                double diff_min_A = std::abs(ratio_min_A - target_ratio);
                
                if (diff_min_A < min_diff) {
                    min_diff = diff_min_A;
                    best_A = min_A;
                    best_B = B;
                }
                
                // Also check max_A in case it gives a smaller difference
                if (max_A > min_A) {
                    double ratio_max_A = static_cast<double>(max_A) / B;
                    double diff_max_A = std::abs(ratio_max_A - target_ratio);
                    
                    if (diff_max_A < min_diff) {
                        min_diff = diff_max_A;
                        best_A = max_A;
                        best_B = B;
                    }
                }
            }
        }
        
        if (!found_solution || best_A == -1) {
            std::cerr << "ERROR: No valid solution found for group " << g << "!" << std::endl;
            std::cerr << "Using original fairness parameter as fallback." << std::endl;
            adjusted_factors[g] = fairness_param;
        } else {
            // Calculate the adjusted factor
            double optimal_ratio = static_cast<double>(best_A) / best_B;
            double original_ratio = static_cast<double>(C) / D;
            adjusted_factors[g] = optimal_ratio / original_ratio;
            
            // std::cout << "Group " << g << " fairness adjustment:\n";
            // std::cout << "  Original group ratio: " << C << "/" << D << " = " << original_ratio << "\n";
            // std::cout << "  Target fairness factor: " << fairness_param << "\n";
            // std::cout << "  Optimal A/B: " << best_A << "/" << best_B << " = " << optimal_ratio << "\n";
            // std::cout << "  Adjusted fairness factor: " << adjusted_factors[g] << "\n";
            // std::cout << "  Constraint checks:\n";
            // std::cout << "    A*K <= C: " << best_A << "*" << K << " = " << best_A*K << " <= " << C << " ✓\n";
            // std::cout << "    B*K <= D: " << best_B << "*" << K << " = " << best_B*K << " <= " << D << " ✓\n";
            // std::cout << "    A/B >= target: " << optimal_ratio << " >= " << target_ratio << " ✓\n";
            // std::cout << "    Difference from target: " << min_diff << "\n\n";
        }
    }
    
    return adjusted_factors;
}

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
        model->optimize();
        if (model->get(GRB_IntAttr_Status) == GRB_INFEASIBLE)
        {
            std::cout << "Assignment problem is infeasible." << std::endl;
            return std::make_pair(std::unique_ptr<GRBModel>(nullptr),
                                  std::vector<std::vector<GRBVar>>());
        }
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
            }
        }
        model->optimize();
        if (model->get(GRB_IntAttr_Status) == GRB_INFEASIBLE)
        {
            std::cout << "Assignment problem is infeasible." << std::endl;
            return std::make_pair(std::unique_ptr<GRBModel>(nullptr),
                                  std::vector<std::vector<GRBVar>>());
        }
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

FairAssignStatus GRB_fairAssignClusters(const std::vector<Eigen::VectorXd> &dataPoints, std::vector<Eigen::VectorXd> &centroids, std::vector<int> &assignment,
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

std::pair<double, std::vector<int>> runFairKMeans(const std::vector<Eigen::VectorXd> &dataPoints, int k, int maxIterations,
                                                  int random_seed, GRBModel &model, std::vector<std::vector<GRBVar>> &x)
{
    int n = dataPoints.size();
    // initialized centroids and do assignment
    std::vector<Eigen::VectorXd> centroids = initializeCentroidsPlusPlus(dataPoints, k, random_seed);
    std::vector<int> assignment(n, -1);
    bool changed = assignClusters(dataPoints, centroids, assignment);
    double currentWCSS = computeWCSS(dataPoints, centroids, assignment);

    for (int iter = 0; iter < maxIterations; ++iter)
    {
        //std::cout << "Fair Lloyd iteration " << iter + 1 << std::endl;
        if (!changed)
        {
            break;
        }
        std::vector<Eigen::VectorXd> oldCentroids = centroids;
        updateCentroids(dataPoints, centroids, assignment, k);
        currentWCSS = computeWCSS(dataPoints, centroids, assignment);
        // Compute centroid shift
        double centroidShiftSquared = 0.0;
        double centroidNormSquared = 0.0;

        for (size_t i = 0; i < centroids.size(); ++i)
        {
            centroidShiftSquared += (centroids[i] - oldCentroids[i]).squaredNorm();
            centroidNormSquared += centroids[i].squaredNorm();
        }

        // Check relative centroid shift
        if ((centroidNormSquared > 0 &&
             centroidShiftSquared <= 1e-6 * centroidNormSquared))
        {
            break;
        }
        // if new centroids shifted, do assignment
        FairAssignStatus assign_status = GRB_fairAssignClusters(dataPoints, centroids, assignment, model, x);
        if (assign_status == FairAssignStatus::INFEASIBLE || assign_status == FairAssignStatus::UNBOUNDED) {
            std::cerr << "Fair assignment became infeasible during Lloyd iteration. Stopping early." << std::endl;
            break;
        }
        changed = (assign_status == FairAssignStatus::SUCCESS);
    }

    // Eigen::MatrixXd PartitionMatrix = createPartitionMatrix(assignment, k);

    return std::make_pair(currentWCSS, assignment);
}
