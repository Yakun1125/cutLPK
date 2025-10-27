#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <string>
#include <vector>

#include <Eigen/Dense>

#include "ordinary_kmeans_solver.h"

namespace py = pybind11;

namespace {

std::string toString(ICPStatus status) {
    switch (status) {
        case ICPStatus::SUCCESS: return "success";
        case ICPStatus::NO_VIOLATED_CUTS: return "no_violated_cuts";
        case ICPStatus::NO_IMPROVEMENT: return "no_improvement";
        case ICPStatus::TIME_OR_LIMIT: return "time_or_limit";
        case ICPStatus::ERROR: return "error";
        case ICPStatus::MAX_ITER: return "max_iter";
        case ICPStatus::INFEASIBLE: return "infeasible";
    }
    return "unknown";
}

std::string toString(BnBStatus status) {
    switch (status) {
        case BnBStatus::OPTIMAL: return "optimal";
        case BnBStatus::NODE_LIMIT: return "node_limit";
        case BnBStatus::TIME_LIMIT: return "time_limit";
        case BnBStatus::ALL_NODES_EXPLORED: return "all_nodes_explored";
        case BnBStatus::ERROR: return "error";
    }
    return "unknown";
}

parameters buildParameters(const py::kwargs& kwargs) {
    parameters params;
    params.cutting_plane_output_file.clear();
    params.bnb_output_file.clear();
    params.fair_clustering_fairness_type.clear();
    params.is_spectral_clustering = false;
    params.cutting_plane_output_level = 0;
    params.bnb_output_level = 0;
    params.bnb_node_limit = 0;

    auto assignInt = [&](const char* key, int& field) {
        if (kwargs.contains(key)) {
            field = kwargs[key].cast<int>();
        }
    };
    auto assignDouble = [&](const char* key, double& field) {
        if (kwargs.contains(key)) {
            field = kwargs[key].cast<double>();
        }
    };
    auto assignBoolToInt = [&](const char* key, int& field) {
        if (kwargs.contains(key)) {
            field = kwargs[key].cast<bool>() ? 1 : 0;
        }
    };
    auto assignString = [&](const char* key, std::string& field) {
        if (kwargs.contains(key)) {
            field = kwargs[key].cast<std::string>();
        }
    };

    assignBoolToInt("warm_start", params.cutting_plane_warm_start);
    assignInt("lloyd_random_starts", params.lloyd_num_random_starts);
    assignInt("random_seed", params.random_seed);
    assignInt("max_iter", params.cutting_plane_max_iter);
    assignInt("output_level", params.cutting_plane_output_level);
    assignInt("bnb_node_limit", params.bnb_node_limit);
    assignInt("bnb_verbose", params.bnb_verbose);
    assignInt("bnb_output_level", params.bnb_output_level);

    assignDouble("time_limit_all", params.cutting_plane_time_limit);
    assignDouble("time_limit", params.cutting_plane_time_limit);
    assignDouble("time_limit_lp", params.cutting_plane_LP_time_limit);
    assignDouble("initial_lp_time_limit", params.cutting_plane_firstLP_time_limit);
    assignDouble("solver_tolerance_per_iter", params.cutting_plane_solver_tol);
    assignDouble("initial_solver_tol", params.cutting_plane_firstLP_solver_tol);
    assignDouble("lb_solver_tol", params.cutting_plane_lb_solver_tol);
    assignDouble("cuts_vio_tol", params.cutting_plane_cuts_vio_tol);
    assignDouble("cuts_act_tol", params.cutting_plane_cuts_act_tol);
    assignDouble("opt_gap", params.cutting_plane_opt_gap);
    assignDouble("bnb_time_limit", params.bnb_time_limit);
    assignDouble("bnb_gap_tol", params.bnb_gap_tol);
    assignDouble("bnb_global_ub", params.bnb_global_ub);

    assignString("solver", params.solver);

    return params;
}

std::vector<Eigen::VectorXd> toEigenPoints(const py::array_t<double, py::array::c_style | py::array::forcecast>& data) {
    if (data.ndim() != 2) {
        throw py::value_error("Input data must be a 2D array");
    }

    const ssize_t num_points = data.shape(0);
    const ssize_t dimension = data.shape(1);

    auto buf = data.unchecked<2>();
    std::vector<Eigen::VectorXd> points(static_cast<size_t>(num_points), Eigen::VectorXd(static_cast<int>(dimension)));

    for (ssize_t i = 0; i < num_points; ++i) {
        for (ssize_t j = 0; j < dimension; ++j) {
            points[static_cast<size_t>(i)](static_cast<int>(j)) = buf(i, j);
        }
    }

    return points;
}

// Helper to convert std::vector to Python list (deep copy)
template<typename T>
py::list vectorToList(const std::vector<T>& vec) {
    py::list result;
    for (const auto& item : vec) {
        result.append(item);
    }
    return result;
}

py::dict runOrdinaryKMeans(py::array_t<double, py::array::c_style | py::array::forcecast> data,
                           int n_clusters,
                           const py::kwargs& kwargs) {
    if (n_clusters <= 0) {
        throw py::value_error("n_clusters must be positive");
    }

    parameters params = buildParameters(kwargs);
    std::vector<Eigen::VectorXd> points = toEigenPoints(data);

    OrdinaryKMeansResult result;
    try {
        result = solveOrdinaryKMeans(points, n_clusters, params);
    } catch (const std::exception& ex) {
        throw py::value_error(ex.what());
    }

    // Extract scalar values
    py::dict output;
    output["cost"] = result.cut_info.upper_bound;
    output["relative_gap"] = result.cut_info.optimality_gap;
    output["lower_bound"] = result.cut_info.lower_bound;
    output["upper_bound"] = result.cut_info.upper_bound;
    output["warm_start_cost"] = result.lloyd_objective;
    output["status"] = toString(result.icp_status);
    output["retcode"] = result.cut_info.retcode;
    output["bnb_executed"] = result.bnb_executed;

    if (result.bnb_executed) {
        output["bnb_status"] = toString(result.bnb_status);
    } else {
        output["bnb_status"] = py::none();
    }

    if (!result.assignment.empty()) {
        output["labels"] = vectorToList(result.assignment);
    } else {
        output["labels"] = py::none();
    }

    return output;
}

}  // namespace

PYBIND11_MODULE(_cutlpk, m) {
    m.doc() = "Python bindings for the cutLPK ordinary k-means solver";
    m.def(
        "run_ordinary_kmeans",
        &runOrdinaryKMeans,
        py::arg("data"),
        py::arg("n_clusters"),
        "Run the iterative cutting-plane solver for ordinary k-means clustering."
    );
}
