#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <string>
#include <vector>
#include <cmath>

#include <Eigen/Dense>

#include "ordinary_kmeans_solver.h"
#include "Lloyd.h"
#include "fair_kmeans_solver.h"
#include "spectral_kmeans_solver.h"
#include "spectral_heuristic.h"

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
    // params.cutting_plane_output_level = 0;
    // params.bnb_output_level = 0;
    // params.bnb_node_limit = 0;


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
    assignInt("max_cuts_firstLP", params.cutting_plane_max_cuts_firstLP);
    assignInt("max_cuts_per_iter", params.cutting_plane_max_cuts_per_iter);
    assignInt("max_cuts_added_iter", params.cutting_plane_max_cuts_added_iter);
    assignInt("max_cuts_separation_size", params.cutting_plane_max_cuts_separation_size);
    assignInt("max_active_cuts_size", params.cutting_plane_max_active_cuts_size);
    assignInt("num_iter_no_improve", params.cutting_plane_num_iter_no_improve);


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
    assignString("cutting_plane_output_file", params.cutting_plane_output_file);
    assignString("bnb_output_file", params.bnb_output_file);


    return params;
}

VectorXdList toEigenPoints(const py::array_t<double, py::array::c_style | py::array::forcecast>& data) {
    if (data.ndim() != 2) {
        throw py::value_error("Input data must be a 2D array");
    }

    const ssize_t num_points = data.shape(0);
    const ssize_t dimension = data.shape(1);

    auto buf = data.unchecked<2>();
    VectorXdList points;
    points.reserve(static_cast<size_t>(num_points));

    for (ssize_t i = 0; i < num_points; ++i) {
        points.emplace_back(static_cast<int>(dimension));
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
    params.cutting_plane_t_upper_bound = n_clusters;

    VectorXdList points = toEigenPoints(data);

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

py::dict runFairKMeans(py::array_t<double, py::array::c_style | py::array::forcecast> data,
                       int n_clusters,
                       py::object groups,
                       const py::kwargs& kwargs) {
    if (n_clusters <= 0) {
        throw py::value_error("n_clusters must be positive");
    }

    parameters params = buildParameters(kwargs);
    params.cutting_plane_t_upper_bound = n_clusters;

    // Set fairness parameters
    if (kwargs.contains("fair_clustering_fairness_type")) {
        params.fair_clustering_fairness_type = kwargs["fair_clustering_fairness_type"].cast<std::string>();
    } else {
        params.fair_clustering_fairness_type = "tau";  // default
    }
    
    if (kwargs.contains("fair_clustering_fairness_param")) {
        params.fair_clustering_fairness_param = kwargs["fair_clustering_fairness_param"].cast<double>();
    } else {
        params.fair_clustering_fairness_param = 0.1;  // default
    }

    VectorXdList points = toEigenPoints(data);

    // Accept group labels as either numpy array or list
    std::vector<std::vector<bool>> dataGroups;
    if (py::isinstance<py::array>(groups)) {
        py::array group_arr = groups.cast<py::array>();
        if (group_arr.ndim() == 1) {
            // group labels, shape (n_points,)
            ssize_t n_points = group_arr.shape(0);
            std::vector<int> raw_labels(n_points);
            auto buf = group_arr.unchecked<int, 1>();
            for (ssize_t i = 0; i < n_points; ++i) {
                raw_labels[i] = buf(i);
            }
            // Map unique labels to consecutive indices
            std::unordered_map<int, int> label_map;
            int next_idx = 0;
            std::vector<int> mapped_labels(n_points);
            for (ssize_t i = 0; i < n_points; ++i) {
                int label = raw_labels[i];
                if (label_map.find(label) == label_map.end()) {
                    label_map[label] = next_idx++;
                }
                mapped_labels[i] = label_map[label];
            }
            int n_groups = next_idx;
            dataGroups.resize(n_points, std::vector<bool>(n_groups, false));
            for (ssize_t i = 0; i < n_points; ++i) {
                dataGroups[i][mapped_labels[i]] = true;
            }
        } else if (group_arr.ndim() == 2) {
            // boolean matrix, shape (n_points, n_groups)
            ssize_t n_points = group_arr.shape(0);
            ssize_t n_groups = group_arr.shape(1);
            auto buf = group_arr.unchecked<bool, 2>();
            dataGroups.resize(n_points);  // N vectors (one per point)
            for (ssize_t i = 0; i < n_points; ++i) {
                dataGroups[i].resize(n_groups);  // Each point has num_groups membership flags
                for (ssize_t g = 0; g < n_groups; ++g) {
                    dataGroups[i][g] = buf(i, g);  // dataGroups[i][g] = point i belongs to group g
                }
            }
        } else {
            throw py::value_error("groups array must be 1D (labels) or 2D (boolean matrix)");
        }
    } else if (py::isinstance<py::list>(groups)) {
        // Try to interpret as boolean matrix (list of lists), shape (n_points, n_groups)
        py::list group_list = groups.cast<py::list>();
        ssize_t n_points = group_list.size();
        if (n_points == 0) {
            throw py::value_error("groups list cannot be empty");
        }
        ssize_t n_groups = group_list[0].cast<py::list>().size();
        
        // Keep in (n_points, n_groups) format
        dataGroups.resize(n_points);
        for (ssize_t i = 0; i < n_points; ++i) {
            py::list point_groups = group_list[i].cast<py::list>();
            if (point_groups.size() != static_cast<size_t>(n_groups)) {
                throw py::value_error("All points must have the same number of groups");
            }
            dataGroups[i].resize(n_groups);
            for (ssize_t g = 0; g < n_groups; ++g) {
                dataGroups[i][g] = point_groups[g].cast<bool>();
            }
        }
    } else {
        throw py::value_error("groups must be a numpy array or list");
    }

    // Compute ratios from dataGroups - dataGroups is (n_points, n_groups)
    size_t n_groups_total = dataGroups.empty() ? 0 : dataGroups[0].size();
    std::vector<int> groupRatio(n_groups_total, 0);
    for (size_t i = 0; i < dataGroups.size(); ++i) {  // for each point
        for (size_t g = 0; g < n_groups_total; ++g) {  // for each group
            if (dataGroups[i][g]) groupRatio[g]++;
        }
    }

    FairKMeansResult result;
    try {
        result = solveFairKMeans(points, n_clusters, dataGroups, groupRatio, params);
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

py::dict runSpectralKMeans(py::array_t<double, py::array::c_style | py::array::forcecast> laplacian,
                           int n_clusters,
                           const py::kwargs& kwargs) {
    if (n_clusters <= 0) {
        throw py::value_error("n_clusters must be positive");
    }

    parameters params = buildParameters(kwargs);
    params.cutting_plane_t_upper_bound = n_clusters;
    params.is_spectral_clustering = true;

    // Convert laplacian to Eigen::MatrixXd
    if (laplacian.ndim() != 2) {
        throw py::value_error("Laplacian must be a 2D array");
    }

    const ssize_t n = laplacian.shape(0);
    const ssize_t m = laplacian.shape(1);
    if (n != m) {
        throw py::value_error("Laplacian must be square");
    }

    auto buf = laplacian.unchecked<2>();
    Eigen::MatrixXd L(n, m);
    for (ssize_t i = 0; i < n; ++i) {
        for (ssize_t j = 0; j < m; ++j) {
            L(i, j) = buf(i, j);
        }
    }

    SpectralKMeansResult result;
    try {
        result = solveSpectralKMeans(L, n_clusters, params);
    } catch (const std::exception& ex) {
        throw py::value_error(ex.what());
    }

    // Extract scalar values
    py::dict output;
    output["cost"] = result.cut_info.upper_bound;
    output["relative_gap"] = result.cut_info.optimality_gap;
    output["lower_bound"] = result.cut_info.lower_bound;
    output["upper_bound"] = result.cut_info.upper_bound;
    output["warm_start_cost"] = result.spectral_objective;
    output["status"] = toString(result.icp_status);
    output["retcode"] = result.cut_info.retcode;
    output["bnb_executed"] = result.bnb_executed;

    if (result.bnb_executed) {
        output["bnb_status"] = toString(result.bnb_status);
    } else {
        output["bnb_status"] = py::none();
    }


    output["labels"] = py::none(); 

    return output;
}

}  // namespace

PYBIND11_MODULE(_cutlpk, m) {
    m.doc() = "Python bindings for the cutLPK clustering solvers";
    m.def(
        "run_ordinary_kmeans",
        &runOrdinaryKMeans,
        py::arg("data"),
        py::arg("n_clusters"),
        "Run the iterative cutting-plane solver for ordinary k-means clustering."
    );
    m.def(
        "run_fair_kmeans",
        &runFairKMeans,
        py::arg("data"),
        py::arg("n_clusters"),
        py::arg("groups"),
        "Run the iterative cutting-plane solver for fair k-means clustering."
    );
    m.def(
        "run_spectral_kmeans",
        &runSpectralKMeans,
        py::arg("laplacian"),
        py::arg("n_clusters"),
        "Run the iterative cutting-plane solver for spectral k-means clustering."
    );
}