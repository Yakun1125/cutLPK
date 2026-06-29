#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>

#include "cutLPK.h"
#include "Utils_Struct.h"

namespace py = pybind11;

// ---------------------------------------------------------------------------
// Helper: convert a Python dict to a C++ parameters struct
// ---------------------------------------------------------------------------
static parameters dict_to_params(py::dict d, int K) {
    parameters p;  // start with all defaults

    // --- Solver ---
    if (d.contains("solver"))              p.solver = py::str(d["solver"]);
    if (d.contains("solver_warm_start"))   p.solver_warm_start = py::bool_(d["solver_warm_start"]);

    // --- Cutting plane: cut management ---
    if (d.contains("max_cuts_init"))           p.cutting_plane_max_cuts_firstLP = py::int_(d["max_cuts_init"]);
    if (d.contains("max_cuts_per_iter"))       p.cutting_plane_max_cuts_per_iter = py::int_(d["max_cuts_per_iter"]);
    if (d.contains("max_cuts_added_iter"))     p.cutting_plane_max_cuts_added_iter = py::int_(d["max_cuts_added_iter"]);
    if (d.contains("max_separation_size"))     p.cutting_plane_max_cuts_separation_size = py::int_(d["max_separation_size"]);
    if (d.contains("max_active_cuts_size"))    p.cutting_plane_max_active_cuts_size = py::int_(d["max_active_cuts_size"]);

    // --- Cutting plane: algorithm control ---
    if (d.contains("max_iter"))                p.cutting_plane_max_iter = py::int_(d["max_iter"]);
    if (d.contains("num_iter_no_improve"))     p.cutting_plane_num_iter_no_improve = py::int_(d["num_iter_no_improve"]);
    if (d.contains("exact_separation"))        p.cutting_plane_exact_separation = py::bool_(d["exact_separation"]);
    if (d.contains("remove_inactive_cuts"))    p.cutting_plane_remove_inactive_cuts = py::bool_(d["remove_inactive_cuts"]);
    if (d.contains("warm_start"))             p.cutting_plane_warm_start = py::int_(d["warm_start"]);

    // --- Cutting plane: time limits ---
    if (d.contains("first_lp_time_limit"))     p.cutting_plane_firstLP_time_limit = py::float_(d["first_lp_time_limit"]);
    if (d.contains("lp_time_limit"))           p.cutting_plane_LP_time_limit = py::float_(d["lp_time_limit"]);
    if (d.contains("time_limit"))              p.cutting_plane_time_limit = py::float_(d["time_limit"]);
    if (d.contains("max_separation_time"))     p.cutting_plane_max_separation_time = py::float_(d["max_separation_time"]);

    // --- Cutting plane: tolerances ---
    if (d.contains("first_lp_solver_tol"))     p.cutting_plane_firstLP_solver_tol = py::float_(d["first_lp_solver_tol"]);
    if (d.contains("solver_tol"))              p.cutting_plane_solver_tol = py::float_(d["solver_tol"]);
    if (d.contains("lb_solver_tol"))           p.cutting_plane_lb_solver_tol = py::float_(d["lb_solver_tol"]);
    if (d.contains("cuts_vio_tol"))            p.cutting_plane_cuts_vio_tol = py::float_(d["cuts_vio_tol"]);
    if (d.contains("cuts_act_tol"))            p.cutting_plane_cuts_act_tol = py::float_(d["cuts_act_tol"]);
    if (d.contains("opt_gap"))                 p.cutting_plane_opt_gap = py::float_(d["opt_gap"]);

    // --- Cutting plane: output ---
    if (d.contains("output_file"))             p.cutting_plane_output_file = py::str(d["output_file"]);
    if (d.contains("verbose"))                 p.cutting_plane_verbose = py::int_(d["verbose"]);
    if (d.contains("output_level"))            p.cutting_plane_output_level = py::int_(d["output_level"]);

    // --- t parameter ---
    if (d.contains("t_upper_bound"))
        p.cutting_plane_t_upper_bound = py::int_(d["t_upper_bound"]);
    else
        p.cutting_plane_t_upper_bound = K;  // default to K

    // --- Heuristic ---
    if (d.contains("random_seed"))             p.random_seed = py::int_(d["random_seed"]);
    if (d.contains("lloyd_random_starts"))     p.lloyd_num_random_starts = py::int_(d["lloyd_random_starts"]);
    if (d.contains("heuristic_only"))          p.heuristic_only = py::bool_(d["heuristic_only"]);

    // --- Branch and bound ---
    if (d.contains("bnb_node_limit"))          p.bnb_node_limit = py::int_(d["bnb_node_limit"]);
    if (d.contains("bnb_time_limit"))          p.bnb_time_limit = py::float_(d["bnb_time_limit"]);
    if (d.contains("bnb_gap_tol"))             p.bnb_gap_tol = py::float_(d["bnb_gap_tol"]);
    if (d.contains("bnb_global_ub"))           p.bnb_global_ub = py::float_(d["bnb_global_ub"]);
    if (d.contains("bnb_cut_iter_limit"))      p.bnb_cut_iter_limit = py::int_(d["bnb_cut_iter_limit"]);
    if (d.contains("bnb_output_file"))         p.bnb_output_file = py::str(d["bnb_output_file"]);
    if (d.contains("bnb_verbose"))             p.bnb_verbose = py::int_(d["bnb_verbose"]);
    if (d.contains("bnb_output_level"))        p.bnb_output_level = py::int_(d["bnb_output_level"]);

    // --- Fair clustering ---
    if (d.contains("fairness_type"))           p.fair_clustering_fairness_type = py::str(d["fairness_type"]);
    if (d.contains("fairness_param"))          p.fair_clustering_fairness_param = py::float_(d["fairness_param"]);
    if (d.contains("fair_assignment_solver"))  p.fair_assignment_solver = py::str(d["fair_assignment_solver"]);

    // --- Spectral clustering ---
    if (d.contains("is_spectral"))             p.is_spectral_clustering = py::bool_(d["is_spectral"]);

    return p;
}

// ---------------------------------------------------------------------------
// Helper: build a Python dict from solve results
// ---------------------------------------------------------------------------
static py::dict build_result_dict(
    double lloyd_obj,
    const cutLPKSolveInfo& info,
    ICPStatus status,
    bool bnb_executed,
    BnBStatus bnb_status,
    const std::vector<int>& assignment
) {
    py::dict r;
    r["cost"] = info.upper_bound;
    r["relative_gap"] = info.optimality_gap;
    r["lower_bound"] = info.lower_bound;
    r["upper_bound"] = info.upper_bound;
    r["warm_start_cost"] = lloyd_obj;
    r["status"] = static_cast<int>(status);
    r["retcode"] = info.retcode;
    r["bnb_executed"] = bnb_executed;
    r["bnb_status"] = static_cast<int>(bnb_status);
    r["solver_time"] = info.total_solver_time;
    r["separation_time"] = info.total_separation_time;
    r["heuristic_time"] = info.total_post_heuristic_time;

    // Labels
    if (!assignment.empty()) {
        r["labels"] = py::array_t<int>(assignment.size(), assignment.data());
    } else {
        r["labels"] = py::none();
    }

    // Solution matrix (optional: can be large)
    // r["X_solution"] = info.best_upper_bound_solution;

    return r;
}

// ---------------------------------------------------------------------------
// Python-callable wrappers
// ---------------------------------------------------------------------------

py::dict run_ordinary_kmeans(
    py::array_t<double, py::array::c_style | py::array::forcecast> data,
    int K,
    py::dict params_dict
) {
    // Convert numpy array → VectorXdList
    auto buf = data.request();
    if (buf.ndim != 2)
        throw std::runtime_error("Input data must be a 2D array");
    int n = static_cast<int>(buf.shape[0]);
    int d = static_cast<int>(buf.shape[1]);
    double* ptr = static_cast<double*>(buf.ptr);

    VectorXdList dataPoints;
    dataPoints.reserve(n);
    for (int i = 0; i < n; ++i) {
        Eigen::VectorXd point(d);
        for (int j = 0; j < d; ++j)
            point(j) = ptr[i * d + j];
        dataPoints.push_back(point);
    }

    parameters p = dict_to_params(params_dict, K);
    OrdinaryKMeansResult result = solveOrdinaryKMeans(dataPoints, K, p);

    return build_result_dict(
        result.lloyd_objective,
        result.cut_info,
        result.icp_status,
        result.bnb_executed,
        result.bnb_status,
        result.assignment
    );
}

py::dict run_fair_kmeans(
    py::array_t<double, py::array::c_style | py::array::forcecast> data,
    int K,
    py::object groups_arg,
    py::dict params_dict
) {
    // Convert numpy array → VectorXdList
    auto buf = data.request();
    if (buf.ndim != 2)
        throw std::runtime_error("Input data must be a 2D array");
    int n = static_cast<int>(buf.shape[0]);
    int d = static_cast<int>(buf.shape[1]);
    double* ptr = static_cast<double*>(buf.ptr);

    VectorXdList dataPoints;
    dataPoints.reserve(n);
    for (int i = 0; i < n; ++i) {
        Eigen::VectorXd point(d);
        for (int j = 0; j < d; ++j)
            point(j) = ptr[i * d + j];
        dataPoints.push_back(point);
    }

    // Parse groups argument: can be list[int] or 2D bool array
    std::vector<std::vector<bool>> dataGroups;
    std::vector<int> groupRatio;

    if (py::isinstance<py::list>(groups_arg) || py::isinstance<py::array>(groups_arg)) {
        // Try as 1D array of group labels
        py::array labels_arr = py::cast<py::array>(groups_arg);
        auto labels_buf = labels_arr.request();
        if (labels_buf.ndim == 1) {
            // 1D: group label per point
            if (static_cast<int>(labels_buf.shape[0]) != n)
                throw std::runtime_error("groups length must match number of data points");

            int* labels_ptr = static_cast<int*>(labels_buf.ptr);
            // Count unique groups
            std::set<int> unique_groups;
            for (int i = 0; i < n; ++i) unique_groups.insert(labels_ptr[i]);
            int num_groups = static_cast<int>(unique_groups.size());
            groupRatio.resize(num_groups, 0);

            dataGroups.resize(n, std::vector<bool>(num_groups, false));
            for (int i = 0; i < n; ++i) {
                int g = labels_ptr[i];
                // Map original label to 0..num_groups-1
                int idx = static_cast<int>(std::distance(unique_groups.begin(), unique_groups.find(g)));
                dataGroups[i][idx] = true;
                groupRatio[idx]++;
            }
        } else if (labels_buf.ndim == 2) {
            // 2D: boolean matrix n x num_groups
            int ng = static_cast<int>(labels_buf.shape[1]);
            groupRatio.resize(ng, 0);
            dataGroups.resize(n, std::vector<bool>(ng, false));
            // assume int/float entries, cast to bool
            double* mat_ptr = static_cast<double*>(labels_buf.ptr);
            for (int i = 0; i < n; ++i) {
                for (int g = 0; g < ng; ++g) {
                    bool val = (mat_ptr[i * ng + g] != 0.0);
                    dataGroups[i][g] = val;
                    if (val) groupRatio[g]++;
                }
            }
        } else {
            throw std::runtime_error("groups must be 1D (labels) or 2D (boolean matrix)");
        }
    } else {
        throw std::runtime_error("groups must be a list/array of labels or a boolean matrix");
    }

    parameters p = dict_to_params(params_dict, K);
    FairKMeansResult result = solveFairKMeans(dataPoints, K, dataGroups, groupRatio, p);

    return build_result_dict(
        result.lloyd_objective,
        result.cut_info,
        result.icp_status,
        result.bnb_executed,
        result.bnb_status,
        result.assignment
    );
}

py::dict run_spectral_kmeans(
    py::array_t<double, py::array::c_style | py::array::forcecast> laplacian,
    int K,
    py::dict params_dict
) {
    auto buf = laplacian.request();
    if (buf.ndim != 2)
        throw std::runtime_error("Laplacian must be a 2D array");
    int n = static_cast<int>(buf.shape[0]);
    if (buf.shape[1] != n)
        throw std::runtime_error("Laplacian must be square");

    double* ptr = static_cast<double*>(buf.ptr);
    Eigen::MatrixXd L(n, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            L(i, j) = ptr[i * n + j];

    parameters p = dict_to_params(params_dict, K);
    p.is_spectral_clustering = true;

    SpectralKMeansResult result = solveSpectralKMeans(L, K, p);

    // Spectral returns slightly different info
    py::dict r;
    r["cost"] = result.cut_info.upper_bound;
    r["relative_gap"] = result.cut_info.optimality_gap;
    r["lower_bound"] = result.cut_info.lower_bound;
    r["upper_bound"] = result.cut_info.upper_bound;
    r["warm_start_cost"] = result.spectral_objective;
    r["status"] = static_cast<int>(result.icp_status);
    r["retcode"] = result.cut_info.retcode;
    r["bnb_executed"] = result.bnb_executed;
    r["bnb_status"] = static_cast<int>(result.bnb_status);
    r["solver_time"] = result.cut_info.total_solver_time;
    r["separation_time"] = result.cut_info.total_separation_time;
    r["heuristic_time"] = result.cut_info.total_post_heuristic_time;
    r["labels"] = py::none();  // spectral doesn't return labels directly

    return r;
}

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------
PYBIND11_MODULE(_cutlpk, m) {
    m.doc() = "cutLPK: LP-based global solver for K-means, fair K-means, and spectral clustering";

    m.def("run_ordinary_kmeans", &run_ordinary_kmeans,
          py::arg("data"), py::arg("K"), py::arg("params") = py::dict(),
          "Solve ordinary K-means clustering.");

    m.def("run_fair_kmeans", &run_fair_kmeans,
          py::arg("data"), py::arg("K"), py::arg("groups"), py::arg("params") = py::dict(),
          "Solve fair K-means clustering with group fairness constraints.");

    m.def("run_spectral_kmeans", &run_spectral_kmeans,
          py::arg("laplacian"), py::arg("K"), py::arg("params") = py::dict(),
          "Solve spectral clustering (ratio-cut) on a graph Laplacian.");
}
