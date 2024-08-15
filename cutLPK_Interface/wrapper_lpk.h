#ifndef CUPDLP_WRAPPER_LPK_H
#define CUPDLP_WRAPPER_LPK_H

#ifdef __cplusplus
extern "C" {
#endif

	int solve_lpk(const char* filename, int num_cluster, int max_init, int max_per_iter, int warm_start, int t_upper_bound, double initial_time_limit, double time_limit_iter, double time_limit_all, const char* solver
		, double initial_pdlp_tol, double tolerance_per_iter, double ub_pdlp_tol, double cuts_vio_tol, double cuts_act_tol, double opt_gap, int random_seed, bool lb_ub_scheme);


	int solve_partial_lpk(float tolerance, float time_limit, void* Xsol, int N, double* dual_obj, double* primal_obj, void* dual_value,
		void* lb, void* ub, void* obj_coef,
		void* cons_lb, void* cons_ub, void* combinedMatrix);

	int cpu_partial_lpk(float tolerance, float time_limit, void* XsolPtr, int N, double* dual_obj, double* primal_obj, void* dualValuePtr,
		void* lbPtr, void* ubPtr, void* objCoefPtr,
		void* consLbPtr, void* consUbPtr, void* combinedMatrixPtr);

	int gurobi_partial_lpk(float tolerance, float time_limit, void* XsolPtr, int N, double* dual_obj, double* primal_obj, void* dualValuePtr,
				void* lbPtr, void* ubPtr, void* objCoefPtr,
				void* consLbPtr, void* consUbPtr, void* combinedMatrixPtr);


#ifdef __cplusplus
}
#endif
// input necessary info to build partial LPK,
// Eigen::SparseMatrix<double, Eigen::ColMajor> Xsol
// int number of points; unordered_map dual_value;
// vec lb; vec ub; vec obj_coef; vec cons_lb; vec cons_ub;
// Eigen::SparseMatrix<double, Eigen::ColMajor> combinedMatrix

#endif 