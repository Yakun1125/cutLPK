#include "../cupdlp/cupdlp.h"
#include "Solver_cupdlp.h"
#include "wrapper_highs.h"
#include "Highs.h"
#include "mps_lp.h"
#include <iostream>
#include <chrono>
#include <stdexcept> // For std::runtime_error
#include <cstring>

int solver_cupdlp(double& dual_obj, double& primal_obj, Eigen::MatrixXd& Xsol, std::vector<validInequality>& cuts, LPK& lp, float tolerance, float time_limit) {
    cupdlp_retcode retcode = RETCODE_OK;
    int numVars = lp.N * (lp.N + 1) / 2;
    int numConstr = lp.consLb.size();


	int numCuts = cuts.size();
    int cuts_idx_start = numConstr - numCuts;

    // print all size info of lp for debugging:
    // lp.printDebugInfo();


    Eigen::VectorXd row_sum; 

    // cuts start when the first time conslb is -infinity and consub is zero
    for (int i = 0; i < numConstr; ++i) {
        if (lp.consLb[i] == -kInfinity && lp.consUb[i] == 0.0) {
            cuts_idx_start = i;
            break;
        }
    }

    HighsModel highs;
    std::vector<int> start, _index;
	std::vector<double> value;

	// Reserve space for the total number of non-zero elements, extract the constraint matrix

    // timing this part
    auto start_time = std::chrono::high_resolution_clock::now();
	_index.reserve(lp.ConsMatrix.nonZeros());
	value.reserve(lp.ConsMatrix.nonZeros());
	start.push_back(0);
	for (int k = 0; k < lp.ConsMatrix.outerSize(); ++k) {
		int colStart = lp.ConsMatrix.outerIndexPtr()[k];
		int colEnd = lp.ConsMatrix.outerIndexPtr()[k + 1];

		for (int idx = colStart; idx < colEnd; ++idx) {
			_index.push_back(lp.ConsMatrix.innerIndexPtr()[idx]);
			value.push_back(lp.ConsMatrix.valuePtr()[idx]);
		}

		start.push_back(colEnd);
	}
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end_time - start_time;
    // std::cout << "Time taken to extract constraint matrix: " << elapsed_time.count() << " seconds" << std::endl;

    highs.lp_.num_col_ = numVars;
	highs.lp_.col_cost_ = lp.objCoef;
	highs.lp_.sense_ = ObjSense::kMinimize;
	highs.lp_.col_lower_ = lp.varLb;
	highs.lp_.col_upper_ = lp.varUb;
	highs.lp_.num_row_ = numConstr;
	highs.lp_.row_lower_ = lp.consLb;
	highs.lp_.row_upper_ = lp.consUb;
	highs.lp_.a_matrix_.format_ = MatrixFormat::kColwise;
	highs.lp_.a_matrix_.start_ = start;
	highs.lp_.a_matrix_.index_ = _index;
	highs.lp_.a_matrix_.value_ = value;
	Highs* model = new Highs();
	model->setOptionValue("log_to_console", false);
	model->passModel(highs);

    // print out number of rows in model
    // std::cout << "Number of rows in model: " << model->getNumRows() << std::endl;

	// some utility variables for post processing
    int col_idx = 0;
	double dual_value_sum = 0.0;

    // std::cout<<"numCuts: "<<numCuts<<std::endl;
	Eigen::VectorXd r;
	Eigen::VectorXd epsilon;
	Eigen::VectorXd delta;
	Eigen::VectorXd row_dual_org_vec;
	int return_code = 0;

	/*********************************
	 Start PDLP part
	**********************************/
    int nCols_pdlp = numVars;
    int nRows_pdlp = numConstr;
    int nnz_pdlp = lp.ConsMatrix.nonZeros();
    int nEqs_pdlp = 0;
    int status_pdlp = -1;

    // Declare all variables at the top to avoid jumping over initializations
    cupdlp_float* rhs = NULL;
    cupdlp_float* cost = NULL;
    cupdlp_float* lower = NULL;
    cupdlp_float* upper = NULL;
    int* csc_beg = NULL;
    int* csc_idx = NULL;
    double* csc_val = NULL;
    double offset = 0.0;
    double sense = 1;
    int* constraint_new_idx = NULL;
    int* constraint_type = NULL;

    int nCols_org = 0;
	int nRows_org = 0;
    cupdlp_float* col_value_org = NULL;
    cupdlp_float* col_dual_org = NULL;
    cupdlp_float* row_value_org = NULL;
    cupdlp_float* row_dual_org = NULL;
    cupdlp_int value_valid = 0;
	cupdlp_int dual_valid = 0;

    char* fout = NULL;
    char* fout_sol = NULL;

    CUPDLPscaling* scaling = (CUPDLPscaling*)cupdlp_malloc(sizeof(CUPDLPscaling));
    CUPDLP_MATRIX_FORMAT src_matrix_format = CSC;
    CUPDLP_MATRIX_FORMAT dst_matrix_format = CSR_CSC;
    CUPDLPcsc* csc_cpu = cupdlp_NULL;
    CUPDLPproblem* prob = cupdlp_NULL;
    CUPDLPwork* w = cupdlp_NULL;
    cupdlp_float alloc_matrix_time = 0.0;
    cupdlp_float cuda_prepare_time = 0.0;
    cupdlp_float copy_vec_time = 0.0;

    cupdlp_bool ifChangeIntParam[N_INT_USER_PARAM] = { false };
    cupdlp_int intParam[N_INT_USER_PARAM] = { 0 };
    cupdlp_bool ifChangeFloatParam[N_FLOAT_USER_PARAM] = { false };
    cupdlp_float floatParam[N_FLOAT_USER_PARAM] = { 0.0 };

    // ifChangeIntParam[IF_SCALING] = false;
    // intParam[IF_SCALING] = 0;

    ifChangeFloatParam[D_TIME_LIM] = true;
    floatParam[D_TIME_LIM] = time_limit;
    ifChangeFloatParam[D_PRIMAL_TOL] = true;
    floatParam[D_PRIMAL_TOL] = tolerance;
    ifChangeFloatParam[D_DUAL_TOL] = true;
    floatParam[D_DUAL_TOL] = tolerance;
    ifChangeFloatParam[D_GAP_TOL] = true;
    floatParam[D_GAP_TOL] = 1e-6;


	getModelSize_highs(model, &nCols_org, &nRows_org, NULL);
	//model2solve = model;

	CUPDLP_CALL(formulateLP_highs(model, &cost, &nCols_pdlp, &nRows_pdlp,
		&nnz_pdlp, &nEqs_pdlp, &csc_beg, &csc_idx,
		&csc_val, &rhs, &lower, &upper, &offset, &sense,
		&nCols_org, &constraint_new_idx, &constraint_type));
	CUPDLP_CALL(Init_Scaling(scaling, nCols_pdlp, nRows_pdlp, cost, rhs));
	// the work object needs to be established first
	// free inside cuPDLP
	CUPDLP_INIT_ZERO(w, 1);

    cuda_prepare_time = getTimeStamp();
    CHECK_CUSPARSE(cusparseCreate(&w->cusparsehandle));
    CHECK_CUBLAS(cublasCreate(&w->cublashandle));
    cuda_prepare_time = getTimeStamp() - cuda_prepare_time;

    CUPDLP_CALL(problem_create(&prob));
    CUPDLP_CALL(csc_create(&csc_cpu));
    csc_cpu->nRows = nRows_pdlp;
    csc_cpu->nCols = nCols_pdlp;
    csc_cpu->nMatElem = nnz_pdlp;
	csc_cpu->colMatBeg = (int*)malloc((1 + nCols_pdlp) * sizeof(int));
	csc_cpu->colMatIdx = (int*)malloc(nnz_pdlp * sizeof(int));
	csc_cpu->colMatElem = (double*)malloc(nnz_pdlp * sizeof(double));
	memcpy(csc_cpu->colMatBeg, csc_beg, (nCols_pdlp + 1) * sizeof(int));
	memcpy(csc_cpu->colMatIdx, csc_idx, nnz_pdlp * sizeof(int));
	memcpy(csc_cpu->colMatElem, csc_val, nnz_pdlp * sizeof(double));

    csc_cpu->cuda_csc = NULL;

    CUPDLP_CALL(PDHG_Scale_Data_cuda(csc_cpu, 1, scaling, cost, lower, upper, rhs));

    CUPDLP_CALL(problem_alloc(prob, nRows_pdlp, nCols_pdlp, nEqs_pdlp, cost,
        offset, sense, csc_cpu, src_matrix_format,
        dst_matrix_format, rhs, lower, upper,
        &alloc_matrix_time, &copy_vec_time));

    w->problem = prob;
    w->scaling = scaling;
    PDHG_Alloc(w);
    w->timers->dScalingTime = 0.0;
    w->timers->dPresolveTime = 0.0;

    CUPDLP_INIT_ZERO(col_value_org, nCols_pdlp);
    CUPDLP_INIT_ZERO(col_dual_org, nCols_pdlp);
    CUPDLP_INIT_ZERO(row_value_org, nRows_pdlp);
    CUPDLP_INIT_ZERO(row_dual_org, nRows_pdlp);
    CUPDLP_COPY_VEC(w->rowScale, scaling->rowScale, cupdlp_float, nRows_pdlp);
	CUPDLP_COPY_VEC(w->colScale, scaling->colScale, cupdlp_float, nCols_pdlp);

    w->timers->AllocMem_CopyMatToDeviceTime += alloc_matrix_time;
	w->timers->CopyVecToDeviceTime += copy_vec_time;
	w->timers->CudaPrepareTime = cuda_prepare_time;


	CUPDLP_CALL(LP_SolvePDHG(w, ifChangeIntParam, intParam, ifChangeFloatParam,
		floatParam, fout, nCols_org, col_value_org, col_dual_org,
		row_value_org, row_dual_org, &value_valid, &dual_valid, 0,
		fout_sol, constraint_new_idx, constraint_type,
		&status_pdlp));

    /*
    Post-processing of the results
    */

    // timing post processing part
    start_time = std::chrono::high_resolution_clock::now();

    primal_obj = Eigen::VectorXd::Map(lp.objCoef.data(), lp.objCoef.size()).dot(Eigen::VectorXd::Map(col_value_org, numVars));
    dual_obj = 0.0;

    col_idx = 0;
	for (int i = 0; i < lp.N; ++i) {
		for (int j = i; j < lp.N; ++j) {
			double value = col_value_org[col_idx++];
			Xsol(i, j) = value;
			if (i != j) {
				Xsol(j, i) = value;  // For symmetric matrix
			}
		}
	}

	// Primal constrraints residual
    r = lp.ConsMatrix * Eigen::VectorXd::Map(col_value_org, numVars);

    // std::cout<<"cut start index: "<<cuts_idx_start<<std::endl;  

    // go through last numCuts of r, which is the residual of the cuts
    for (int cut_idx = 0; cut_idx < numCuts; ++cut_idx) {
        cuts[cut_idx].violation = r[cuts_idx_start + cut_idx];
        cuts[cut_idx].dual_value = row_dual_org[cuts_idx_start + cut_idx];
    }

    // check nRows_pdlp
    // std::cout<< "nRows_pdlp: " << nRows_pdlp << ", numConstr: " << numConstr << std::endl;
    // if (nRows_pdlp != numConstr) {
    //     std::cerr << "Error: nRows_pdlp (" << nRows_pdlp << ") does not match numConstr (" << numConstr << ")." << std::endl;
    //     return_code = 3;
    //     goto exit_cleanup;
    // }

    r = lp.ConsMatrix.transpose() * Eigen::VectorXd::Map(row_dual_org, numConstr) - Eigen::VectorXd::Map(lp.objCoef.data(), lp.objCoef.size());

	// Then we compute the dual_obj manually
	dual_value_sum = row_dual_org[0] * lp.consUb[0];
	for (int i = 1; i < lp.N + 1; i++) {
		dual_value_sum += row_dual_org[i];
	}

	for (int i = lp.N + 1; i < cuts_idx_start; i++) {
		if (row_dual_org[i] > 0) {
			dual_value_sum += row_dual_org[i] * lp.consLb[i];
		}
		else {
			dual_value_sum += row_dual_org[i] * lp.consUb[i];
		}
	}


	col_idx = 0;
	for (int i = 0; i < lp.N; ++i) {
		for (int j = i; j < lp.N; ++j) {
			if (r[col_idx] > 0) {
				dual_value_sum -= r[col_idx] * 1;//col_value_org[col_idx]largest_x
			}
			col_idx++;
		}
	}


	// clean all local var: r and row_dual_org_vec
	r.resize(0);
	row_dual_org_vec.resize(0);

	if (status_pdlp == 0) {
		return_code = 0;
	}
	else if (status_pdlp == 4) {// 4 means time limit reached and it's acceptable
		return_code = 1;
	}
	else {
		// print termCode
		return_code = 2;
	}
	dual_obj = dual_value_sum;

    end_time = std::chrono::high_resolution_clock::now();
    elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    // std::cout << "Post-processing time: " << elapsed_time.count() / 1e3 << " seconds" << std::endl;

exit_cleanup:
    deleteModel_highs(model);
    if (col_value_org) cupdlp_free(col_value_org);
    if (col_dual_org) cupdlp_free(col_dual_org);
    if (row_value_org) cupdlp_free(row_value_org);
    if (row_dual_org) cupdlp_free(row_dual_org);
    if (csc_cpu) csc_clear(csc_cpu);
    if (prob) problem_clear(prob);
    if (cost) free(cost);
    if (lower) free(lower);
    if (upper) free(upper);
    if (rhs) free(rhs);
    if (csc_beg) free(csc_beg);
    if (csc_idx) free(csc_idx);
    if (csc_val) free(csc_val);
    if (constraint_new_idx != NULL) cupdlp_free(constraint_new_idx);
	if (constraint_type != NULL) cupdlp_free(constraint_type);

    if (scaling) {
		scaling_clear(scaling);
	}
    return return_code;
}