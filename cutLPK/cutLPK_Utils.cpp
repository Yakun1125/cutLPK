#include "cutLPK_Utils.h"

//namespace pdlp = ::operations_research::pdlp;

void constructLPK(LPK& lp, Eigen::MatrixXd& dis_matrix, int N, int K, std::vector<Eigen::Triplet<int>>& basic_triplets) {
	lp.N = N;
	int numVars = N * (N + 1) / 2;
	lp.varLb = std::vector<double>(numVars, 0.0);
	lp.varUb = std::vector<double>(numVars, kInfinity);
	lp.objCoef = std::vector<double>(numVars, 0.0);
	Eigen::SparseMatrix<double, Eigen::ColMajor> ConsMatrix;// constraint matrix
	int index = 0; // index mapping logic is (i,j) j>i index = i*(2*N-i+1)/2+j-i
	for (int i = 0; i < N; ++i) {
		for (int j = i; j < N; ++j) {
			lp.objCoef[index++] = dis_matrix(i, j);
		}
	}
	// basic part constraint bounds
	lp.consLb = std::vector<double> (1 + N, 1);
	lp.consUb = std::vector<double> (1 + N, 1);
	lp.consLb[0] = K;
	lp.consUb[0] = K;
	// Reserve space for triplets based on an estimate of non-zero elements
	basic_triplets.reserve((N + 1) * N);
	// Add non-zeros for the original matrix part
	for (int i = 0; i < N; ++i) {
		int col = i * (2 * N - i + 1) / 2;
		basic_triplets.emplace_back(0, col, 1); // First row, diagonal elements set to 1
	}

	for (int i = 1; i <= N; ++i) {
		for (int j = 0; j < N; ++j) {
			int col = std::min(i - 1, j) * (2 * N - std::min(i - 1, j) + 1) / 2 + std::max(i - 1, j) - std::min(i - 1, j);
			basic_triplets.emplace_back(i, col, 1); // Subsequent rows
		}
	}
}

initializationInfo addInitialCuts(const parameters& params, int N, int K, int cuts_idx_start,
	const std::vector<Eigen::VectorXd>& dataPoints, Eigen::MatrixXd& Lloyd_Xsol, std::vector<Eigen::Triplet<int>>& cuts_triplets, std::vector<validInequality>& cuts) {
	initializationInfo initInfo;
	
	unsigned long long totalCombinations = static_cast<unsigned long long>(N) * (N - 1) * (N - 2) / 2;
	int initial_size = static_cast<int>(std::min(totalCombinations, static_cast<unsigned long long>(params.max_init)));
	cuts.reserve(initial_size);
	cuts_triplets.reserve(4 * initial_size);
	double KmeansPlusPlus_Cost = kInfinity;

	// timing this part
	auto start = std::chrono::high_resolution_clock::now();
	std::tie(KmeansPlusPlus_Cost, Lloyd_Xsol) = runKMeans(dataPoints, K, 100000, 50, params.random_seed);
	auto end = std::chrono::high_resolution_clock::now();
	initInfo.Lloyd_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	initInfo.Lloyd_Obj = KmeansPlusPlus_Cost;

	int added_count = 0;
	unsigned long long scaned_count = 0;
	unsigned long long act_size = 0;
	unsigned long long total_size = (N) * (N - 1) * (N - 2) / 2;

	// timeing Identify cuts
	start = std::chrono::high_resolution_clock::now();
	if (params.warm_start == 1) {
		int size_each_i = params.max_init / N;// when problem size is large, we added first size_each_i triangle inequalities for each i,j 
		for (int i = 0; i < N; i++) {
			int added_count_i = 0;
			for (int j = 0; j < N; j++) {
				if (i != j) {
					for (int k = j + 1; k < N; ++k) {
						if (k != i) {
							double violation = Lloyd_Xsol(i, j) + Lloyd_Xsol(i, k) - Lloyd_Xsol(i, i) - Lloyd_Xsol(j, k);

							if (violation > -params.cuts_act_tol && violation < params.cuts_act_tol) {
								int newRow = cuts_idx_start + added_count;
								cuts.emplace_back(validInequality(std::list<int>{i, j, k}, violation));
								cuts_triplets.emplace_back(newRow, std::min(i, j) * (2 * N - std::min(i, j) + 1) / 2 + std::max(i, j) - std::min(i, j), 1);
								cuts_triplets.emplace_back(newRow, std::min(i, k) * (2 * N - std::min(i, k) + 1) / 2 + std::max(i, k) - std::min(i, k), 1);
								cuts_triplets.emplace_back(newRow, i * (2 * N - i + 1) / 2, -1);
								cuts_triplets.emplace_back(newRow, j * (2 * N - j + 1) / 2 + k - j, -1);
								added_count_i++;
								added_count++;
							}
						}
						if (added_count_i >= size_each_i) {
							break;
						}
					}
				}
				if (added_count_i >= size_each_i) {
					break;
				}
			}
		}
	}
	else {
		std::mt19937 gen(params.random_seed);
		std::uniform_real_distribution<double> dis(0.0, 1.0);

		for (int i = 0; i < N; ++i) {
			for (int j = 0; j < N; ++j) {
				if (i != j) {
					for (int k = j + 1; k < N; ++k) {
						if (k != i) {
							double violation = Lloyd_Xsol(i, j) + Lloyd_Xsol(i, k) - Lloyd_Xsol(i, i) - Lloyd_Xsol(j, k);

							if (violation > -params.cuts_act_tol && violation < params.cuts_act_tol) {
								unsigned long long remaining_combinations = static_cast<unsigned long long>(N) * (N - 1) * (N - 2) / 2 - scaned_count;
								double p = double(initial_size - added_count) / double(remaining_combinations);
								act_size++;
								if (dis(gen) < p) {
									int newRow = cuts_idx_start + added_count;
									cuts.emplace_back(validInequality(std::list<int>{i, j, k}, violation));
									cuts_triplets.emplace_back(newRow, std::min(i, j) * (2 * N - std::min(i, j) + 1) / 2 + std::max(i, j) - std::min(i, j), 1);
									cuts_triplets.emplace_back(newRow, std::min(i, k) * (2 * N - std::min(i, k) + 1) / 2 + std::max(i, k) - std::min(i, k), 1);
									cuts_triplets.emplace_back(newRow, i * (2 * N - i + 1) / 2, -1);
									cuts_triplets.emplace_back(newRow, j * (2 * N - j + 1) / 2 + k - j, -1);
									added_count++;
								}
							}
							scaned_count++;
						}
					}
				}
			}
		}
	}
	end = std::chrono::high_resolution_clock::now();
	initInfo.cuts_Identified_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	initInfo.act_cuts_size = act_size;
	initInfo.added_cuts_size = added_count;

	return initInfo;
}

int solver_cupdlp(double& dual_obj, double& primal_obj, Eigen::MatrixXd& Xsol, std::vector<validInequality>& cuts, const LPK& lp, float tolerance, float time_limit) {
	cupdlp_retcode retcode = RETCODE_OK;

	HighsModel highs;
	int numVars = lp.N * (lp.N + 1) / 2;
	int numConstr = lp.consLb.size();

	std::vector<int> start, _index;
	std::vector<double> value;

	// Reserve space for the total number of non-zero elements, extract the constraint matrix
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

	// Add variables to the model
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

	/*********************************
     Start PDLP part
    **********************************/
	int nCols_pdlp = 0;
	int nRows_pdlp = 0;
	int nEqs_pdlp = 0;
	int nnz_pdlp = 0;
	int status_pdlp = -1;

	cupdlp_float* rhs = NULL;
	cupdlp_float* cost = NULL;
	cupdlp_float* lower = NULL;
	cupdlp_float* upper = NULL;

	// -------------------------
	int* csc_beg = NULL, * csc_idx = NULL;
	double* csc_val = NULL;

	// for model to solve, need to free
	double offset = 0.0;  // true objVal = sig * c'x + offset, sig = 1 (min) or -1 (max)
	double sense = 1;  // 1 (min) or -1 (max)
	int* constraint_new_idx = NULL;
	int* constraint_type = NULL;

	// for model to solve, need not to free
	int nCols = 0;
	//cupdlp_float* col_value = cupdlp_NULL;
	//cupdlp_float* col_dual = cupdlp_NULL;
	//cupdlp_float* row_value = cupdlp_NULL;
	//cupdlp_float* row_dual = cupdlp_NULL;


	// for original model, need to free
	int nCols_org = 0;
	int nRows_org = 0;
	cupdlp_float* col_value_org = cupdlp_NULL;
	cupdlp_float* col_dual_org = cupdlp_NULL;
	cupdlp_float* row_value_org = cupdlp_NULL;
	cupdlp_float* row_dual_org = cupdlp_NULL;

	cupdlp_int value_valid = 0;
	cupdlp_int dual_valid = 0;

	//void* model2solve = NULL;

	CUPDLPscaling* scaling =
		(CUPDLPscaling*)cupdlp_malloc(sizeof(CUPDLPscaling));

	// claim solvers variables
	// prepare pointers
	CUPDLP_MATRIX_FORMAT src_matrix_format = CSC;
	CUPDLP_MATRIX_FORMAT dst_matrix_format = CSR_CSC;
	CUPDLPcsc* csc_cpu = cupdlp_NULL;
	CUPDLPproblem* prob = cupdlp_NULL;

	//
	CUPDLPwork* w = cupdlp_NULL;
	cupdlp_float alloc_matrix_time = 0.0;
	cupdlp_float cuda_prepare_time = 0.0;
	cupdlp_float copy_vec_time = 0.0;
	cupdlp_bool ifChangeIntParam[N_INT_USER_PARAM] = { false };
	cupdlp_int intParam[N_INT_USER_PARAM] = { 0 };
	cupdlp_bool ifChangeFloatParam[N_FLOAT_USER_PARAM] = { false };
	cupdlp_float floatParam[N_FLOAT_USER_PARAM] = { 0.0 };
	ifChangeFloatParam[D_TIME_LIM] = true;
	floatParam[D_TIME_LIM] = time_limit;
	ifChangeFloatParam[D_PRIMAL_TOL] = true;
	floatParam[D_PRIMAL_TOL] = tolerance;
	ifChangeFloatParam[D_DUAL_TOL] = true;
	floatParam[D_DUAL_TOL] = tolerance;
	ifChangeFloatParam[D_GAP_TOL] = true;
	floatParam[D_GAP_TOL] = 1e-4;
	char* fout = NULL;
	char* fout_sol = NULL;

	int col_idx = 0;
	int numCuts = cuts.size();
	int cuts_idx_start = numConstr - numCuts;
	double dual_value_sum = 0.0;
	Eigen::VectorXd r;
	Eigen::VectorXd epsilon;
	Eigen::VectorXd delta;
	Eigen::VectorXd row_dual_org_vec;
	int return_code = 0;

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

#if !(CUPDLP_CPU)
	cuda_prepare_time = getTimeStamp();
	CHECK_CUSPARSE(cusparseCreate(&w->cusparsehandle));
	CHECK_CUBLAS(cublasCreate(&w->cublashandle));
	cuda_prepare_time = getTimeStamp() - cuda_prepare_time;
#endif
	CUPDLP_CALL(problem_create(&prob));

	// currently, only supprot that input matrix is CSC, and store both CSC and
	// CSR
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

#if !(CUPDLP_CPU)
	csc_cpu->cuda_csc = NULL;
#endif

	CUPDLP_CALL(PDHG_Scale_Data_cuda(csc_cpu, 1, scaling, cost, lower,
		upper, rhs));



	CUPDLP_CALL(problem_alloc(prob, nRows_pdlp, nCols_pdlp, nEqs_pdlp, cost,
		offset, sense, csc_cpu, src_matrix_format,
		dst_matrix_format, rhs, lower, upper,
		&alloc_matrix_time, &copy_vec_time));

	// solve
	w->problem = prob;
	w->scaling = scaling;
	PDHG_Alloc(w);
	w->timers->dScalingTime = 0.0;
	w->timers->dPresolveTime = 0.0;
	CUPDLP_COPY_VEC(w->rowScale, scaling->rowScale, cupdlp_float, nRows_pdlp);
	CUPDLP_COPY_VEC(w->colScale, scaling->colScale, cupdlp_float, nCols_pdlp);

#if !(CUPDLP_CPU)
	w->timers->AllocMem_CopyMatToDeviceTime += alloc_matrix_time;
	w->timers->CopyVecToDeviceTime += copy_vec_time;
	w->timers->CudaPrepareTime = cuda_prepare_time;
#endif

	CUPDLP_INIT_ZERO(col_value_org, nCols_org);
	CUPDLP_INIT_ZERO(col_dual_org, nCols_org);
	CUPDLP_INIT_ZERO(row_value_org, nRows_org);
	CUPDLP_INIT_ZERO(row_dual_org, nRows_org);

	CUPDLP_CALL(LP_SolvePDHG(w, ifChangeIntParam, intParam, ifChangeFloatParam,
		floatParam, fout, nCols_org, col_value_org, col_dual_org,
		row_value_org, row_dual_org, &value_valid, &dual_valid, 0,
		fout_sol, constraint_new_idx, constraint_type,
		&status_pdlp));

	/*****************************************************
     Update Xsol and dual value for next lpk construction
    ******************************************************/
    // Report the Xsol
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

	// go through last numCuts of r, which is the residual of the cuts
    for (int i = 0; i < numCuts; ++i) {
		cuts[i].violation = r[cuts_idx_start + i];
	}


	// dual constraints residual
    // compute vector r by r = ConsMatrix^T * row_dual_org(every row) - obj_coef
    //r = Eigen::VectorXd::Map(obj_coef.data(), obj_coef.size());
    //row_dual_org_vec = Eigen::VectorXd::Map(row_dual_org, numConstr);
    //r -= ConsMatrix.transpose() * row_dual_org_vec;

	//debug  this part, print size info


	r = lp.ConsMatrix.transpose() * Eigen::VectorXd::Map(row_dual_org, numConstr) - Eigen::VectorXd::Map(lp.objCoef.data(), lp.objCoef.size());

	// Then we compute the dual_obj manually
	dual_value_sum = row_dual_org[0] * lp.consUb[0];
	for (int i = 1; i < lp.N + 1; i++) {
		dual_value_sum += row_dual_org[i];
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
	if (w->resobj->termIterate == LAST_ITERATE) {
		//*dual_obj = w->resobj->dDualObj;
		primal_obj = w->resobj->dPrimalObj;
	}
	else {
		//*dual_obj = w->resobj->dDualObjAverage;
		primal_obj = w->resobj->dPrimalObjAverage;
	}
	if (w->resobj->termCode == OPTIMAL) {
		return_code = 0;
	}
	else if (w->resobj->termCode == TIMELIMIT_OR_ITERLIMIT) {
		return_code = 1;
	}
	else {
		return_code = 2;
	
	}
	dual_obj = dual_value_sum;

exit_cleanup:
	// free model and solution
	deleteModel_highs(model);
	if (col_value_org != NULL) cupdlp_free(col_value_org);
	if (col_dual_org != NULL) cupdlp_free(col_dual_org);
	if (row_value_org != NULL) cupdlp_free(row_value_org);
	if (row_dual_org != NULL) cupdlp_free(row_dual_org);

	// free problem
	if (scaling) {
		scaling_clear(scaling);
	}

	if (cost != NULL) cupdlp_free(cost);
	if (csc_beg != NULL) cupdlp_free(csc_beg);
	if (csc_idx != NULL) cupdlp_free(csc_idx);
	if (csc_val != NULL) cupdlp_free(csc_val);
	if (rhs != NULL) cupdlp_free(rhs);
	if (lower != NULL) cupdlp_free(lower);
	if (upper != NULL) cupdlp_free(upper);
	if (constraint_new_idx != NULL) cupdlp_free(constraint_new_idx);
	if (constraint_type != NULL) cupdlp_free(constraint_type);

	// free memory
	csc_clear(csc_cpu);
	problem_clear(prob);

	return return_code;
}

Eigen::MatrixXd postHeuristic(const Eigen::MatrixXd& Xsol, const std::vector<Eigen::VectorXd>& dataPoints, int k, int maxIterations) {
	int N = dataPoints.size();
	// compute best rank-k approximation X_k for Xsol
	Eigen::SelfAdjointEigenSolver<Eigen::SparseMatrix<double>> eigensolver(Eigen::SparseMatrix<double>(Xsol.sparseView()));
	if (eigensolver.info() != Eigen::Success) {
		throw std::runtime_error("Eigenvalue decomposition failed!");
	}

	Eigen::VectorXd eigenvalues = eigensolver.eigenvalues();
	Eigen::MatrixXd eigenvectors = eigensolver.eigenvectors();
	std::vector<int> idx(eigenvalues.size());
	std::iota(idx.begin(), idx.end(), 0);

	// Sort indices based on comparing absolute values of corresponding eigenvalues
	std::sort(idx.begin(), idx.end(),
		[&eigenvalues](int i1, int i2) { return fabs(eigenvalues(i1)) > fabs(eigenvalues(i2)); });

	// Arrange the eigenvalues and eigenvectors in descending order of their absolute values
	Eigen::VectorXd sortedEigenvalues(k);
	Eigen::MatrixXd sortedEigenvectors(eigenvectors.rows(), k);

	for (int i = 0; i < k; ++i) {
		sortedEigenvalues(i) = eigenvalues(idx[i]);
		sortedEigenvectors.col(i) = eigenvectors.col(idx[i]);
	}

	// Construct the rank-k approximation matrix
	Eigen::MatrixXd X_k = sortedEigenvectors * sortedEigenvalues.asDiagonal() * sortedEigenvectors.transpose();

	int d = dataPoints[0].size();

	// Use QR decomposition with column pivoting to find k independent rows
	Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(X_k.transpose());
	Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> P = qr.colsPermutation();

	std::unordered_set<int> selectedRows;
	for (int i = 0; i < k; ++i) {
		int idx = P.indices()(i);
		selectedRows.insert(idx);
	}

	std::vector<Eigen::VectorXd> centroids;
	for (const auto& rowIndex : selectedRows) {
		Eigen::VectorXd centroid = Eigen::VectorXd::Zero(d);
		for (int i = 0; i < d; ++i) {
			for (int j = 0; j < N; ++j) {
				centroid(i) += X_k(rowIndex, j) * dataPoints[j](i);
			}
		}
		centroids.push_back(centroid);
	}


	Eigen::MatrixXd bestPartitionMatrix = Eigen::MatrixXd::Zero(N, N);
	double bestWCSS = std::numeric_limits<double>::max();

	// run Llyod's method until centroids are fixed
	std::vector<int> assignment(N, 0);
	double currentWCSS = std::numeric_limits<double>::max();
	for (int iter = 0; iter < maxIterations; ++iter) {
		bool changed = assignClusters(dataPoints, centroids, assignment);
		updateCentroids(dataPoints, centroids, assignment, k);
		double newWCSS = computeWCSS(dataPoints, centroids, assignment);

		if (!changed || std::abs(newWCSS - currentWCSS) < 1e-6) {
			break;
		}
		currentWCSS = newWCSS;
	}
	bestPartitionMatrix = createPartitionMatrix(assignment, N, k);

	//std::cout << std::fixed << std::setprecision(6) <<" ,Llyod's method obj = " << currentWCSS << std::endl; std::cout.unsetf(std::ios_base::fixed);

	return bestPartitionMatrix;
}

void separation_scheme(const Eigen::MatrixXd& Xsol, std::vector<std::list<validInequality>>& violated_cuts, int max_T, int N, int maxSize, double cuts_vio_tol) {
	int max_list_size = 0;
#pragma omp parallel for shared(violated_cuts, max_list_size)
	for (int source = 0; source < N; ++source) {
		for (int j = 0; j < N; ++j) {
			if (j != source) {
				std::list<int> chain = { j };
				int current_node = j;
				double current_cost = -Xsol(source, source) + Xsol(source, j);

				for (int size = 2; size <= max_T; ++size) {
					if (max_list_size >= maxSize) break; // Early exit check

					int best_next_node = -1;
					double max_next_cost = -std::numeric_limits<double>::infinity();
					std::list<int> potential_chain;

					for (int next = current_node + 1; next < N; ++next) {
						if (next != source) {
							double additional_cost = 0.0;
							for (int k : chain) {
								if (k < next) {
									additional_cost += Xsol(k, next);
								}
							}
							double next_cost = current_cost + Xsol(source, next) - additional_cost;
							if (next_cost > max_next_cost) {
								max_next_cost = next_cost;
								best_next_node = next;
								potential_chain = chain;
								potential_chain.push_back(next);
							}
						}
					}

					if (best_next_node != -1 && max_list_size < maxSize) {
						if (max_next_cost > cuts_vio_tol && potential_chain.size() > 1) {
#pragma omp critical
							{
								// create a list of int start with source and then potential_chain
                                std::list<int> cut = potential_chain;
                                cut.push_front(source);
								violated_cuts[size - 2].push_back(validInequality(cut, max_next_cost));
#pragma omp atomic
								max_list_size++;
							}
#pragma omp flush(max_list_size)
						}
						chain = potential_chain;
						current_cost = max_next_cost;
						current_node = best_next_node;
					}
					else {
						break;
					}
				}
			}
		}
	}
}



