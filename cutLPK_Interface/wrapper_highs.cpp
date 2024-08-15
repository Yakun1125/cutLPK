#include "wrapper_highs.h"

#include <cassert>
#include <string>
#include <Eigen/Sparse>
#include <vector>
#include <tuple>
#include <fstream>
#include <sstream>

#include "Highs.h"
// using namespace std;
using std::cout;
using std::endl;

extern "C" void *createModel_highs() { return new Highs(); }

extern "C" void deleteModel_highs(void *model) {
  if (model != NULL) delete (Highs *)model;
  // free(model);
}

int solve_all_lpk(void* model, const char* filename, int num_cluster) {
    std::ifstream file(filename);
    std::string line;
    std::vector<Eigen::VectorXd> dataPoints;

    // Read data points from file
    while (std::getline(file, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        std::vector<double> dataPoint;

        while (std::getline(lineStream, cell, ',')) {
            dataPoint.push_back(std::stod(cell)); // Convert string to double
        }

        Eigen::VectorXd point = Eigen::Map<Eigen::VectorXd>(dataPoint.data(), dataPoint.size());
        dataPoints.push_back(point);
    }

    Eigen::MatrixXd dis_matrix;
    int N;
    N = dataPoints.size();
    dis_matrix.resize(N, N);

    if (N < 1) {
        return 1;
    }

    // Compute squared Euclidean distances
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            dis_matrix(i, j) = (dataPoints[i] - dataPoints[j]).squaredNorm();
        }
    }

    std::vector<std::tuple<int, int, int>> TriangleIneq;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (i != j) {
                for (int k = j + 1; k < N; ++k) {
                    if (k != i) {
                        TriangleIneq.push_back(std::make_tuple(i, j, k));
                    }
                }
            }
        }
    }

    //int num_cluster = 3;

    HighsModel highs;

    // Prepare vectors for defining model
    // Number of variables for the upper triangular part, including the diagonal
    int numVars = N * (N + 1) / 2;
    std::vector<double> lb(numVars, 0); // Lower bounds
    std::vector<double> ub(numVars, 1); // Upper bounds
    std::vector<double> obj_coef(numVars, 0); // Objective function coefficients
    // Fill objective coefficients based on the upper triangular part of dis_matrix
    int index = 0;
    for (int i = 0; i < N; ++i) {
        for (int j = i; j < N; ++j) {
            obj_coef[index++] = dis_matrix(i, j);
        }
    }
    // index mapping logic is (i,j) j>i index = i*(2*N-i+1)/2+j-i

    /************************
    Implement the basic part
    *************************/
    std::vector<double> cons_lb(1 + N + TriangleIneq.size(), 1);
    std::vector<double> cons_ub(1 + N + TriangleIneq.size(), 1);
    std::fill(cons_lb.begin() + N + 1, cons_lb.end(), -1e30);
    std::fill(cons_ub.begin() + N + 1, cons_ub.end(), 0);
    cons_lb[0] = num_cluster;
    cons_ub[0] = num_cluster;

    // Reserve space for triplets based on an estimate of non-zero elements
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve((N + 1) * N + 4 * TriangleIneq.size()); // Assuming each constraint adds approximately 4 non-zeros

    // Add non-zeros for the original matrix part
    for (int i = 0; i < N; ++i) {
        int col = i * (2 * N - i + 1) / 2;
        triplets.emplace_back(0, col, 1); // First row, diagonal elements set to 1
    }

    for (int i = 1; i <= N; ++i) {
        for (int j = 0; j < N; ++j) {
            int col = std::min(i - 1, j) * (2 * N - std::min(i - 1, j) + 1) / 2 + std::max(i - 1, j) - std::min(i - 1, j);
            triplets.emplace_back(i, col, 1); // Subsequent rows
        }
    }

    /************************
    Add triangle
    *************************/

    int baseRow = N + 1; // New constraints start after the original matrix rows
    for (int idx = 0; idx < TriangleIneq.size(); ++idx) {
        int i, j, k;
        std::tie(i, j, k) = TriangleIneq[idx];
        int newRow = baseRow + idx; // New row index for each constraint

        // Add triplets for the new constraints
        // Note: Adjust the logic here based on your specific constraint rules
        triplets.emplace_back(newRow, std::min(i, j) * (2 * N - std::min(i, j) + 1) / 2 + std::max(i, j) - std::min(i, j), 1);
        triplets.emplace_back(newRow, std::min(i, k) * (2 * N - std::min(i, k) + 1) / 2 + std::max(i, k) - std::min(i, k), 1);
        triplets.emplace_back(newRow, i * (2 * N - i + 1) / 2, -1);
        triplets.emplace_back(newRow, j * (2 * N - j + 1) / 2 + k - j, -1);
    }

    // Create the combined matrix from triplets
    Eigen::SparseMatrix<double, Eigen::ColMajor> combinedMatrix(baseRow + TriangleIneq.size(), N * (N + 1) / 2);
    combinedMatrix.setFromTriplets(triplets.begin(), triplets.end());
    std::vector<int> start, _index;
    std::vector<double> value;

    // Reserve space if you have an estimate of the total number of non-zero elements
    _index.reserve(combinedMatrix.nonZeros());
    value.reserve(combinedMatrix.nonZeros());

    start.push_back(0); // Start of the first column

    for (int k = 0; k < combinedMatrix.outerSize(); ++k) { // Iterate through each column
        int colStart = combinedMatrix.outerIndexPtr()[k]; // Start index of the current column in the values/indices array
        int colEnd = combinedMatrix.outerIndexPtr()[k + 1]; // End index (one past the last element) of the current column

        for (int idx = colStart; idx < colEnd; ++idx) {
            _index.push_back(combinedMatrix.innerIndexPtr()[idx]); // Row index of the non-zero element
            value.push_back(combinedMatrix.valuePtr()[idx]); // Value of the non-zero element
        }

        start.push_back(colEnd); // Start of the next column is the end of the current column
    }

    //std::cout << "Combined Matrix:\n" << Eigen::MatrixXd(combinedMatrix) << std::endl;

    // Add variables to the model
    highs.lp_.num_col_ = numVars;
    highs.lp_.col_cost_ = obj_coef;
    highs.lp_.sense_ = ObjSense::kMinimize;
    highs.lp_.col_lower_ = lb;
    highs.lp_.col_upper_ = ub;

    //std::cout << cons_lb[0] << " " << cons_ub[0] << std::endl;
    highs.lp_.num_row_ = 1 + N + TriangleIneq.size();
    highs.lp_.row_lower_ = cons_lb;
    highs.lp_.row_upper_ = cons_ub;
    highs.lp_.a_matrix_.format_ = MatrixFormat::kColwise;
    highs.lp_.a_matrix_.start_ = start;
    highs.lp_.a_matrix_.index_ = _index;
    highs.lp_.a_matrix_.value_ = value;


    ((Highs*)model)->passModel(highs);

    return 0;
}

extern "C" int create_highs(void *model) {
    HighsModel highs;
    highs.lp_.num_col_ = 2;
    highs.lp_.num_row_ = 3;
    highs.lp_.sense_ = ObjSense::kMinimize;
    highs.lp_.offset_ = 3;
    highs.lp_.col_cost_ = { 1.0, 1.0 };
    highs.lp_.col_lower_ = { 0.0, 1.0 };
    highs.lp_.col_upper_ = { 4.0, 1.0e30 };
    highs.lp_.row_lower_ = { -1.0e30, 5.0, 6.0 };
    highs.lp_.row_upper_ = { 7.0, 15.0, 1.0e30 };
    //
    // Here the orientation of the matrix is column-wise
    highs.lp_.a_matrix_.format_ = MatrixFormat::kColwise;
    // a_start_ has num_col_1 entries, and the last entry is the number
    // of nonzeros in A, allowing the number of nonzeros in the last
    // column to be defined
    highs.lp_.a_matrix_.start_ = { 0, 2, 5 };
    highs.lp_.a_matrix_.index_ = { 1, 2, 0, 1, 2 };
    highs.lp_.a_matrix_.value_ = { 1.0, 3.0, 1.0, 2.0, 2.0 };
    //
    // Create a Highs instance
    HighsStatus return_status;
    //
    // Pass the model to HiGHS
    return_status = ((Highs*)model)->passModel(highs);
    //assert(return_status == HighsStatus::kOk);

    return 0;
}

extern "C" int loadMps_highs(void *model, const char *filename) {
  string str = string(filename);
  // model is lhs <= Ax <= rhs, l <= x <= u
  cout << "--------------------------------------------------" << endl;
  cout << "reading file..." << endl;
  cout << "\t" << std::string(filename) << endl;
  cout << "--------------------------------------------------" << endl;

  HighsStatus return_status = HighsStatus::kOk;
  return_status = ((Highs *)model)->readModel(str);

  if (return_status != HighsStatus::kOk) {
    printf("Error: readModel return status = %d\n", (int)return_status);
    return 1;
  }

  // relax MIP to LP
  const HighsLp &lp = ((Highs *)model)->getLp();
  if (lp.integrality_.size()) {
    for (int i = 0; i < lp.num_col_; i++) {
      if (lp.integrality_[i] != HighsVarType::kContinuous) {
        ((Highs *)model)->changeColIntegrality(i, HighsVarType::kContinuous);
      }
    }
  }

  return 0;
}

// ok 0, timeout 1, infeasOrUnbounded 2, opt 3
extern "C" int presolvedModel_highs(void *presolve, void *model) {
  cout << "--------------------------------------------------" << endl;
  cout << "running presolve" << endl;
  cout << "--------------------------------------------------" << endl;

  int retcode = 0;

  HighsStatus return_status;
  return_status = ((Highs *)model)->presolve();

  assert(return_status == HighsStatus::kOk);

  HighsPresolveStatus model_presolve_status =
      ((Highs *)model)->getModelPresolveStatus();

  // cout << "presolve status: " << (int)model_presolve_status << endl;
  if (model_presolve_status == HighsPresolveStatus::kTimeout) {
    printf("Presolve timeout: return status = %d\n", (int)return_status);
    retcode = 1;
  } else if (model_presolve_status == HighsPresolveStatus::kInfeasible ||
             model_presolve_status ==
                 HighsPresolveStatus::kUnboundedOrInfeasible) {
    retcode = 2;
  } else if (model_presolve_status == HighsPresolveStatus::kReducedToEmpty) {
    retcode = 3;
  }
  HighsLp lp = ((Highs *)model)->getPresolvedLp();
  ((Highs *)presolve)->passModel(lp);

  return retcode;
}

extern "C" void *postsolvedModel_highs(
    void *model, int nCols_pre, int nRows_pre, double *col_value_pre,
    double *col_dual_pre, double *row_value_pre, double *row_dual_pre,
    int value_valid_pre, int dual_valid_pre, int nCols_org, int nRows_org,
    double *col_value_org, double *col_dual_org, double *row_value_org,
    double *row_dual_org) {
  cout << "--------------------------------------------------" << endl;
  cout << "running postsolve" << endl;
  cout << "--------------------------------------------------" << endl;

  // construct solution
  HighsSolution solution_pre;
  HighsStatus return_status;

  solution_pre.value_valid = (bool)value_valid_pre;
  solution_pre.dual_valid = (bool)dual_valid_pre;
  solution_pre.col_value =
      vector<double>(col_value_pre, col_value_pre + nCols_pre);
  solution_pre.col_dual =
      vector<double>(col_dual_pre, col_dual_pre + nCols_pre);
  solution_pre.row_value =
      vector<double>(row_value_pre, row_value_pre + nRows_pre);
  solution_pre.row_dual =
      vector<double>(row_dual_pre, row_dual_pre + nRows_pre);

  // postsolve
  return_status = ((Highs *)model)->postsolve(solution_pre);
  assert(return_status == HighsStatus::kOk);

  // assign solution
  HighsSolution solution_org = ((Highs *)model)->getSolution();
  assert(solution_org.col_value.size() == nCols_org);
  assert(solution_org.col_dual.size() == nCols_org);
  assert(solution_org.row_value.size() == nRows_org);
  assert(solution_org.row_dual.size() == nRows_org);

  for (int i = 0; i < nCols_org; i++) {
    col_value_org[i] = solution_org.col_value[i];
    col_dual_org[i] = solution_org.col_dual[i];
  }

  for (int i = 0; i < nRows_org; i++) {
    row_value_org[i] = solution_org.row_value[i];
    row_dual_org[i] = solution_org.row_dual[i];
  }

  return model;
}

void getModelSize_highs(void *model, int *nCols, int *nRows, int *nnz) {
  const HighsLp &lp = ((Highs *)model)->getLp();

  if (nCols) {
    *nCols = lp.num_col_;
  }
  if (nRows) {
    *nRows = lp.num_row_;
  }
  if (nnz) {
    *nnz = lp.a_matrix_.numNz();
  }

  return;
}

void writeJsonFromHiGHS_highs(const char *fout, void *model) {
  FILE *fptr;

  printf("--------------------------------\n");
  printf("--- saving to %s\n", fout);
  printf("--------------------------------\n");

  // Open a file in writing mode
  fptr = fopen(fout, "w");

  fprintf(fptr, "{");
  fprintf(fptr, "\n");

  // solver
  fprintf(fptr, "\"solver\":\"%s\",", "HiGHS presolver");

  // status
  HighsModelStatus model_status = ((Highs *)model)->getModelStatus();
  string model_status_str = ((Highs *)model)->modelStatusToString(model_status);
  const char *model_status_c_str = nullptr;
  model_status_c_str = model_status_str.c_str();
  fprintf(fptr, "\"terminationCode\":\"%s\",", model_status_c_str);

  // objective
  double objective = ((Highs *)model)->getObjectiveValue();
  fprintf(fptr, "\"dPrimalObj\":%.14f,", objective);
  fprintf(fptr, "\"dDualObj\":%.14f", objective);

  fprintf(fptr, "\n");
  fprintf(fptr, "}");
  // Close the file
  fclose(fptr);

  // cout << model_status_c_str << endl;

  return;
}

void postsolveModelFromEmpty_highs(void *model) {
  HighsStatus return_status;

  HighsSolution sol;
  HighsBasis basis;
  sol.value_valid = true;
  sol.dual_valid = true;
  basis.valid = true;

  return_status = ((Highs *)model)->postsolve(sol, basis);
  assert(return_status == HighsStatus::kOk);
}

void writeSolFromHiGHS_highs(const char *fout, void *model) {
  const HighsLp &lp = ((Highs *)model)->getLp();
  HighsSolution sol = ((Highs *)model)->getSolution();

  int nCols = lp.num_col_;
  int nRows = lp.num_row_;
  double *col_value = sol.col_value.data();
  double *col_dual = sol.col_dual.data();
  double *row_value = sol.row_value.data();
  double *row_dual = sol.row_dual.data();

  FILE *fptr;

  printf("--------------------------------\n");
  printf("--- saving sol to %s\n", fout);
  printf("--------------------------------\n");
  // Open a file in writing mode
  fptr = fopen(fout, "w");
  fprintf(fptr, "{");

  // nCols
  fprintf(fptr, "\n");

  fprintf(fptr, "\"nCols\": %d", nCols);

  // nRows
  fprintf(fptr, ",\n");

  fprintf(fptr, "\"nRows\": %d", nRows);

  // col value
  fprintf(fptr, ",\n");

  fprintf(fptr, "\"col_value\": [");
  if (col_value && nCols) {
    for (int i = 0; i < nCols - 1; ++i) {
      fprintf(fptr, "%.14f,", col_value[i]);
    }
    fprintf(fptr, "%.14f", col_value[nCols - 1]);
  }
  fprintf(fptr, "]");

  // col dual
  fprintf(fptr, ",\n");
  fprintf(fptr, "\"col_dual\": [");
  if (col_dual && nCols) {
    for (int i = 0; i < nCols - 1; ++i) {
      fprintf(fptr, "%.14f,", col_dual[i]);
    }
    fprintf(fptr, "%.14f", col_dual[nCols - 1]);
  }
  fprintf(fptr, "]");

  // row value
  fprintf(fptr, ",\n");
  fprintf(fptr, "\"row_value\": [");
  if (row_value && nRows) {
    for (int i = 0; i < nRows - 1; ++i) {
      fprintf(fptr, "%.14f,", row_value[i]);
    }
    fprintf(fptr, "%.14f", row_value[nRows - 1]);
  }
  fprintf(fptr, "]");

  // row dual
  fprintf(fptr, ",\n");
  fprintf(fptr, "\"row_dual\": [");
  if (row_dual && nRows) {
    for (int i = 0; i < nRows - 1; ++i) {
      fprintf(fptr, "%.14f,", row_dual[i]);
    }
    fprintf(fptr, "%.14f", row_dual[nRows - 1]);
  }
  fprintf(fptr, "]");

  // end writing
  fprintf(fptr, "\n");
  fprintf(fptr, "}");

  // Close the file
  fclose(fptr);
}

/*
 * formulate
 *                A x =  b
 *         l1 <= G1 x
 *               G2 x <= u2
 *         l3 <= G3 x <= u3
 * with bounds
 *             l <= x <= u
 * as
 *                A x =  b
 *               G3 x - z = 0
 *               G1 x >= l1
 *              -G2 x >= -u2
 * with bounds
 *             l <= x <= u
 *            l3 <= z <= u3
 * do not pre-allocate pointers except model, nCols, nRows, nnz and nEqs
 * set them to NULL is a better practice
 * but do remember to free them
 */
extern "C" int formulateLP_highs(void *model, double **cost, int *nCols,
                                 int *nRows, int *nnz, int *nEqs, int **csc_beg,
                                 int **csc_idx, double **csc_val, double **rhs,
                                 double **lower, double **upper, double *offset,
                                 double *sense_origin, int *nCols_origin,
                                 int **constraint_new_idx,
                                 int **constraint_type) {
  int retcode = 0;

  const HighsLp &lp = ((Highs *)model)->getLp();

  // problem size for malloc
  int nCols_highs = lp.num_col_;
  int nRows_highs = lp.num_row_;
  int nnz_highs = lp.a_matrix_.numNz();
  *nCols_origin = nCols_highs;
  *nRows = nRows_highs;  // need not recalculate
  *nCols = nCols_highs;  // need recalculate
  *nEqs = 0;             // need recalculate
  *nnz = nnz_highs;      // need recalculate
  *offset = lp.offset_;  // need not recalculate
  if (lp.sense_ == ObjSense::kMinimize) {
    *sense_origin = 1.0;
    //printf("Minimize\n");
  } else if (lp.sense_ == ObjSense::kMaximize) {
    *sense_origin = -1.0;
    //printf("Maximize\n");
  }
  //if (*offset != 0.0) {
  //  printf("Has obj offset %f\n", *offset);
  //} else {
  //  printf("No obj offset\n");
  //}
  // allocate buffer memory

  const double *lhs_highs = lp.row_lower_.data();
  const double *rhs_highs = lp.row_upper_.data();
  const int *A_csc_beg = lp.a_matrix_.start_.data();
  const int *A_csc_idx = lp.a_matrix_.index_.data();
  const double *A_csc_val = lp.a_matrix_.value_.data();
  int has_lower, has_upper;

  CUPDLP_INIT(*constraint_type, nRows_highs);
  CUPDLP_INIT(*constraint_new_idx, *nRows);

  // recalculate nRows and nnz for Ax - z = 0
  for (int i = 0; i < nRows_highs; i++) {
    has_lower = lhs_highs[i] > -1e20;
    has_upper = rhs_highs[i] < 1e20;

    // count number of equations and rows
    if (has_lower && has_upper && lhs_highs[i] == rhs_highs[i]) {
      (*constraint_type)[i] = EQ;
      (*nEqs)++;
    } else if (has_lower && !has_upper) {
      (*constraint_type)[i] = GEQ;
    } else if (!has_lower && has_upper) {
      (*constraint_type)[i] = LEQ;
    } else if (has_lower && has_upper) {
      (*constraint_type)[i] = BOUND;
      (*nCols)++;
      (*nnz)++;
      (*nEqs)++;
    } else {
      // printf("Error: constraint %d has no lower and upper bound\n", i);
      // retcode = 1;
      // goto exit_cleanup;

      // what if regard free as bounded
      printf("Warning: constraint %d has no lower and upper bound\n", i);
      (*constraint_type)[i] = BOUND;
      (*nCols)++;
      (*nnz)++;
      (*nEqs)++;
    }
  }

  // allocate memory
  CUPDLP_INIT(*cost, *nCols);
  CUPDLP_INIT(*lower, *nCols);
  CUPDLP_INIT(*upper, *nCols);
  CUPDLP_INIT(*csc_beg, *nCols + 1);
  CUPDLP_INIT(*csc_idx, *nnz);
  CUPDLP_INIT(*csc_val, *nnz);
  CUPDLP_INIT(*rhs, *nRows);

  // cost, lower, upper
  for (int i = 0; i < nCols_highs; i++) {
    (*cost)[i] = lp.col_cost_[i] * (*sense_origin);
    (*lower)[i] = lp.col_lower_[i];

    (*upper)[i] = lp.col_upper_[i];
  }
  // slack costs
  for (int i = nCols_highs; i < *nCols; i++) {
    (*cost)[i] = 0.0;
  }
  // slack bounds
  for (int i = 0, j = nCols_highs; i < *nRows; i++) {
    if ((*constraint_type)[i] == BOUND) {
      (*lower)[j] = lhs_highs[i];
      (*upper)[j] = rhs_highs[i];
      j++;
    }
  }

  for (int i = 0; i < *nCols; i++) {
    if ((*lower)[i] < -1e20) (*lower)[i] = -INFINITY;
    if ((*upper)[i] > 1e20) (*upper)[i] = INFINITY;
  }

  // permute LP rhs
  // EQ or BOUND first
  for (int i = 0, j = 0; i < *nRows; i++) {
    if ((*constraint_type)[i] == EQ) {
      (*rhs)[j] = lhs_highs[i];
      (*constraint_new_idx)[i] = j;
      j++;
    } else if ((*constraint_type)[i] == BOUND) {
      (*rhs)[j] = 0.0;
      (*constraint_new_idx)[i] = j;
      j++;
    }
  }
  // then LEQ or GEQ
  for (int i = 0, j = *nEqs; i < *nRows; i++) {
    if ((*constraint_type)[i] == LEQ) {
      (*rhs)[j] = -rhs_highs[i];  // multiply -1
      (*constraint_new_idx)[i] = j;
      j++;
    } else if ((*constraint_type)[i] == GEQ) {
      (*rhs)[j] = lhs_highs[i];
      (*constraint_new_idx)[i] = j;
      j++;
    }
  }

  // formulate and permute LP matrix
  // beg remains the same
  for (int i = 0; i < nCols_highs + 1; i++) (*csc_beg)[i] = A_csc_beg[i];
  for (int i = nCols_highs + 1; i < *nCols + 1; i++)
    (*csc_beg)[i] = (*csc_beg)[i - 1] + 1;

  // row idx changes
  for (int i = 0, k = 0; i < nCols_highs; i++) {
    // same order as in rhs
    // EQ or BOUND first
    for (int j = (*csc_beg)[i]; j < (*csc_beg)[i + 1]; j++) {
      if ((*constraint_type)[A_csc_idx[j]] == EQ ||
          (*constraint_type)[A_csc_idx[j]] == BOUND) {
        (*csc_idx)[k] = (*constraint_new_idx)[A_csc_idx[j]];
        (*csc_val)[k] = A_csc_val[j];
        k++;
      }
    }
    // then LEQ or GEQ
    for (int j = (*csc_beg)[i]; j < (*csc_beg)[i + 1]; j++) {
      if ((*constraint_type)[A_csc_idx[j]] == LEQ) {
        (*csc_idx)[k] = (*constraint_new_idx)[A_csc_idx[j]];
        (*csc_val)[k] = -A_csc_val[j];  // multiply -1
        k++;
      } else if ((*constraint_type)[A_csc_idx[j]] == GEQ) {
        (*csc_idx)[k] = (*constraint_new_idx)[A_csc_idx[j]];
        (*csc_val)[k] = A_csc_val[j];
        k++;
      }
    }
  }

  // slacks for BOUND
  for (int i = 0, j = nCols_highs; i < *nRows; i++) {
    if ((*constraint_type)[i] == BOUND) {
      (*csc_idx)[(*csc_beg)[j]] = (*constraint_new_idx)[i];
      (*csc_val)[(*csc_beg)[j]] = -1.0;
      j++;
    }
  }

exit_cleanup:

  return retcode;
}