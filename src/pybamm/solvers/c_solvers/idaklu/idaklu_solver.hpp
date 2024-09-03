#ifndef PYBAMM_CREATE_IDAKLU_SOLVER_HPP
#define PYBAMM_CREATE_IDAKLU_SOLVER_HPP

#include "IDAKLUSolverOpenMP_solvers.hpp"
#include <idas/idas.h>
#include <memory>

/**
 * Creates a concrete solver given a linear solver, as specified in
 * options_cpp.linear_solver.
 * @brief Create a concrete solver given a linear solver
 */
template<class ExprSet>
IDAKLUSolver *create_idaklu_solver(
  int number_of_states,
  int number_of_parameters,
  const typename ExprSet::BaseFunctionType &rhs_alg,
  const typename ExprSet::BaseFunctionType &jac_times_cjmass,
  const np_array_int &jac_times_cjmass_colptrs,
  const np_array_int &jac_times_cjmass_rowvals,
  const int jac_times_cjmass_nnz,
  const int jac_bandwidth_lower,
  const int jac_bandwidth_upper,
  const typename ExprSet::BaseFunctionType &jac_action,
  const typename ExprSet::BaseFunctionType &mass_action,
  const typename ExprSet::BaseFunctionType &sens,
  const typename ExprSet::BaseFunctionType &events,
  const int number_of_events,
  np_array rhs_alg_id,
  np_array atol_np,
  double rel_tol,
  int inputs_length,
  const std::vector<typename ExprSet::BaseFunctionType*>& var_fcns,
  const std::vector<typename ExprSet::BaseFunctionType*>& dvar_dy_fcns,
  const std::vector<typename ExprSet::BaseFunctionType*>& dvar_dp_fcns,
  py::dict py_opts
) {
  auto setup_opts = SetupOptions(py_opts);
  auto solver_opts = SolverOptions(py_opts);
  auto functions = std::make_unique<ExprSet>(
    rhs_alg,
    jac_times_cjmass,
    jac_times_cjmass_nnz,
    jac_bandwidth_lower,
    jac_bandwidth_upper,
    jac_times_cjmass_rowvals,
    jac_times_cjmass_colptrs,
    inputs_length,
    jac_action,
    mass_action,
    sens,
    events,
    number_of_states,
    number_of_events,
    number_of_parameters,
    var_fcns,
    dvar_dy_fcns,
    dvar_dp_fcns,
    setup_opts
  );

  IDAKLUSolver *idakluSolver = nullptr;

  // Instantiate solver class
  if (setup_opts.linear_solver == "SUNLinSol_Dense")
  {
    DEBUG("\tsetting SUNLinSol_Dense linear solver");
    idakluSolver = new IDAKLUSolverOpenMP_Dense<ExprSet>(
      atol_np,
      rel_tol,
      rhs_alg_id,
      number_of_parameters,
      number_of_events,
      jac_times_cjmass_nnz,
      jac_bandwidth_lower,
      jac_bandwidth_upper,
      std::move(functions),
      setup_opts,
      solver_opts
     );
  }
  else if (setup_opts.linear_solver == "SUNLinSol_KLU")
  {
    DEBUG("\tsetting SUNLinSol_KLU linear solver");
    idakluSolver = new IDAKLUSolverOpenMP_KLU<ExprSet>(
      atol_np,
      rel_tol,
      rhs_alg_id,
      number_of_parameters,
      number_of_events,
      jac_times_cjmass_nnz,
      jac_bandwidth_lower,
      jac_bandwidth_upper,
      std::move(functions),
      setup_opts,
      solver_opts
     );
  }
  else if (setup_opts.linear_solver == "SUNLinSol_Band")
  {
    DEBUG("\tsetting SUNLinSol_Band linear solver");
    idakluSolver = new IDAKLUSolverOpenMP_Band<ExprSet>(
      atol_np,
      rel_tol,
      rhs_alg_id,
      number_of_parameters,
      number_of_events,
      jac_times_cjmass_nnz,
      jac_bandwidth_lower,
      jac_bandwidth_upper,
      std::move(functions),
      setup_opts,
      solver_opts
     );
  }
  else if (setup_opts.linear_solver == "SUNLinSol_SPBCGS")
  {
    DEBUG("\tsetting SUNLinSol_SPBCGS_linear solver");
    idakluSolver = new IDAKLUSolverOpenMP_SPBCGS<ExprSet>(
      atol_np,
      rel_tol,
      rhs_alg_id,
      number_of_parameters,
      number_of_events,
      jac_times_cjmass_nnz,
      jac_bandwidth_lower,
      jac_bandwidth_upper,
      std::move(functions),
      setup_opts,
      solver_opts
     );
  }
  else if (setup_opts.linear_solver == "SUNLinSol_SPFGMR")
  {
    DEBUG("\tsetting SUNLinSol_SPFGMR_linear solver");
    idakluSolver = new IDAKLUSolverOpenMP_SPFGMR<ExprSet>(
      atol_np,
      rel_tol,
      rhs_alg_id,
      number_of_parameters,
      number_of_events,
      jac_times_cjmass_nnz,
      jac_bandwidth_lower,
      jac_bandwidth_upper,
      std::move(functions),
      setup_opts,
      solver_opts
     );
  }
  else if (setup_opts.linear_solver == "SUNLinSol_SPGMR")
  {
    DEBUG("\tsetting SUNLinSol_SPGMR solver");
    idakluSolver = new IDAKLUSolverOpenMP_SPGMR<ExprSet>(
      atol_np,
      rel_tol,
      rhs_alg_id,
      number_of_parameters,
      number_of_events,
      jac_times_cjmass_nnz,
      jac_bandwidth_lower,
      jac_bandwidth_upper,
      std::move(functions),
      setup_opts,
      solver_opts
     );
  }
  else if (setup_opts.linear_solver == "SUNLinSol_SPTFQMR")
  {
    DEBUG("\tsetting SUNLinSol_SPGMR solver");
    idakluSolver = new IDAKLUSolverOpenMP_SPTFQMR<ExprSet>(
      atol_np,
      rel_tol,
      rhs_alg_id,
      number_of_parameters,
      number_of_events,
      jac_times_cjmass_nnz,
      jac_bandwidth_lower,
      jac_bandwidth_upper,
      std::move(functions),
      setup_opts,
      solver_opts
     );
  }

  if (idakluSolver == nullptr) {
    throw std::invalid_argument("Unsupported solver requested");
  }

  return idakluSolver;
}

template<class ExprSet>
void process_time_series(
    const std::vector<py::array_t<double>>& ts_np,
    const std::vector<py::array_t<double>>& ys_np,
    const std::vector<np_array_dense>& inputs_np,
    const std::vector<const typename ExprSet::BaseFunctionType*>& funcs,
    double* out,
    const bool is_f_contiguous,
    const int len
) {
    // Buffer for non-f-contiguous arrays
    std::vector<double> y_buffer;

    int count = 0;
    for (size_t i = 0; i < ts_np.size(); i++) {
        const auto& t_i = ts_np[i].unchecked<1>();
        const auto& y_i = ys_np[i].unchecked<2>();  // y_i is 2D
        const auto inputs_data_i = inputs_np[i].data();
        const auto func_i = *funcs[i];

        int M = y_i.shape(0);
        if (!is_f_contiguous && y_buffer.size() < M) {
            y_buffer.resize(M); // Resize the buffer
        }

        for (size_t j = 0; j < t_i.size(); j++) {
            const double t_ij = t_i(j);

            // Use a view of y_i
            if (!is_f_contiguous) {
                for (int k = 0; k < M; k++) {
                    y_buffer[k] = y_i(k, j);
                }
            }
            const double* y_ij = is_f_contiguous ? &y_i(0, j) : y_buffer.data();

            // Prepare CasADi function arguments
            std::vector<const double*> args = { &t_ij, y_ij, inputs_data_i };
            std::vector<double*> results = { &out[count] };
            // Call the CasADi function with proper arguments
            (func_i)(args, results);

            count += len;
        }
    }
}

template<class ExprSet>
const py::array_t<double> observe_0D(
    const std::vector<py::array_t<double>>& ts_np,
    const std::vector<py::array_t<double>>& ys_np,
    const std::vector<np_array_dense>& inputs_np,
    const std::vector<const typename ExprSet::BaseFunctionType*>& funcs,
    const bool is_f_contiguous,
    const int size0
) {
    // Create a numpy array to manage the output
    py::array_t<double> out_array(size0);
    auto out = out_array.mutable_data();

    process_time_series<ExprSet>(
        ts_np, ys_np, inputs_np, funcs, out, is_f_contiguous, 1
    );

    return out_array;
}

template<class ExprSet>
const py::array_t<double> observe_1D(
    const std::vector<py::array_t<double>>& ts_np,
    const std::vector<py::array_t<double>>& ys_np,
    const std::vector<np_array_dense>& inputs_np,
    const std::vector<const typename ExprSet::BaseFunctionType*>& funcs,
    const bool is_f_contiguous,
    const int size0,
    const int size1
) {
    // Create a numpy array to manage the output
    py::array_t<double, py::array::f_style> out_array({size1, size0});
    auto out = out_array.mutable_data();

    process_time_series<ExprSet>(
        ts_np, ys_np, inputs_np, funcs, out, is_f_contiguous, size1
    );

    return out_array;
}

template<class ExprSet>
const py::array_t<double> observe_2D(
    const std::vector<py::array_t<double>>& ts_np,
    const std::vector<py::array_t<double>>& ys_np,
    const std::vector<np_array_dense>& inputs_np,
    const std::vector<const typename ExprSet::BaseFunctionType*>& funcs,
    const bool is_f_contiguous,
    const int size0,
    const int size1,
    const int size2
) {
    // Create a numpy array to manage the output
    py::array_t<double, py::array::f_style> out_array({size1, size2, size0});
    auto out = out_array.mutable_data();

    process_time_series<ExprSet>(
        ts_np, ys_np, inputs_np, funcs, out, is_f_contiguous, size1 * size2
    );

    return out_array;
}

#endif // PYBAMM_CREATE_IDAKLU_SOLVER_HPP
