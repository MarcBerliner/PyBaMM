#include "Expressions/Expressions.hpp"
#include "sundials_functions.hpp"

template <class ExprSet>
IDAKLUSolverOpenMP<ExprSet>::IDAKLUSolverOpenMP(
  np_array atol_np_input,
  double rel_tol,
  np_array rhs_alg_id_input,
  int number_of_parameters_input,
  int number_of_events_input,
  int jac_times_cjmass_nnz_input,
  int jac_bandwidth_lower_input,
  int jac_bandwidth_upper_input,
  std::unique_ptr<ExprSet> functions_arg,
  const SetupOptions &setup_input,
  const SolverOptions &solver_input
) :
  atol_np(atol_np_input),
  rhs_alg_id(rhs_alg_id_input),
  number_of_states(atol_np_input.request().size),
  number_of_parameters(number_of_parameters_input),
  number_of_events(number_of_events_input),
  jac_times_cjmass_nnz(jac_times_cjmass_nnz_input),
  jac_bandwidth_lower(jac_bandwidth_lower_input),
  jac_bandwidth_upper(jac_bandwidth_upper_input),
  functions(std::move(functions_arg)),
  setup_opts(setup_input),
  solver_opts(solver_input)
{
  // Construction code moved to Initialize() which is called from the
  // (child) IDAKLUSolver_* class constructors.
  DEBUG("IDAKLUSolverOpenMP:IDAKLUSolverOpenMP");
  auto atol = atol_np.unchecked<1>();

  // create SUNDIALS context object
  SUNContext_Create(NULL, &sunctx);  // calls null-wrapper if Sundials Ver<6

  // allocate memory for solver
  ida_mem = IDACreate(sunctx);

  // create the vector of initial values
  AllocateVectors();
  if (number_of_parameters > 0) {
    yyS = N_VCloneVectorArray(number_of_parameters, yy);
    ypS = N_VCloneVectorArray(number_of_parameters, yp);
  }
  // set initial values
  realtype *atval = N_VGetArrayPointer(avtol);
  for (int i = 0; i < number_of_states; i++) {
    atval[i] = atol[i];
  }

  for (int is = 0; is < number_of_parameters; is++) {
    N_VConst(RCONST(0.0), yyS[is]);
    N_VConst(RCONST(0.0), ypS[is]);
  }

  // create Matrix objects
  SetMatrix();

  // initialise solver
  IDAInit(ida_mem, residual_eval<ExprSet>, 0, yy, yp);

  // set tolerances
  rtol = RCONST(rel_tol);
  IDASVtolerances(ida_mem, rtol, avtol);

  // Set events
  IDARootInit(ida_mem, number_of_events, events_eval<ExprSet>);

  // Set user data
  void *user_data = functions.get();
  IDASetUserData(ida_mem, user_data);

  // Specify preconditioner type
  precon_type = SUN_PREC_NONE;
  if (this->setup_opts.preconditioner != "none") {
    precon_type = SUN_PREC_LEFT;
  }
}

template <class ExprSet>
void IDAKLUSolverOpenMP<ExprSet>::AllocateVectors() {
  // Create vectors
  yy = N_VNew_OpenMP(number_of_states, setup_opts.num_threads, sunctx);
  yp = N_VNew_OpenMP(number_of_states, setup_opts.num_threads, sunctx);
  avtol = N_VNew_OpenMP(number_of_states, setup_opts.num_threads, sunctx);
  id = N_VNew_OpenMP(number_of_states, setup_opts.num_threads, sunctx);
}

template <class ExprSet>
void IDAKLUSolverOpenMP<ExprSet>::SetSolverOptions() {
  // Maximum order of the linear multistep method
  CheckErrors(IDASetMaxOrd(ida_mem, solver_opts.max_order_bdf));

  // Maximum number of steps to be taken by the solver in its attempt to reach
  // the next output time
  CheckErrors(IDASetMaxNumSteps(ida_mem, solver_opts.max_num_steps));

  // Initial step size
  CheckErrors(IDASetInitStep(ida_mem, solver_opts.dt_init));

  // Maximum absolute step size
  CheckErrors(IDASetMaxStep(ida_mem, solver_opts.dt_max));

  // Maximum number of error test failures in attempting one step
  CheckErrors(IDASetMaxErrTestFails(ida_mem, solver_opts.max_error_test_failures));

  // Maximum number of nonlinear solver iterations at one step
  CheckErrors(IDASetMaxNonlinIters(ida_mem, solver_opts.max_nonlinear_iterations));

  // Maximum number of nonlinear solver convergence failures at one step
  CheckErrors(IDASetMaxConvFails(ida_mem, solver_opts.max_convergence_failures));

  // Safety factor in the nonlinear convergence test
  CheckErrors(IDASetNonlinConvCoef(ida_mem, solver_opts.nonlinear_convergence_coefficient));

  // Suppress algebraic variables from error test
  CheckErrors(IDASetSuppressAlg(ida_mem, solver_opts.suppress_algebraic_error));

  // Positive constant in the Newton iteration convergence test within the initial
  // condition calculation
  CheckErrors(IDASetNonlinConvCoefIC(ida_mem, solver_opts.nonlinear_convergence_coefficient_ic));

  // Maximum number of steps allowed when icopt=IDA_YA_YDP_INIT in IDACalcIC
  CheckErrors(IDASetMaxNumStepsIC(ida_mem, solver_opts.max_num_steps_ic));

  // Maximum number of the approximate Jacobian or preconditioner evaluations
  // allowed when the Newton iteration appears to be slowly converging
  CheckErrors(IDASetMaxNumJacsIC(ida_mem, solver_opts.max_num_jacobians_ic));

  // Maximum number of Newton iterations allowed in any one attempt to solve
  // the initial conditions calculation problem
  CheckErrors(IDASetMaxNumItersIC(ida_mem, solver_opts.max_num_iterations_ic));

  // Maximum number of linesearch backtracks allowed in any Newton iteration,
  // when solving the initial conditions calculation problem
  CheckErrors(IDASetMaxBacksIC(ida_mem, solver_opts.max_linesearch_backtracks_ic));

  // Turn off linesearch
  CheckErrors(IDASetLineSearchOffIC(ida_mem, solver_opts.linesearch_off_ic));

  // Ratio between linear and nonlinear tolerances
  CheckErrors(IDASetEpsLin(ida_mem, solver_opts.epsilon_linear_tolerance));

  // Increment factor used in DQ Jv approximation
  CheckErrors(IDASetIncrementFactor(ida_mem, solver_opts.increment_factor));

  int LS_type = SUNLinSolGetType(LS);
  if (LS_type == SUNLINEARSOLVER_DIRECT || LS_type == SUNLINEARSOLVER_MATRIX_ITERATIVE) {
    // Enable or disable linear solution scaling
    CheckErrors(IDASetLinearSolutionScaling(ida_mem, solver_opts.linear_solution_scaling));
  }
}



template <class ExprSet>
void IDAKLUSolverOpenMP<ExprSet>::SetMatrix() {
  // Create Matrix object
  if (setup_opts.jacobian == "sparse") {
    DEBUG("\tsetting sparse matrix");
    J = SUNSparseMatrix(
      number_of_states,
      number_of_states,
      jac_times_cjmass_nnz,
      CSC_MAT,
      sunctx
    );
  } else if (setup_opts.jacobian == "banded") {
    DEBUG("\tsetting banded matrix");
    J = SUNBandMatrix(
      number_of_states,
      jac_bandwidth_upper,
      jac_bandwidth_lower,
      sunctx
    );
  } else if (setup_opts.jacobian == "dense" || setup_opts.jacobian == "none") {
    DEBUG("\tsetting dense matrix");
    J = SUNDenseMatrix(
      number_of_states,
      number_of_states,
      sunctx
    );
  } else if (setup_opts.jacobian == "matrix-free") {
    DEBUG("\tsetting matrix-free");
    J = NULL;
  } else {
    throw std::invalid_argument("Unsupported matrix requested");
  }
}

template <class ExprSet>
void IDAKLUSolverOpenMP<ExprSet>::Initialize() {
  // Call after setting the solver

  // attach the linear solver
  if (LS == nullptr) {
    throw std::invalid_argument("Linear solver not set");
  }
  CheckErrors(IDASetLinearSolver(ida_mem, LS, J));

  if (setup_opts.preconditioner != "none") {
    DEBUG("\tsetting IDADDB preconditioner");
    // setup preconditioner
    CheckErrors(IDABBDPrecInit(
      ida_mem, number_of_states, setup_opts.precon_half_bandwidth,
      setup_opts.precon_half_bandwidth, setup_opts.precon_half_bandwidth_keep,
      setup_opts.precon_half_bandwidth_keep, 0.0, residual_eval_approx<ExprSet>, NULL));
  }

  if (setup_opts.jacobian == "matrix-free") {
    CheckErrors(IDASetJacTimes(ida_mem, NULL, jtimes_eval<ExprSet>));
  } else if (setup_opts.jacobian != "none") {
    CheckErrors(IDASetJacFn(ida_mem, jacobian_eval<ExprSet>));
  }

  if (number_of_parameters > 0) {
    CheckErrors(IDASensInit(ida_mem, number_of_parameters, IDA_SIMULTANEOUS,
      sensitivities_eval<ExprSet>, yyS, ypS));
    CheckErrors(IDASensEEtolerances(ida_mem));
  }

  CheckErrors(SUNLinSolInitialize(LS));

  auto id_np_val = rhs_alg_id.unchecked<1>();
  realtype *id_val;
  id_val = N_VGetArrayPointer(id);

  int ii;
  for (ii = 0; ii < number_of_states; ii++) {
    id_val[ii] = id_np_val[ii];
  }

  // Variable types: differential (1) and algebraic (0)
  CheckErrors(IDASetId(ida_mem, id));
}

template <class ExprSet>
IDAKLUSolverOpenMP<ExprSet>::~IDAKLUSolverOpenMP() {
  bool sensitivity = number_of_parameters > 0;
  // Free memory
  if (sensitivity) {
      IDASensFree(ida_mem);
  }

  CheckErrors(SUNLinSolFree(LS));

  SUNMatDestroy(J);
  N_VDestroy(avtol);
  N_VDestroy(yy);
  N_VDestroy(yp);
  N_VDestroy(id);

  if (sensitivity) {
    N_VDestroyVectorArray(yyS, number_of_parameters);
    N_VDestroyVectorArray(ypS, number_of_parameters);
  }

  IDAFree(&ida_mem);
  SUNContext_Free(&sunctx);
}

template <class ExprSet>
void IDAKLUSolverOpenMP<ExprSet>::CalcVars(
    realtype *y_return,
    size_t length_of_return_vector,
    size_t teval_i,
    realtype *tret,
    realtype *yval,
    const std::vector<realtype*>& ySval,
    realtype *yS_return,
    size_t *ySk
) {
  DEBUG("IDAKLUSolver::CalcVars");
  // Evaluate functions for each requested variable and store
  size_t j = 0;
  for (auto& var_fcn : functions->var_fcns) {
    (*var_fcn)({tret, yval, functions->inputs.data()}, {&res[0]});
    // store in return vector
    for (size_t jj=0; jj<var_fcn->nnz_out(); jj++) {
      y_return[teval_i*length_of_return_vector + j++] = res[jj];
    }
  }
  // calculate sensitivities
  CalcVarsSensitivities(tret, yval, ySval, yS_return, ySk);
}

template <class ExprSet>
void IDAKLUSolverOpenMP<ExprSet>::CalcVarsSensitivities(
    realtype *tret,
    realtype *yval,
    const std::vector<realtype*>& ySval,
    realtype *yS_return,
    size_t *ySk
) {
  DEBUG("IDAKLUSolver::CalcVarsSensitivities");
  // Calculate sensitivities
  std::vector<realtype> dens_dvar_dp = std::vector<realtype>(number_of_parameters, 0);
  for (size_t dvar_k = 0; dvar_k < functions->dvar_dy_fcns.size(); dvar_k++) {
    // Isolate functions
    Expression* dvar_dy = functions->dvar_dy_fcns[dvar_k];
    Expression* dvar_dp = functions->dvar_dp_fcns[dvar_k];
    // Calculate dvar/dy
    (*dvar_dy)({tret, yval, functions->inputs.data()}, {&res_dvar_dy[0]});
    // Calculate dvar/dp and convert to dense array for indexing
    (*dvar_dp)({tret, yval, functions->inputs.data()}, {&res_dvar_dp[0]});
    for (int k=0; k<number_of_parameters; k++) {
      dens_dvar_dp[k] = 0;
    }
    for (int k=0; k<dvar_dp->nnz_out(); k++) {
      dens_dvar_dp[dvar_dp->get_row()[k]] = res_dvar_dp[k];
    }
    // Calculate sensitivities
    for (int paramk = 0; paramk < number_of_parameters; paramk++) {
      yS_return[*ySk] = dens_dvar_dp[paramk];
      for (int spk = 0; spk < dvar_dy->nnz_out(); spk++) {
        yS_return[*ySk] += res_dvar_dy[spk] * ySval[paramk][dvar_dy->get_col()[spk]];
      }
      (*ySk)++;
    }
  }
}

template <class ExprSet>
Solution IDAKLUSolverOpenMP<ExprSet>::solve(
    np_array t_np,
    np_array t_saveat_np,
    np_array y0_np,
    np_array yp0_np,
    np_array_dense inputs
)
{
  DEBUG("IDAKLUSolver::solve");

  int number_of_timesteps = t_np.request().size;
  auto t = t_np.unchecked<1>();
  auto t_saveat = t_saveat_np.unchecked<1>();
  realtype t0 = RCONST(t(0));
  auto y0 = y0_np.unchecked<1>();
  auto yp0 = yp0_np.unchecked<1>();
  auto n_coeffs = number_of_states + number_of_parameters * number_of_states;
  bool const sensitivity = number_of_parameters > 0;

  if (y0.size() != n_coeffs) {
    throw std::domain_error(
      "y0 has wrong size. Expected " + std::to_string(n_coeffs) +
      " but got " + std::to_string(y0.size()));
  }

  if (yp0.size() != n_coeffs) {
    throw std::domain_error(
      "yp0 has wrong size. Expected " + std::to_string(n_coeffs) +
      " but got " + std::to_string(yp0.size()));
  }

  // set inputs
  auto p_inputs = inputs.unchecked<2>();
  for (int i = 0; i < functions->inputs.size(); i++) {
    functions->inputs[i] = p_inputs(i, 0);
  }

  // set initial conditions
  realtype *yval = N_VGetArrayPointer(yy);
  realtype *ypval = N_VGetArrayPointer(yp);
  std::vector<realtype *> ySval(number_of_parameters);
  std::vector<realtype *> ypSval(number_of_parameters);
  for (int p = 0 ; p < number_of_parameters; p++) {
    ySval[p] = N_VGetArrayPointer(yyS[p]);
    ypSval[p] = N_VGetArrayPointer(ypS[p]);
    for (int i = 0; i < number_of_states; i++) {
      ySval[p][i] = y0[i + (p + 1) * number_of_states];
      ypSval[p][i] = yp0[i + (p + 1) * number_of_states];
    }
  }

  for (int i = 0; i < number_of_states; i++) {
    yval[i] = y0[i];
    ypval[i] = yp0[i];
  }

  SetSolverOptions();

  CheckErrors(IDAReInit(ida_mem, t0, yy, yp));
  if (sensitivity) {
    CheckErrors(IDASensReInit(ida_mem, IDA_SIMULTANEOUS, yyS, ypS));
  }

  // correct initial values
  int const init_type = solver_opts.init_all_y_ic ? IDA_Y_INIT : IDA_YA_YDP_INIT;
  if (solver_opts.calc_ic) {
    DEBUG("IDACalcIC");
    // IDACalcIC will throw a warning if it fails to find initial conditions
    IDACalcIC(ida_mem, init_type, t(1));
  }

  if (sensitivity) {
    CheckErrors(IDAGetSens(ida_mem, &t0, yyS));
  }

  realtype tret;
  realtype tfinal = t(number_of_timesteps - 1);
  realtype tprev = t0;

  // set return vectors
  int length_of_return_vector = 0;
  int length_of_final_sv_slice = 0;
  size_t max_res_size = 0;  // maximum result size (for common result buffer)
  size_t max_res_dvar_dy = 0, max_res_dvar_dp = 0;
  if (functions->var_fcns.size() > 0) {
    // return only the requested variables list after computation
    for (auto& var_fcn : functions->var_fcns) {
      max_res_size = std::max(max_res_size, size_t(var_fcn->out_shape(0)));
      length_of_return_vector += var_fcn->nnz_out();
      for (auto& dvar_fcn : functions->dvar_dy_fcns) {
        max_res_dvar_dy = std::max(max_res_dvar_dy, size_t(dvar_fcn->out_shape(0)));
      }
      for (auto& dvar_fcn : functions->dvar_dp_fcns) {
        max_res_dvar_dp = std::max(max_res_dvar_dp, size_t(dvar_fcn->out_shape(0)));
      }
      length_of_final_sv_slice = number_of_states;
    }
  } else {
    // Return full y state-vector
    length_of_return_vector = number_of_states;
  }
  realtype *t_return = new realtype[number_of_timesteps];
  realtype *y_return = new realtype[number_of_timesteps *
                                    length_of_return_vector];
  realtype *yS_return = new realtype[number_of_parameters *
                                     number_of_timesteps *
                                     length_of_return_vector];
  realtype *yterm_return = new realtype[length_of_final_sv_slice];

  std::vector<std::vector<realtype>> y_adaptive(1, std::vector<realtype>(number_of_states, 0.0));
  std::vector<realtype> t_adaptive(1, t(0));

  res.resize(max_res_size);
  res_dvar_dy.resize(max_res_dvar_dy);
  res_dvar_dp.resize(max_res_dvar_dp);

  // Initial state (teval_i=0)
  int teval_i = 0;
  size_t ySk = 0;
  t_return[teval_i] = t(teval_i);
  if (functions->var_fcns.size() > 0) {
    // Evaluate functions for each requested variable and store
    CalcVars(y_return, length_of_return_vector, teval_i,
             &tret, yval, ySval, yS_return, &ySk);
  } else {
    // Retain complete copy of the state vector
    for (int j = 0; j < number_of_states; j++) {
      y_return[j] = yval[j];
      y_adaptive[0][j] = yval[j];
    }
    for (int j = 0; j < number_of_parameters; j++) {
      const int base_index = j * number_of_timesteps * number_of_states;
      for (int k = 0; k < number_of_states; k++) {
        yS_return[base_index + k] = ySval[j][k];
      }
    }
  }

  // Subsequent states (teval_i>0)
  teval_i += 1;
  realtype t_next = t(teval_i);

  realtype t_save;
  int tsaveat_i = 0;
  bool use_t_adaptive = t_saveat.size() == 0;
  if (!use_t_adaptive) {
    t_save = t_saveat(tsaveat_i);
  }
  bool print_debug = false;
  IDASetStopTime(ida_mem, t_next);
  int retval;
  while (true) {
    DEBUG("IDASolve");
    retval = IDASolve(ida_mem, tfinal, &tret, yy, yp, IDA_ONE_STEP);

    if (retval < 0) {
      // failed
      break;
    } else if (tprev == tret) {
      // IDA sometimes returns the same time point twice
      // instead of erroring. Assign a retval and break
      retval = IDA_ERR_FAIL;
      break;
    }

    bool hit_final_time = tret >= tfinal;
    if (print_debug) { py::print("DEBUG tprev =  ", tprev); }
    if (print_debug) { py::print("DEBUG tret =   ", tret); }
    if (print_debug) { py::print("DEBUG tsav_i = ", tsaveat_i); }

    bool hit_saveat = !use_t_adaptive && t_save >= tprev;
    bool hit_discontinuity = retval == IDA_TSTOP_RETURN;
    bool hit_event = retval == IDA_ROOT_RETURN;

    if (hit_saveat) {
      // Save the state at the requested time
      while (tsaveat_i <= (t_saveat.size()-1) && t_save <= tret) {
        if (print_debug) { py::print("DEBUG saveat at t = ", t_save); }
        if (print_debug) { py::print("DEBUG tret at t =   ", tret); }

        if (t_save != t_next) {
          CheckErrors(IDAGetDky(ida_mem, t_save, 0, yy));

          t_adaptive.emplace_back(t_save);
          y_adaptive.emplace_back(number_of_states, 0.0);
          for (size_t j = 0; j < number_of_states; ++j) {
            y_adaptive.back()[j] = yval[j];
          }
        }

        tsaveat_i += 1;
        if (tsaveat_i == (t_saveat.size())) {
          // Reached the final t_saveat value
          break;
        }
        t_save = t_saveat(tsaveat_i);
      }
    }

    if (use_t_adaptive || hit_discontinuity || hit_event) {
      if (hit_saveat) {
        if (print_debug) { py::print("DEBUG IDAGetCurrentY"); }
        CheckErrors(IDAGetDky(ida_mem, tret, 0, yy));
      }

      if (print_debug) { py::print("DEBUG tadapt at t =  ", tret); }
      t_adaptive.emplace_back(tret);
      y_adaptive.emplace_back(number_of_states, 0.0);
      for (size_t j = 0; j < number_of_states; ++j) {
        y_adaptive.back()[j] = yval[j];
      }
    }

    if (sensitivity) {
      CheckErrors(IDAGetSens(ida_mem, &tret, yyS));
    }


    if (hit_event || hit_discontinuity) {
    // Evaluate and store results for the time step
    if (print_debug) { py::print("DEBUG discon at t =  ", tret); }
    t_return[teval_i] = tret;
    if (functions->var_fcns.size() > 0) {
      // Evaluate functions for each requested variable and store
      // NOTE: Indexing of yS_return is (time:var:param)
      CalcVars(y_return, length_of_return_vector, teval_i,
                &tret, yval, ySval, yS_return, &ySk);
    } else {
      // Retain complete copy of the state vector
      for (int j = 0; j < number_of_states; j++) {
        y_return[teval_i * number_of_states + j] = yval[j];
      }
      for (int j = 0; j < number_of_parameters; j++) {
        const int base_index =
          j * number_of_timesteps * number_of_states +
          teval_i * number_of_states;
        for (int k = 0; k < number_of_states; k++) {
          // NOTE: Indexing of yS_return is (time:param:yvec)
          yS_return[base_index + k] = ySval[j][k];
        }
      }
    }
    if (hit_discontinuity && !hit_final_time) {
      teval_i += 1;
      // Reinitialize the solver to deal with the discontinuity
      t_next = t(teval_i);
      IDACalcIC(ida_mem, init_type, t_next);
      CheckErrors(IDAReInit(ida_mem, tret, yy, yp));
      if (sensitivity) {
        CheckErrors(IDASensReInit(ida_mem, IDA_SIMULTANEOUS, yyS, ypS));
      }
      if (print_debug) { py::print("DEBUG Discontinuity at t = ", tret); }

      CheckErrors(IDASetStopTime(ida_mem, t_next));
    }
    }

    if (hit_final_time || retval == IDA_ROOT_RETURN) {
      if (functions->var_fcns.size() > 0) {
        // store final state slice if outout variables are specified
        yterm_return = yval;
      }
      break;
    }
    tprev = tret;
  }

  // IDAGetDky(ida_mem, 0.0, 0, yy);
  // for (size_t j = 0; j < number_of_states; ++j) {
  //   py::print(yval[j]);
  // }

  if (solver_opts.print_stats) {
    PrintStats();
  }


  // Hack for now to return the adaptive solution
  realtype *t_return2 = new realtype[t_adaptive.size()];
  for (size_t i = 0; i < t_adaptive.size(); i++) {
    t_return2[i] = t_adaptive[i];
  }
  realtype *y_return2 = new realtype[t_adaptive.size() * number_of_states];
  for (size_t i = 0; i < y_adaptive.size(); i++) {
    for (size_t j = 0; j < number_of_states; j++) {
      y_return2[i * number_of_states + j] = y_adaptive[i][j];
    }
  }

  py::capsule free_t_when_done2(
    t_return2,
    [](void *f) {
      realtype *vect = reinterpret_cast<realtype *>(f);
      delete[] vect;
    }
  );
  py::capsule free_y_when_done2(
    y_return2,
    [](void *f) {
      realtype *vect = reinterpret_cast<realtype *>(f);
      delete[] vect;
    }
  );

  np_array t_ret2 = np_array(
    t_adaptive.size(),
    &t_return2[0],
    free_t_when_done2
  );
  np_array y_ret2 = np_array(
    t_adaptive.size() * number_of_states,
    &y_return2[0],
    free_y_when_done2
  );




  py::capsule free_t_when_done(
    t_return,
    [](void *f) {
      realtype *vect = reinterpret_cast<realtype *>(f);
      delete[] vect;
    }
  );
  py::capsule free_y_when_done(
    y_return,
    [](void *f) {
      realtype *vect = reinterpret_cast<realtype *>(f);
      delete[] vect;
    }
  );
  py::capsule free_yS_when_done(
    yS_return,
    [](void *f) {
      realtype *vect = reinterpret_cast<realtype *>(f);
      delete[] vect;
    }
  );
  py::capsule free_yterm_when_done(
    yterm_return,
    [](void *f) {
      realtype *vect = reinterpret_cast<realtype *>(f);
      delete[] vect;
    }
  );

  np_array t_ret = np_array(
    teval_i,
    &t_return[0],
    free_t_when_done
  );
  np_array y_ret = np_array(
    teval_i * length_of_return_vector,
    &y_return[0],
    free_y_when_done
  );
  // Note: Ordering of vector is different if computing variables vs returning
  // the complete state vector
  np_array yS_ret;
  if (functions->var_fcns.size() > 0) {
    yS_ret = np_array(
      std::vector<ptrdiff_t> {
        number_of_timesteps,
        length_of_return_vector,
        number_of_parameters
      },
      &yS_return[0],
      free_yS_when_done
    );
  } else {
    yS_ret = np_array(
      std::vector<ptrdiff_t> {
        number_of_parameters,
        number_of_timesteps,
        length_of_return_vector
      },
      &yS_return[0],
      free_yS_when_done
    );
  }
  np_array y_term = np_array(
    length_of_final_sv_slice,
    &yterm_return[0],
    free_yterm_when_done
  );

  Solution sol(retval, t_ret2, y_ret2, yS_ret, y_term);
  // Solution sol(retval, t_adaptive, y_adaptive, yS_ret, y_term);

  return sol;
}

template <class ExprSet>
void IDAKLUSolverOpenMP<ExprSet>::CheckErrors(int const & flag) {
  if (flag < 0) {
    auto message = (std::string("IDA failed with flag ") + std::to_string(flag)).c_str();
    py::set_error(PyExc_ValueError, message);
    throw py::error_already_set();
  }
}

template <class ExprSet>
void IDAKLUSolverOpenMP<ExprSet>::PrintStats() {
  long nsteps, nrevals, nlinsetups, netfails;
  int klast, kcur;
  realtype hinused, hlast, hcur, tcur;

  CheckErrors(IDAGetIntegratorStats(
    ida_mem,
    &nsteps,
    &nrevals,
    &nlinsetups,
    &netfails,
    &klast,
    &kcur,
    &hinused,
    &hlast,
    &hcur,
    &tcur
  ));

  long nniters, nncfails;
  CheckErrors(IDAGetNonlinSolvStats(ida_mem, &nniters, &nncfails));

  long int ngevalsBBDP = 0;
  if (setup_opts.using_iterative_solver) {
    CheckErrors(IDABBDPrecGetNumGfnEvals(ida_mem, &ngevalsBBDP));
  }

  py::print("Solver Stats:");
  py::print("\tNumber of steps =", nsteps);
  py::print("\tNumber of calls to residual function =", nrevals);
  py::print("\tNumber of calls to residual function in preconditioner =",
            ngevalsBBDP);
  py::print("\tNumber of linear solver setup calls =", nlinsetups);
  py::print("\tNumber of error test failures =", netfails);
  py::print("\tMethod order used on last step =", klast);
  py::print("\tMethod order used on next step =", kcur);
  py::print("\tInitial step size =", hinused);
  py::print("\tStep size on last step =", hlast);
  py::print("\tStep size on next step =", hcur);
  py::print("\tCurrent internal time reached =", tcur);
  py::print("\tNumber of nonlinear iterations performed =", nniters);
  py::print("\tNumber of nonlinear convergence failures =", nncfails);
}
