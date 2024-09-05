#ifndef PYBAMM_CREATE_OBSERVE_HPP
#define PYBAMM_CREATE_OBSERVE_HPP

#include "IDAKLUSolverOpenMP_solvers.hpp"
#include <idas/idas.h>
#include <memory>

/**
 * @brief Loops over the solution and generates the observable output
 */
template<class ExprSet>
void process_time_series(
    const std::vector<np_array>& ts_np,
    const std::vector<np_array>& ys_np,
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

/**
 * @brief Observe 0D variables
 */
template<class ExprSet>
const py::array_t<double> observe_0D(
    const std::vector<np_array>& ts_np,
    const std::vector<np_array>& ys_np,
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

/**
 * @brief Observe 1D variables
 */
template<class ExprSet>
const py::array_t<double> observe_1D(
    const std::vector<np_array>& ts_np,
    const std::vector<np_array>& ys_np,
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

/**
 * @brief Observe 2D variables
 */
template<class ExprSet>
const py::array_t<double> observe_2D(
    const std::vector<np_array>& ts_np,
    const std::vector<np_array>& ys_np,
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

#endif // PYBAMM_CREATE_OBSERVE_HPP
