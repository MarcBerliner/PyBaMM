#ifndef PYBAMM_CREATE_HERMITE_INTERPOLATION_HPP
#define PYBAMM_CREATE_HERMITE_INTERPOLATION_HPP

#include "IDAKLUSolverOpenMP_solvers.hpp"
#include <idas/idas.h>
#include <memory>
#include <vector>
#include "common.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>  // For numpy support in pybind11

namespace py = pybind11;

class HermiteInterpolation {
private:
    const py::detail::unchecked_reference<double, 1>& t_;
    const py::detail::unchecked_reference<double, 2>& y_;
    const py::detail::unchecked_reference<double, 2>& yp_;
    std::vector<double> c_;
    std::vector<double> d_;
    size_t current_j_;

    void compute_c_d(size_t j, const std::vector<double>& y_start, const std::vector<double>& y_end,
                     const std::vector<double>& yp_start, const std::vector<double>& yp_end) {
        const double h_full = t_(j + 1) - t_(j);
        const double inv_h = 1.0 / h_full;
        const double inv_h2 = inv_h * inv_h;
        const double inv_h3 = inv_h2 * inv_h;

        for (size_t i = 0; i < y_start.size(); ++i) {
            c_[i] = 3.0 * (y_end[i] - y_start[i]) * inv_h2 - (2.0 * yp_start[i] + yp_end[i]) * inv_h;
            d_[i] = 2.0 * (y_start[i] - y_end[i]) * inv_h3 + (yp_start[i] + yp_end[i]) * inv_h2;
        }
        current_j_ = j;
    }

public:
    HermiteInterpolation(
        const py::detail::unchecked_reference<double, 1>& t,
        const py::detail::unchecked_reference<double, 2>& y,
        const py::detail::unchecked_reference<double, 2>& yp
    ) : t_(t), y_(y), yp_(yp), c_(y.shape(0)), d_(y.shape(0)), current_j_(0) {}

    void interpolate(std::vector<double>& out, const double t_interp, const size_t j,
                     const std::vector<double>& y_start, const std::vector<double>& yp_start) {
        const double h = t_interp - t_(j);
        const double h2 = h * h;
        const double h3 = h2 * h;

        for (size_t i = 0; i < out.size(); ++i) {
            out[i] = y_start[i] + yp_start[i] * h + c_[i] * h2 + d_[i] * h3;
        }
    }

    void hermite_then_func(
        const py::detail::unchecked_reference<double, 1>& t_interp,
        const vector<np_array_dense>& inputs_np,
        const vector<const typename ExprSet::BaseFunctionType*>& funcs,
        double* out,
        const int len,
        const ssize_t N_interp,
        const ssize_t N_interval
    ) {
        std::vector<double> y_interp(y_.shape(0));
        ssize_t count = 0;
        ssize_t i_interp = 0;
        double t_interp_next = t_interp(0);
        size_t j = 0;

        for (ssize_t k = 0; k < N_interval; ++k) {
            while (j < t_.shape(0) - 1 && t_(j + 1) <= t_interp_next) {
                ++j;
            }

            interpolate(y_interp, t_interp_next, j, std::vector<double>(y_.data(j), y_.data(j) + y_.shape(0)),
                        std::vector<double>(yp_.data(j), yp_.data(j) + yp_.shape(0)));
            observe(t_(j), y_interp.data(), inputs_np[j].data(), funcs[j], &out[count], len);

            count += len;
            ++i_interp;
            if (i_interp < N_interp) {
                t_interp_next = t_interp(i_interp);
            }
        }
    }

    void func_then_hermite(
        const py::detail::unchecked_reference<double, 1>& t_interp,
        const vector<np_array_dense>& inputs_np,
        const vector<const typename ExprSet::BaseFunctionType*>& funcs,
        double* out,
        const int len,
        const ssize_t N_interp,
        const ssize_t N_interval
    ) {
        std::vector<double> y_start(y_.shape(0)), y_end(y_.shape(0));
        std::vector<double> yp_start(y_.shape(0)), yp_end(y_.shape(0));
        std::vector<double> y_temp(y_.shape(0));
        ssize_t count = 0;
        size_t j = 0;
        const double epsilon = 1e-8;

        for (ssize_t k = 0; k < N_interval; ++k) {
            size_t j_end = j + 1;
            while (j_end < t_.shape(0) && t_(j_end) <= t_interp(k * N_interp / N_interval)) {
                ++j_end;
            }
            --j_end;

            // Compute function at start and end points
            observe(t_(j), y_.data(j), inputs_np[j].data(), funcs[j], y_start.data(), len);
            observe(t_(j_end), y_.data(j_end), inputs_np[j_end].data(), funcs[j_end], y_end.data(), len);

            // Compute derivatives using finite difference with epsilon
            observe(t_(j) + epsilon, y_.data(j), inputs_np[j].data(), funcs[j], y_temp.data(), len);
            for (int i = 0; i < len; ++i) {
                yp_start[i] = (y_temp[i] - y_start[i]) / epsilon;
            }

            observe(t_(j_end) + epsilon, y_.data(j_end), inputs_np[j_end].data(), funcs[j_end], y_temp.data(), len);
            for (int i = 0; i < len; ++i) {
                yp_end[i] = (y_temp[i] - y_end[i]) / epsilon;
            }

            // Compute Hermite coefficients
            compute_c_d(j, y_start, y_end, yp_start, yp_end);

            // Interpolate for all points in this interval
            for (ssize_t i = k * N_interp / N_interval; i < (k + 1) * N_interp / N_interval && i < N_interp; ++i) {
                interpolate(std::vector<double>(&out[count], &out[count + len]), t_interp(i), j, y_start, yp_start);
                count += len;
            }

            j = j_end;
        }
    }
};

void observe(
    const double& t,
    const double* y_interp,
    const double* inputs,
    const typename ExprSet::BaseFunctionType* func,
    double* out,
    const int len
) {
    std::vector<const double*> args = {&t, y_interp, inputs};
    std::vector<double*> results = {out};
    func(args, results);
}

#endif // PYBAMM_CREATE_HERMITE_INTERPOLATION_HPP
