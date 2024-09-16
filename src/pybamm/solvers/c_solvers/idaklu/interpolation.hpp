#ifndef PYBAMM_CREATE_INTERPOLATION_HPP
#define PYBAMM_CREATE_INTERPOLATION_HPP

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <stdexcept>

namespace py = pybind11;

class CubicSpline {
public:
    // Constructor that takes numpy arrays for t, y, and yp by reference
    CubicSpline(
        const py::array_t<double>& t_input,
        const py::array_t<double>& y_input,
        const py::array_t<double>& yp_input
        )
        : t(t_input), y(y_input), yp(yp_input), c_(compute_c()), d_(compute_d()) {}

    // Optimized interpolation function for a vector of t_interp_np values
    py::array_t<double> Interpolate(const py::array_t<double>& t_interp_np) const {
        if (t_interp_np.ndim() != 1) {
            throw std::invalid_argument("t_interp_np must be a 1D array.");
        }

        const auto t_interp = t_interp_np.unchecked<1>();  // Access t_interp_np as a 1D array
        py::array_t<double> out_array(t_interp_np.size());  // Initialize output array with the correct size
        auto out = out_array.mutable_unchecked<1>();  // Access mutable data of the output array

        // Retrieve sorted t values and use binary search for efficient interval finding
        const auto t_ptr = t.unchecked<1>();
        const auto y_ptr = y.unchecked<1>();  // Access y as a 1D array
        const auto yp_ptr = yp.unchecked<1>();  // Access yp as a 1D array
        const auto n = t.size();

        const bool use_binary_search = t_interp_np.size() < n;

        ssize_t i_itp = 0;
        ssize_t i_interp = extrapolate_left(out_array, t_interp_np);
        double t_data_val = t_ptr(i_itp);
        // py::print("first i_itp", i_interp);
        for (;i_interp < t_interp_np.size(); ++i_interp) {
            double t_val = t_interp(i_interp);

            // In some conditions, a binary search is unnecessary
            if (t_val > t_data_val) {
                if (use_binary_search) {
                    i_itp = binary_search(t_val, i_itp);
                } else {
                    i_itp = silly_search(t_val, i_itp);
                }

                if (i_itp == n - 2) {
                    break;
                }
                t_data_val = t_ptr(i_itp);
            }
            t_data_val = t_ptr(i_itp);

            // py::print("t_val", t_val);
            // py::print("i_itp", i_itp);

            double h = t_val - t_data_val;

            // Use the coefficients stored in c and d
            out(i_interp) = y_ptr(i_itp) + yp_ptr(i_itp) * h + c_[i_itp] * h * h + d_[i_itp] * h * h * h;
        }

        // extrapolate right if needed
        if (i_interp < t_interp_np.size()) {
            extrapolate_right(i_interp, out_array, t_interp_np);
        }

        return out_array;  // Return result as a numpy array
    }

private:
    const py::array_t<double> t;  // Reference to numpy arrays for t
    const py::array_t<double> y;  // Reference to numpy arrays for y
    const py::array_t<double> yp; // Reference to numpy arrays for yp
    const std::vector<double> c_;  // Vector to store c coefficients
    const std::vector<double> d_;  // Vector to store d coefficients

    // Compute cubic spline coefficients and return vector c
    std::vector<double> compute_c() const {
        if (t.ndim() != 1 || y.ndim() != 1 || yp.ndim() != 1) {
            throw std::invalid_argument("t, y, and yp must be 1D arrays.");
        }

        const auto n = t.size();
        std::vector<double> c(n - 1);

        // Use raw pointers for faster access
        const double* t_ptr = t.data();
        const double* y_ptr = y.data();
        const double* yp_ptr = yp.data();

        for (ssize_t i_itp = 0; i_itp < n - 1; ++i_itp) {
            double h = t_ptr[i_itp + 1] - t_ptr[i_itp];
            double inv_h = 1 / h;
            double inv_h_sq = inv_h * inv_h;

            // Calculate c coefficients
            c[i_itp] = (3 * (y_ptr[i_itp + 1] - y_ptr[i_itp]) * inv_h_sq) - (2 * yp_ptr[i_itp] + yp_ptr[i_itp + 1]) * inv_h;
        }
        return c;
    }

    // Compute cubic spline coefficients and return vector d
    std::vector<double> compute_d() const {
        if (t.ndim() != 1 || y.ndim() != 1 || yp.ndim() != 1) {
            throw std::invalid_argument("t, y, and yp must be 1D arrays.");
        }

        const auto n = t.size();
        std::vector<double> d(n - 1);

        // Use raw pointers for faster access
        const double* t_ptr = t.data();
        const double* y_ptr = y.data();
        const double* yp_ptr = yp.data();

        for (ssize_t i_itp = 0; i_itp < n - 1; ++i_itp) {
            double h = t_ptr[i_itp + 1] - t_ptr[i_itp];
            double inv_h = 1 / h;
            double inv_h_sq = inv_h * inv_h;

            // Calculate d coefficients
            d[i_itp] = (2 * (y_ptr[i_itp] - y_ptr[i_itp + 1]) * inv_h_sq * inv_h) + (yp_ptr[i_itp] + yp_ptr[i_itp + 1]) * inv_h_sq;
        }
        return d;
    }

    ssize_t binary_search(
        const double t_val,
        ssize_t low) const {

        const auto t_ptr = t.unchecked<1>();
        ssize_t high = t_ptr.size() - 2;

        while (low <= high) {
            ssize_t mid = low + (high - low) / 2;
            double t_mid = t_ptr(mid);

            if (t_val < t_mid) {
                high = mid - 1;
            } else if (t_val > t_mid) {
                low = mid + 1;
            } else {
                return mid;  // Exact match
            }
        }

        return high;  // Return the largest index where t_val <= t_ptr(index)
    }

    ssize_t silly_search(
        const double t_val,
        ssize_t low) const {

        const auto t_ptr = t.unchecked<1>();
        const auto n = t.size();

        // Ensure we stay within bounds
        while (low < n - 2 && t_val > t_ptr(low + 1)) {
            ++low;
        }
        return low;  // Return the largest index where t_val <= t_ptr(index)
    }

    // Optimized interpolation function for a vector of t_interp_np values
    ssize_t extrapolate_left(
        py::array_t<double>& out_array,
        const py::array_t<double>& t_interp_np) const {

        auto out = out_array.mutable_unchecked<1>();  // Access mutable data of the output array
        const auto t_interp = t_interp_np.unchecked<1>();  // Access t_interp_np as a 1D array
        const auto y_ptr = y.unchecked<1>();  // Access y as a 1D array
        const auto yp_ptr = yp.unchecked<1>();  // Access yp as a 1D array

        // Retrieve sorted t values and use binary search for efficient interval finding
        const auto t_ptr = t.unchecked<1>();
        const auto n = t.size();

        auto t0 = t_ptr(0);

        ssize_t i_interp = 0;  // Initial interval index
        for (i_interp = 0; i_interp < t_interp_np.size(); ++i_interp) {
            double t_val = t_interp(i_interp);
            if (t_val >= t0) {
                break;
            }

            double h = t_val - t0;
            // Use the coefficients stored in c and d
            out(i_interp) = y_ptr(0) + yp_ptr(0) * h + c_[0] * h * h + d_[0] * h * h * h;
        }

        return i_interp;  // Return result as a numpy array
    }
    void extrapolate_right(
        ssize_t i_interp,
        py::array_t<double>& out_array,
        const py::array_t<double>& t_interp_np) const {

        auto out = out_array.mutable_unchecked<1>();  // Access mutable data of the output array
        const auto t_interp = t_interp_np.unchecked<1>();  // Access t_interp_np as a 1D array
        const auto y_ptr = y.unchecked<1>();  // Access y as a 1D array
        const auto yp_ptr = yp.unchecked<1>();  // Access yp as a 1D array

        // Retrieve sorted t values and use binary search for efficient interval finding
        const auto t_ptr = t.unchecked<1>();
        const auto n = t.size();

        auto t0 = t_ptr(n - 2);

        for (; i_interp < t_interp_np.size(); ++i_interp) {
            double t_val = t_interp(i_interp);

            double h = t_val - t0;
            // Use the coefficients stored in c and d
            out(i_interp) = y_ptr(n-2) + yp_ptr(n-2) * h + c_[n-2] * h * h + d_[n-2] * h * h * h;
        }
    }
};

#endif // PYBAMM_CREATE_INTERPOLATION_HPP
