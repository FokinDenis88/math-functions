#include <iostream>

#include "math-functions.hpp"

int main() {
    std::cout << fn_math::GaussianErrorLinearUnitFn<double>(1) << '\n';
    std::cout << fn_math::LogisticFn<double>(1) << '\n';
    std::cout << fn_math::ScaledExponentialLinearUnitFn<double>(1) << '\n';
}