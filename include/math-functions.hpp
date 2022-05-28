#ifndef MATH_FUNCTIONS_HPP
#define MATH_FUNCTIONS_HPP

#include <cmath>
// For MaxoutFn
#include <vector>
#include <algorithm>

// Mathematical functions
namespace fn_math {
    // System of equations
    // 0, if x < 0
    // 1, if x >= 0
    template<typename T = double>
    inline unsigned short BinaryStepFn(T x) {
        return x < 0 ? 0 : 1;
    };

    // Exponential Linear Unit (ELU)
    // a(e^x - 1)   if x <= 0
    // x            if x > 0
    template <typename T = double>
    inline T ExponentialLinearUnitFn(T a, T x) {
        return x > 0 ? x : a * (std::exp(x) - 1);
    };

    // f(x) = exp^(-x^2)
    template <typename T = double>
    inline T GaussianFn(T x) {
        return std::exp(-std::pow(x, 2));
    };

    // Gaussian Error Linear Unit (GELU)
    // f(x) = 1/2*x(1+erf(x/sqrt(2)))
    template <typename T = double>
    inline T GaussianErrorLinearUnitFn(T x) {
        return 1.0 / 2.0 * x * (1 + std::erf(x / std::sqrt(2)));
    };

    // f(x) = exp(-(mod(x - c)^2) / (2sigma^2))
    template <typename T = double>
    inline T GaussianRBFFn(T x, T c, T sigma) {
        return std::exp(-std::pow(x - c, 2) / (2 * std::pow(sigma, 2)));
    };

    // f(x) = 1     if a*x + b > 0
    template <typename T = double>
    inline T HeavisideFn(T a, T x, T b) {
        return (a * x + b > 0) ? 1 : 0;
    };

    // Hyperbolic tangent (tanh)
    // f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    template <typename T = double>
    inline T HyperbolicTangentFn(T x) {
        return (std::exp(x) - std::exp(-x)) / (std::exp(x) + std::exp(-x));
    };

    // f(x) = x
    template <typename T = double>
    inline T IdentityFn(T x) {
        return x;
    };

    // Leaky rectified linear unit (Leaky_ReLU)
    // 0.01*x   if x < 0
    // x        if x >= 0
    template <typename T = double>
    inline T LeakyRectifiedLinearUnitFn(T x) {
        return x < 0 ? 0.01 * x : x;
    };

    // f(x) = a*x + b
    template <typename T = double>
    inline T LinearFn(T a, T x, T b) {
        return a * x + b;
    };

    // f(x) = 1/(1+exp^(-x))
    template <typename T = double>
    inline T LogisticFn(T x) {
        return 1 / (1 + std::exp(-x));
    };

    // f(x) = max xi        x = Vector
    template <typename T = double>
    inline T MaxoutFn(std::vector<T> x) {
        return std::max_element(x.begin, x.end);
    };

    // f(x) = x*tanh(ln(1+exp^x))
    template <typename T = double>
    inline T MishFn(T x) {
        return x * std::tanh(std::log(1 + std::exp(x)));
    };

    // f(x) = sqrt(mod(x - c)^2 + a^2)
    template <typename T = double>
    inline T MultiquadraticsFn(T x, T c, T a) {
        return std::sqrt(std::pow(x - c, 2) + std::pow(a, 2));
    };

    // Parameteric rectified linear unit (PReLU)
    // a*x      if x < 0
    // x        if x >= 0
    template <typename T = double>
    inline T ParametricRectifiedLinearUnitFn(T a, T x) {
        return x < 0 ? a * x : x;
    };

    // Rectified linear unit (ReLU)
    // 0        if x <= 0
    // x        if x > 0
    template <typename T = double>
    inline T RectifiedLinearUnitFn(T x) {
        return x > 0 ? x : 0;
    };

    // Scaled exponential linear unit (SELU)
    // lambda* a(exp(x) - 1)    if x < 0
    // lambda* x                if x >= 0
    template <typename T = double>
    inline T ScaledExponentialLinearUnitFn(T x) {
        // lambda = 1.0507
        // alpha = 1.67326
        return (x < 0) ? 1.0507 * 1.67326 * (std::exp(x) - 1) : 1.0507 * x;
    };

    // Sigmoid linear unit Sigmoid shrinkage, SiL, Swish-‍1 (SiLU)
    // x / (1 + exp(-x))
    template <typename T = double>
    inline T SigmoidLinearUnitFn(T x) {
        return x / (1 + std::exp(-x));
    };

    // f(x) = a*x + b
    //template <typename T = double>
    //inline T SoftmaxFn(T a, T x, T b) {
    //    return a * x + b;
    //};

    // ln(1 + exp(x))
    template <typename T = double>
    inline T SoftplusFn(T x) {
        return std::log(1 + std::exp(x));
    };
}

#endif // !MATH_FUNCTIONS_HPP