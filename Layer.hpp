/*
author : @rebwar_ai
*/
#ifndef LAYER_HPP
#define LAYER_HPP

#include <vector>
#include <functional>
#include <cmath>
#include <stdexcept>
#include <random>

// Define a type alias for activation functions
using ActivationFunction = std::function<double(double)>;

namespace Activation {
    inline double relu(double x) { return (x > 0.0) ? x : 0.0; }
    inline double reluDerivative(double x) { return (x > 0.0) ? 1.0 : 0.0; }

    inline double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }
    inline double sigmoidDerivative(double x) {
        double s = sigmoid(x);
        return s * (1.0 - s);
    }
}

// Enum for specifying activation types
enum class ActivationType {
    None,
    ReLU,
    Sigmoid
};

// Return a pair of activation function and its derivative based on type
inline std::pair<ActivationFunction, ActivationFunction>
getActivationPair(ActivationType type) {
    using namespace Activation;
    switch (type) {
        case ActivationType::ReLU:
            return {relu, reluDerivative};
        case ActivationType::Sigmoid:
            return {sigmoid, sigmoidDerivative};
        case ActivationType::None:
        default:
            return {ActivationFunction{}, ActivationFunction{}};
    }
}

// Layer structure
struct Layer {
private:
    ActivationFunction activation;
    ActivationFunction activation_derivative;

public:
    int layer_index;
    int size;
    std::vector<double> z;      // Pre-activation values
    std::vector<double> a;      // Activation output values
    std::vector<double> bias;   // Biases (not for input layer)
    std::vector<double> gradient;   // gradient (not for input layer)

    // Constructor
    Layer(int index, int size, ActivationType act_type)
        : layer_index(index),size(size), z(size, 0.0), a(size, 0.0) {
        if(size <= 0 )
        {
            throw std::invalid_argument("Layer sizes must be positive !");
        }
        // Only add bias for non-input layers
        if (index != 0) {
            gradient = std::vector<double>(size, 0.0);
            bias = std::vector<double>(size, 0.0);
            // auto [act, deriv] = getActivationPair(act_type);
            // activation = act;
            // activation_derivative = deriv;
            //--------------------------
            // std::pair<ActivationFunction, ActivationFunction> pair = getActivationPair(act_type);
            // activation = pair.first;
            // activation_derivative = pair.second;
            activation = getActivationPair(act_type).first;
            activation_derivative = getActivationPair(act_type).second;
        }
    }
    
    // std::vector<double>& fillBiasRandom(double min = -0.05, double max = 0.05) {
    //     if (bias.empty()) {
    //         throw std::runtime_error("Cannot fill random values: this layer has no biases.");
    //     }

    //     std::random_device rd;
    //     std::mt19937 gen(rd());
    //     std::uniform_real_distribution<> dist(min, max);

    //     for (double& b : bias) {
    //         b = dist(gen);
    //     }

    //     return bias;
    // }

    // Apply activation function
    double applyActivation(double x) const {
        if (!activation) {
            throw std::runtime_error("This layer has no activation function!");
        }
        return activation(x);
    }

    // Apply derivative of activation function
    double applyActivationDerivative(double x) const {
        if (!activation_derivative) {
            throw std::runtime_error("This layer has no activation derivative!");
        }
        return activation_derivative(x);
    }

    // Optional helper methods
    bool hasActivation() const { return static_cast<bool>(activation); }
    bool hasDerivative() const { return static_cast<bool>(activation_derivative); }
};

#endif // LAYER_HPP
