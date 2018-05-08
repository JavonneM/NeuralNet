/**
 * MIT License

 * Copyright (c) 2018 Javonne Jason Martin

 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:

 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.

 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "NeuralNetwork.hpp"
#include <random>
#include <iostream>
#include <set>
/**
 * Constructor for the Neural network
 * @param layers, the size of each layer including input and output layers
 * @param dense, default true, the network will be fully connected TODO implementation for nonfully connected
 */
NeuralNetwork::NeuralNetwork(std::vector<int> layers, float learningRate, bool dense) {
    int i, j, k;
    assert(layers.size() > 1);
    _learningRate = learningRate;
    for(i = 1; i < layers.size(); i++) {
        _weights.emplace_back(Eigen::MatrixXf(layers[i], layers[i - 1]));
        _deltaW.emplace_back(Eigen::MatrixXf(layers[i], layers[i - 1]));
        _bias.emplace_back(Eigen::VectorXf(layers[i]));
    }
    _delta = std::vector<Eigen::MatrixXf>(_weights.size());
    _a = std::vector<Eigen::MatrixXf>(layers.size());
    _z = std::vector<Eigen::MatrixXf>(layers.size());
    std::random_device rd;
    std::mt19937 gen(0);//should be rd() instead of 0
    std::uniform_real_distribution<> dis(0, 1.0);
    for (i = 0; i < _weights.size(); i++) {
        for (j = 0; j < _weights[i].rows(); j++) {
            for (k = 0; k < _weights[i].cols(); k++) {
                _weights[i](j, k) = (float) dis(gen);
            }
        }
//        debug_print("Layer %d Rows: %ld Cols: %ld\n", i, _weights[i].rows(), _weights[i].cols());
//        debug_print(_weights[i]);
//        std::cout << _weights[i] << std::endl;
//        exit(0);
    }
//    std::cout << "Bias" << std::endl;
    for (i = 0; i < _bias.size(); i++) {
        for (j = 0; j < _bias[i].size(); j++) {
            _bias[i][j] = (float) dis(gen);
        }
//        std::cout << "Layer " << i << std::endl;
//        std::cout << _bias[i] << std::endl << std::endl;
    }
    _layersSize = layers;
}
/**
 * Perform the prediction of the neural network
 * @param input, the initial input
 * @param output, a reference a matrix where the result will be stored
 */
void NeuralNetwork::predict(Eigen::MatrixXf& input, Eigen::MatrixXf& output) {
    int i;
    assert(_weights.size() == _bias.size());
    _a[0] = input;
    for(i = 0; i < _weights.size(); i++) {
        _z[i] = (_weights[i] * _a[i]).colwise() + _bias[i];
        _a[i + 1] = _z[i];
        applyActivationFunction(_a[i + 1]);
    }
    output = _a[i];
}
/**
 * Set the activation function to use on each ouput neuron
 * @param function, the function type
 */
void NeuralNetwork::setActivationFunction(ACTIVATION function) {
    _activationFunction = function;
}
/**
 * Get the Activation function that will be used
 * @return ACTIVIATION, an enum
 */
NeuralNetwork::ACTIVATION NeuralNetwork::getActivationFunction() {
    return _activationFunction;
}

/**
    * Train the neural network using the input data and the expected output data.
    * @param input, the input data <Input Layer> x <Datapoints>.
    * @param output, the expected output data <Output layer> x <datapoints>.
    */
void NeuralNetwork::train(Eigen::MatrixXf& input, Eigen::MatrixXf& expectedOutput, bool printIteration) {

    float cost;
    long layerIndex;
    Eigen::MatrixXf output;
    predict(input, output);
    if (printIteration) {
        cost = costFunction(expectedOutput, _a[getLayerSize() - 1]).array().sum()/expectedOutput.cols();
        std::cout << "Network Cost: " << cost << std::endl;
    }
    Eigen::MatrixXf layerOutput;

    for (layerIndex = ((long) getLayerSize()) - 1l; layerIndex > 0; layerIndex--) {
        layerOutput = _a[layerIndex];
        Eigen::MatrixXf previousdx;
        if (layerIndex == ((long) getLayerSize()) - 1l) {
            previousdx = costFunctionDerivative(expectedOutput, layerOutput); //error //delta(C)/delta(a^{L})
        } else {
            previousdx = _delta[layerIndex];
            previousdx = _weights[layerIndex].transpose() * previousdx; // delta(z^{L})/delta(a^{L - 1}) * delta(C)/delta(z^{L}) = delta(C)/delta(z^{l})
        }
        _delta[layerIndex - 1] = previousdx.array() * applyDerivativeActivationFunction(layerOutput).array(); //delta(C)/delta(z^{L}) * delta(z^{L})/delta(a^{L - 1}) = delta(C)/delta(a^{L - 1})
        _deltaW[layerIndex - 1] = _a[layerIndex - 1] * _delta[layerIndex - 1].transpose();
    }
    for(int i = 0; i < _weights.size(); i++) {
        _weights[i] = (_weights[i] + _learningRate * _deltaW[i].transpose());
        _bias[i] = (_bias[i] + _learningRate * (_delta[i].rowwise().mean() ));
    }

}

/**
 * Returns the number of neurons in the input layer.
 * @return int.
 */
int NeuralNetwork::getInputSize() {
    return _layersSize[0];
}
/**
 * Returns the number of neurons in the output layer.
 * @return int.
 */
int NeuralNetwork::getOutputSize() {
    return _layersSize[_layersSize.size() - 1];
}
/**
 * Get number of layers
 * @return unsigned long
 */
unsigned long NeuralNetwork::getLayerSize() {
    return _layersSize.size();
}
float NeuralNetwork::costFunction(float C, float y) {
    return (C - y) * (C - y);
}
/**
 * Cost function used
 * TODO implement system for different cost functions
 * @param expectedOuput
 * @param output
 * @return
 */
Eigen::MatrixXf NeuralNetwork::costFunction(Eigen::MatrixXf& expectedOuput, Eigen::MatrixXf& output) {
    Eigen::MatrixXf difference = expectedOuput - output;
    return difference.array().square().matrix();
}
/**
 * Cost function derivative
 * TODO implement system for different cost functions
 * @param expectedOuput
 * @param output
 * @return
 */
Eigen::MatrixXf NeuralNetwork::costFunctionDerivative(Eigen::MatrixXf& expectedOuput, Eigen::MatrixXf& output) {
    Eigen::MatrixXf difference = 2 * (expectedOuput - output);
    return difference;
}
/**
 * Derivative of the Cost function
 * TODO implement syste for different cost functions
 * @param matrix
 */
Eigen::MatrixXf NeuralNetwork::applyDerivativeActivationFunction(Eigen::MatrixXf& matrix) {

    float (*activationFunction)(float);
    switch (_activationFunction) {
        case SIGMOID:
            activationFunction = sigmoidDerivative;
            break;
        case RELU:
            activationFunction = reluDerivative;
            break;
        case SOFTMAX:
            activationFunction = softmaxDerivative;
            break;
        default:
            activationFunction = sigmoidDerivative;
            std::cerr << "Requested activation function doesn't exist reverting to sigmoid" << std::endl;
    }
    return matrix.unaryExpr(activationFunction);
}



/**
 * Applies the activation function to a input matrix elementwise
 * @param input, the _z matrix obtained from applying the weights and bias
 */
void NeuralNetwork::applyActivationFunction(Eigen::MatrixXf& input) {
//    int (*minus)(int,int) = subtraction;
    float (*activationFunction)(float);
    switch (_activationFunction) {
        case SIGMOID:
            activationFunction = sigmoid;
            break;
        case RELU:
            activationFunction = relu;
            break;
        case SOFTMAX:
            activationFunction = softmax;
            break;
        default:
            activationFunction = sigmoid;
            std::cerr << "Requested activation function doesn't exist reverting to sigmoid" << std::endl;
    }
    input = input.unaryExpr(activationFunction);
}
/**
 * These are the activation functions and their derivatives
 * These functions need to prototyped and added to applyActivationFunction
 * and applyDerivativeActivationFunction
 */

/**
 * Sigmoid function
 * @param z
 * @return
 */
static float sigmoid(float z) {
    return 1.f/(1.f + expf(-z));
}
static float sigmoidDerivative(float z) {
    return z * (1 - z);
}
static float relu(float z) {
    return 1.f/(1.f + expf(-z));
}
static float reluDerivative(float z) {
    return 1.f/(1.f + expf(-z));
}
static float softmax(float z) {
    return 1.f/(1.f + expf(-z));
}
static float softmaxDerivative(float z) {
    return 1.f/(1.f + expf(-z));
}

