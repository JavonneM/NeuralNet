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
/**
 * Explaination
 * The weight matrix _weights is an vector of matrices
 * Each element in a row is the incoming weight to a particular node ie
 * a row contains the weights incident on the next neuron
 *
 * This allows the properagtion to be performed by computing _w * _a
 * Where _w is the weight matrix and _a is the input from the previous layer
 *
 */
/**
 *
 * @param layers
 * @param dense
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
    std::cout << "Bias" << std::endl;
    for (i = 0; i < _bias.size(); i++) {
        for (j = 0; j < _bias[i].size(); j++) {
            _bias[i][j] = (float) dis(gen);
        }
        std::cout << "Layer " << i << std::endl;
        std::cout << _bias[i] << std::endl << std::endl;
    }
    _layersSize = layers;
}
/**
 * Perform the prediction of the neural network
 * @param input, the initial input
 * @param output, a reference a matrix where the result will be stored
 */
void NeuralNetwork::predict(Eigen::MatrixXf input, Eigen::MatrixXf& output) {
//    std::cout << std::endl << "Start of Prediction" << std::endl;
    int i;
    assert(_weights.size() == _bias.size());
//    std::cout << "Input" << std::endl;
//    std::cout << input << std::endl;
    _a[0] = input;
    for(i = 0; i < _weights.size(); i++) {
//        std::cout << "Weights" << std::endl;
//        std::cout << _weights[i] << std::endl;
//        std::cout << "LayerInput" << std::endl;
//        std::cout << _a[i] << std::endl;
//        std::cout << "Bias" << std::endl << _bias[i] << std::endl;
        _z[i] = (_weights[i] * _a[i]).colwise() + _bias[i];// + _bias[i];
//        std::cout << "LayerOutput" << std::endl << _z[i] << std::endl;
//        std::cout << "Z[" << i << "] = " << _z[i] << std::endl;
        _a[i + 1] = _z[i];
        applyActivationFunction(_a[i + 1]);
//        exit(0);
//        std::cout << "A[" << i + 1 << "] = " << _a[i + 1] << std::endl;
    }
    output = _a[i];
//    std::cout << "Input to first layer: " << std::endl << input << std::endl;
//    std::cout << "Output to finallayer: " << std::endl << output << std::endl;
//    std::cout << "End of Prediction" << std::endl << std::endl;
}

/**
    * Train the neural network using the input data and the expected output data.
    * @param input, the input data <Input Layer> x <Datapoints>.
    * @param output, the expected output data <Output layer> x <datapoints>.
    */
void NeuralNetwork::train(Eigen::MatrixXf& input, Eigen::MatrixXf& expectedOutput, long iterations) {
    std::cout << std::endl << "Start of Training" << std::endl;
    long it;
    for(it = 0l; it < iterations; it++) {
        Eigen::MatrixXf layerOutput;
        long layerIndex;
        predict(input, layerOutput);

        float cost;
        if (it % 1000 == 0) {
            cost = costFunction(expectedOutput, _a[getLayerSize() - 1]).array().sum()/expectedOutput.cols();
            std::cout << "Network Cost: " << cost << std::endl;
        }

        for (layerIndex = ((long) getLayerSize()) - 1l; layerIndex > 0; layerIndex--) {
            layerOutput = _a[layerIndex];
            Eigen::MatrixXf previousdx;
            if (layerIndex == ((long) getLayerSize()) - 1l) {
                previousdx = costFunctionDerivative(expectedOutput, layerOutput); //error //delta(C)/delta(a^{L})
            } else {
                previousdx = _delta[layerIndex];
                previousdx = _weights[layerIndex].transpose() * previousdx; // delta(z^{L})/delta(a^{L - 1}) * delta(C)/delta(z^{L}) = delta(C)/delta(z^{l})
            }
            _delta[layerIndex - 1] = previousdx.array() * sigmoidDerivativeMatrix(layerOutput).array(); //delta(C)/delta(z^{L}) * delta(z^{L})/delta(a^{L - 1}) = delta(C)/delta(a^{L - 1})
            _deltaW[layerIndex - 1] = _a[layerIndex - 1] * _delta[layerIndex - 1].transpose();
        }
        for(int i = 0; i < _weights.size(); i++) {
            _weights[i] = (_weights[i] + _learningRate * _deltaW[i].transpose());
            _bias[i] = (_bias[i] + _learningRate * (_delta[i].rowwise().mean() ));
        }
    }
    std::cout << "End of Training" << std::endl << std::endl;
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
 * @return
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
    return matrix.unaryExpr(&sigmoidDerivative);
}

Eigen::MatrixXf sigmoidDerivativeMatrix(Eigen::MatrixXf& z) {
    return z.unaryExpr(&sigmoidDerivative);
}

float sigmoidDerivative(float z) {
    return z * (1 - z);
}
float sigmoid(float z) {
    return 1.f/(1.f + expf(-z));
}
void NeuralNetwork::applyActivationFunction(Eigen::MatrixXf& input) {
    int i, j;
    for (i = 0; i < input.rows(); i ++ ) {
        for (j = 0; j < input.cols(); j++) {
            input(i, j) = sigmoid(input(i, j));
        }
    }
}