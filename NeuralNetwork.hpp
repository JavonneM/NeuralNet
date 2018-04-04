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

#ifndef NEURALNETWORK_NEURALNETWORK_HPP
#define NEURALNETWORK_NEURALNETWORK_HPP
#include "debug.hpp"
#include <Eigen/Dense>
class NeuralNetwork {
public:
    /**
     * Initialises the neural network with a vector of the size of each layer.
     * @param layers, the size of each layer stored in a vector.
     * @param dense
     */
    NeuralNetwork(std::vector<int> layers, float learningRate, bool dense=true);
    /**
     * Train the neural network using the input data and the expected output data.
     * @param input, the input data <Input Layer> x <Datapoints>.
     * @param output, the expected output data <Output layer> x <datapoints>.
     * @param iterations, the number of iterations that the network should train for.
     */
    void train(Eigen::MatrixXf& input, Eigen::MatrixXf& expectedOutput, long iterations);
    /**
     * Use the network to predict the output using the input
     * @param input, the input is a matrix of <datapoints> x <Input layer>.
     * @param output, the output of the neural network <Output layer> x <datapoints>.
     */
    void predict(Eigen::MatrixXf input, Eigen::MatrixXf& output);
    /**
     * Returns the number of neurons in the input layer.
     * @return int.
     */
    int getInputSize();
    /**
     * Returns the number of neurons in the output layer.
     * @return int.
     */
    int getOutputSize();
    /**
     * Get number of layers
     * @return
     */
    unsigned long getLayerSize();

private:
    /**
     * Apply the activation function to the input matrix
     * @param input, the matrix that the activation function is applied to elementwise
     */
    void applyActivationFunction(Eigen::MatrixXf& input);
    /**
     * Computes the cost function
     * @param x, the resulted output from the nerual network after the prediction
     * @param y, the expected result
     * @return the cost
     */
    float costFunction(float x, float y);
    /**
     * Applies the cost function to a series of inputs and ouputs
     * @param expectedOuput, the resulted output from the nerual network after the prediction
     * @param output, the expected result
     * @return a Matrix with the costs
     */
    Eigen::MatrixXf costFunction(Eigen::MatrixXf& expectedOuput, Eigen::MatrixXf& output);
    /**
     * Computes the partial derivative of the cost function as a matrix
     * @param expectedOuput, the resulted output from the nerual network after the prediction
     * @param output, the expected result
     * @return a Matrix containing the partial derivatives of the cost function
     */
    Eigen::MatrixXf costFunctionDerivative(Eigen::MatrixXf& expectedOuput, Eigen::MatrixXf& output);

    /**
     * Applies the derivative of the activation funciton
     * @param matrix
     * @return
     */
    Eigen::MatrixXf applyDerivativeActivationFunction(Eigen::MatrixXf& matrix);

    float _learningRate = 1;
    std::vector<Eigen::MatrixXf> _weights;
    std::vector<Eigen::VectorXf> _bias;
    std::vector<Eigen::MatrixXf> _delta;
    std::vector<Eigen::MatrixXf> _deltaW;
    std::vector<Eigen::VectorXf> _deltaB;
    std::vector<Eigen::MatrixXf> _z;
    std::vector<Eigen::MatrixXf> _a;
    std::vector<int> _layersSize;

};
Eigen::MatrixXf sigmoidDerivativeMatrix(Eigen::MatrixXf& z);
float sigmoidDerivative(float z);
float sigmoid(float z);


#endif //NEURALNETWORK_NEURALNETWORK_HPP
