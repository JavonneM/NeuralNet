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
#include <iostream>
#include <fstream>
#include "debug.hpp"

#include "NeuralNetwork.hpp"
/**
 * Generate testing data to test the neural network
 */
void generateSinData(Eigen::MatrixXf& in, int inputSize, Eigen::MatrixXf& out, int outputSize, int datapoints) {
    int i;
    Eigen::MatrixXf input(inputSize, datapoints);
    Eigen::MatrixXf output(outputSize, datapoints);
    std::random_device rd;
    std::mt19937 gen(rd());//should be rd() instead of 0
    std::uniform_real_distribution<> dis(0, M_PI);
    for (i = 0 ; i < datapoints; i++) {
        input(0, i) = (float) dis(gen);
        output(0, i) = (sinf(input(0, i)));
    }
    in = input;
    out = output;
}
void generateBasicData(Eigen::MatrixXf& in, int inputSize, Eigen::MatrixXf& out, int outputSize) {
    int datapoints = 3, i;
    Eigen::MatrixXf input(inputSize, datapoints);
    Eigen::MatrixXf output(outputSize, datapoints);
    std::random_device rd;
    std::mt19937 gen(1);//should be rd() instead of 0
    std::uniform_real_distribution<> dis(0, 1);
    for (i = 0 ; i < datapoints; i++) {
        input(0, i) = dis(gen);
        if(input(0, i) > 0.5) {
            output(0, i) = 0;
        } else {
            output(0, i) = 1;
        }

    }
    in = input;
    out = output;
}
//for 2 x 4 x 1
void generateBasicData2(Eigen::MatrixXf& in, int inputSize, Eigen::MatrixXf& out, int outputSize) {
    int datapoints = 10, i, j;
    Eigen::MatrixXf input(inputSize, datapoints);
    Eigen::MatrixXf output(datapoints, outputSize);
    std::random_device rd;
    std::mt19937 gen(1);//should be rd() instead of 0
    std::uniform_real_distribution<> dis(0, 1);

    for (i = 0; i < datapoints; i++) {
        for (j = 0 ; j < inputSize; j++) {
            input(j, i) = static_cast<float>(dis(gen));
        }
        for (j = 0; j < outputSize; j++) {
            output(i, j) = 0;
        }

    }
    in = input;
    out = output;
}
void generateBasicDataXOR(Eigen::MatrixXf& in, int inputSize, Eigen::MatrixXf& out, int outputSize) {
    int datapoints = 3, i;
    Eigen::MatrixXf input(inputSize, datapoints);
    Eigen::MatrixXf output(outputSize, datapoints);

    input << 0, 0, 1,
             1, 0, 1;
    output << 1, 0, 0,
              0, 1, 1;
    in = input;
    out = output;
    std::cout << "Generating test data for XOR" << std::endl;
}
/**
 * Testing the neural network using the constructed Neural network class
 * @return
 */
int main() {
    Eigen::MatrixXf input, expectedOutput;
    std::vector<int> layers = {2, 1, 4, 2};
    generateBasicDataXOR(input, layers[0], expectedOutput, layers[layers.size() - 1]);
//    std::vector<int> layers = {1, 4, 2, 1};
//    generateBasicData(input, layers[0], expectedOutput, layers[layers.size() - 1]);
//    generateBasicData2(input, layers[0], expectedOutput, layers[layers.size() - 1]);
    Eigen::MatrixXf output(expectedOutput.rows(), expectedOutput.cols());
    NeuralNetwork neuralNetwork(layers, 0.5);
    neuralNetwork.setActivationFunction(NeuralNetwork::ACTIVATION::RELU);
//    generateSinData(input, layers[0], expectedOutput, layers[layers.size() - 1], 200);
    neuralNetwork.predict(input, output);
    debug_print("Before Training");
    debug_print("Output");
    debug_print(output);
    debug_print("Expected");
    debug_print(expectedOutput);
    long iteration;
    debug_print("Training");
    for(iteration = 0l; iteration < 10000; iteration++) {
        neuralNetwork.train(input, expectedOutput, iteration%1000 == 0);
    }
    debug_print("Training Complete");
//    generateSinData(input, layers[0], expectedOutput, layers[layers.size() - 1], 10000);
    debug_print("Prediction on trained data");
    neuralNetwork.predict(input, output);


    //Write data to file for graphing using graphTool.py
    std::ofstream file;
    file.open("data.txt");
    file << "input" << std::endl;
    file << input << std::endl;
    file << "output" << std::endl;
    file << output << std::endl;
    file << "expected" << std::endl;
    file << expectedOutput << std::endl;
    file.flush();
    file.close();
    debug_print("Input");
    debug_print(input);
    debug_print("Output");
    debug_print(output);
    debug_print("ExpectedOutput");
    debug_print(expectedOutput);
    return 0;
}