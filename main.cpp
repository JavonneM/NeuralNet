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

void generateSinData(Eigen::MatrixXf& in, int inputSize, Eigen::MatrixXf& out, int outputSize) {
    int datapoints = 100, i;
    Eigen::MatrixXf input(inputSize, datapoints);
    Eigen::MatrixXf output(outputSize, datapoints);
    std::random_device rd;
    std::mt19937 gen(1);//should be rd() instead of 0
    std::uniform_real_distribution<> dis(0, M_PI);
    for (i = 0 ; i < datapoints; i++) {
        input(0, i) = (float) dis(gen);
        output(0, i) = (sinf(input(0, i)) + 1) * 0.5f;
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
}
int main() {
    Eigen::MatrixXf input, expectedOutput;
//    std::vector<int> layers = {2, 1, 4, 2};
//    generateBasicDataXOR(input, layers[0], expectedOutput, layers[layers.size() - 1]);
    std::vector<int> layers = {1, 4, 2, 1};
    generateSinData(input, layers[0], expectedOutput, layers[layers.size() - 1]);

//    generateBasicData(input, layers[0], expectedOutput, layers[layers.size() - 1]);
//    generateBasicData2(input, layers[0], expectedOutput, layers[layers.size() - 1]);
    Eigen::MatrixXf output(expectedOutput.rows(), expectedOutput.cols());
    std::cout << "Input" << std::endl << input << std::endl;
    NeuralNetwork neuralNetwork(layers, 0.4);
    std::cout << "Size input: " << input.rows() << " " << input.cols() << std::endl;
    std::cout << "Size ExpectedOutput: " << expectedOutput.rows() << " " << expectedOutput.cols() << std::endl;
    std::cout << "Size output: " << output.rows() << " " << output.cols() << std::endl;
//    std::cout << neuralNetwork._weights << std::endl;
    neuralNetwork.predict(input, output);
    neuralNetwork.train(input, expectedOutput, 100000);
    neuralNetwork.predict(input, output);
    std::cout << "Input" << std::endl << input << std::endl;
    std::cout << "Output" << std::endl << output << std::endl;
    std::cout << "Expected Output" << std::endl << expectedOutput << std::endl;

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
    return 0;
}