#ifndef QLEARNINGCUDA_H
#define QLEARNINGCUDA_H

#include <iostream>
#include <vector>
#include <fstream>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <limits>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "qlearning.h" // Use functions and variables from qlearning.h

using namespace std;

// Use extern declarations for shared global variables
extern double gammaLR;
extern double learningRate;
extern double epsilon, min_epsilon;
extern double decay_rate;

// Function declarations
void qLearningCUDA(int n, vector<vector<double>>& rMatrix, int goal, int max_iterations,
                   ofstream& logStateSpace, ofstream& logOptimalPath);

// Modified detectOptimalPath for flattened Q-matrix
pair<vector<int>, double> detectOptimalPath(int start, int goal, const vector<double>& rMatrixFlat,
                                            const vector<double>& qMatrixFlat, int n);

#endif // QLEARNINGCUDA_H
