#ifndef QLEARNING_OPENMP_H
#define QLEARNING_OPENMP_H

#include <iostream>
#include <vector>
#include <fstream>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <limits>
#include <omp.h>

using namespace std;

// Q-Learning Hyperparameters
extern double gammaLR;
extern double learningRate;
extern double epsilon, min_epsilon;
extern double decay_rate;

// Function declarations
vector<int> getAvailableActionsOpenMP(int state, const vector<vector<double>>& rMatrix);
int getRandomActionOpenMP(const vector<int>& availableActions);
double updateQOpenMP(int currentState, int action, vector<vector<double>>& rMatrix, vector<vector<double>>& qMatrix, int goal);
pair<vector<int>, double> detectOptimalPathOpenMP(int start, int goal, const vector<vector<double>>& rMatrix, const vector<vector<double>>& qMatrix);
void qLearningOpenMP(int n, vector<vector<double>>& rMatrix, int goal, int max_iterations, ofstream& logStateSpace, ofstream& logOptimalPath);

#endif // QLEARNING_OPENMP_H
