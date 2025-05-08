#ifndef QLEARNING_H
#define QLEARNING_H

#include <iostream>
#include <vector>
#include <fstream>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <limits>

using namespace std;

// Q-Learning Hyperparameters
extern double gammaLR;                           // Discount factor
extern double learningRate;                      // Learning rate
extern double epsilon, min_epsilon;             // Exploration settings
extern double decay_rate;                        // Epsilon decay per episode

// Function declarations
vector<int> getAvailableActions(int state, const vector<vector<double>>& rMatrix);
int getRandomAction(const vector<int>& availableActions);
double updateQ(int currentState, int action, vector<vector<double>>& rMatrix, vector<vector<double>>& qMatrix, int goal);
int epsilonGreedyAction(int state, const vector<vector<double>>& qMatrix, const vector<int>& availableActions);
pair<vector<int>, double> detectOptimalPath(int start, int goal, const vector<vector<double>>& rMatrix, const vector<vector<double>>& qMatrix);
int qLearning(int n, vector<vector<double>>& rMatrix, int goal, int max_iterations, ofstream& logStateSpace, ofstream& logOptimalPath);

#endif // QLEARNING_H
