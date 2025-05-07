// ================================
// File: qLearning_openMP.h (Partially Parallelized - Inner Loop Only)
// ================================
#ifndef QLEARNING_H
#define QLEARNING_H

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
double gammaLR = 0.95;
double learningRate = 0.1;
double epsilon = 1.0, min_epsilon = 0.01;
double decay_rate = 0.999;

vector<int> getAvailableActions(int state, const vector<vector<double>>& rMatrix) {
    vector<int> actions;
    for (int j = 0; j < rMatrix[state].size(); j++) {
        if (rMatrix[state][j] > 0 && rMatrix[state][j] != std::numeric_limits<double>::infinity()) {
            actions.push_back(j);
        }
    }
    return actions;
}

int getRandomAction(const vector<int>& availableActions) {
    unsigned int seed = omp_get_thread_num() + time(NULL);
return availableActions[rand_r(&seed) % availableActions.size()];

}

double updateQ(int currentState, int action, vector<vector<double>>& rMatrix, vector<vector<double>>& qMatrix, int goal) {
    double cost = rMatrix[currentState][action];
    if (cost <= 0 || cost == std::numeric_limits<double>::infinity())
        return qMatrix[currentState][action];

    double reward = -cost;
    double maxQ = -std::numeric_limits<double>::infinity();
    auto nextActions = getAvailableActions(action, rMatrix);
    if (!nextActions.empty()) {
        #pragma omp parallel for reduction(max:maxQ)
        for (int i = 0; i < nextActions.size(); ++i) {
            int next = nextActions[i];
            maxQ = max(maxQ, qMatrix[action][next]);
        }
    } else {
        maxQ = 0.0;
    }

    double target = reward + gammaLR * maxQ;
    qMatrix[currentState][action] += learningRate * (target - qMatrix[currentState][action]);
    return qMatrix[currentState][action];
}

pair<vector<int>, double> detectOptimalPath(int start, int goal, const vector<vector<double>>& rMatrix, const vector<vector<double>>& qMatrix) {
    vector<int> path;
    double totalCost = 0.0;
    int currentState = start;
    path.push_back(currentState);
    vector<bool> visited(rMatrix.size(), false);
    visited[currentState] = true;
    int step_count = 0;
    const int max_steps = rMatrix.size() * 2;

    while (currentState != goal && step_count++ < max_steps) {
        vector<int> actions = getAvailableActions(currentState, rMatrix);
        if (actions.empty()) return {{}, std::numeric_limits<double>::infinity()};

        int nextState = -1;
        double maxQ = -std::numeric_limits<double>::infinity();
        for (int action : actions) {
            if (!visited[action] && qMatrix[currentState][action] > maxQ) {
                nextState = action;
                maxQ = qMatrix[currentState][action];
            }
        }
        if (nextState == -1) return {{}, std::numeric_limits<double>::infinity()};

        totalCost += rMatrix[currentState][nextState];
        currentState = nextState;
        path.push_back(currentState);
        visited[currentState] = true;
    }

    return (currentState == goal) ? make_pair(path, totalCost) : make_pair(vector<int>(), std::numeric_limits<double>::infinity());
}

void qLearningOpenMP(int n, vector<vector<double>>& rMatrix, int goal, int max_iterations, ofstream& logStateSpace, ofstream& logOptimalPath) {
    vector<vector<double>> qMatrix(n, vector<double>(n, 0.0));
    double previousCost = std::numeric_limits<double>::infinity();
    int stableCostCount = 0;

    for (int ep = 0; ep < max_iterations; ep++) {
        int state = 0;
        auto actions = getAvailableActions(state, rMatrix);
        if (actions.empty()) continue;

        while (state != goal) {
            int action;
            if ((double)rand() / RAND_MAX < epsilon) {
                action = getRandomAction(actions);
            } else {
                action = actions[0];
                double best_value = qMatrix[state][action];
                for (int a : actions) {
                    if (qMatrix[state][a] > best_value) {
                        action = a;
                        best_value = qMatrix[state][a];
                    }
                }
            }

            updateQ(state, action, rMatrix, qMatrix, goal);
            state = action;
            actions = getAvailableActions(state, rMatrix);
            if (actions.empty()) break;
        }

        if (epsilon > min_epsilon)
            epsilon *= decay_rate;

        auto [optimalPath, totalCost] = detectOptimalPath(0, goal, rMatrix, qMatrix);

        if (ep % 10 == 0) {
            logOptimalPath << "[Episode " << ep << "] ";
            if (!optimalPath.empty()) {
                for (int node : optimalPath) logOptimalPath << node << " ";
                logOptimalPath << " | Cost: " << totalCost << "\n";
            } else {
                logOptimalPath << "No path found\n";
            }
        }

        if (totalCost == previousCost) {
            stableCostCount++;
        } else {
            stableCostCount = 0;
        }
        previousCost = totalCost;

        if (stableCostCount >= 10) {
            logOptimalPath << "[Converged at Episode " << ep << "] ";
            if (!optimalPath.empty()) {
                for (int node : optimalPath) logOptimalPath << node << " ";
                logOptimalPath << " | Cost: " << totalCost << "\n";
            } else {
                logOptimalPath << "No path found\n";
            }
            break;
        }
    }

    auto [optimalPath, totalCost] = detectOptimalPath(0, goal, rMatrix, qMatrix);
    logOptimalPath << "[Final] ";
    if (!optimalPath.empty()) {
        for (int node : optimalPath) {
            logOptimalPath << node << " ";
        }
        logOptimalPath << " | Cost: " << totalCost << "\n";
    } else {
        logOptimalPath << "No path found\n";
    }
}

#endif // QLEARNING_H
