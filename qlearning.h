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
double gammaLR = 0.95;                           // Discount factor (increased for longer-term rewards)
double learningRate = 0.1;                      // Learning rate
double epsilon = 1.0, min_epsilon = 0.01;       // Exploration settings (reduced min_epsilon)
double decay_rate = 0.999;                      // Epsilon decay per episode (slower decay)

// Get valid actions from a given state
vector<int> getAvailableActions(int state, const vector<vector<double>>& rMatrix) {
    vector<int> actions;
    for (int j = 0; j < rMatrix[state].size(); j++) {
        if (rMatrix[state][j] > 0 && rMatrix[state][j] != std::numeric_limits<double>::infinity()) {
            actions.push_back(j);
        }
    }
    if (actions.empty()) {
        cerr << "Warning: No available actions for state " << state << "." << endl;
    }
    return actions;
}

// Random action selection from valid actions
int getRandomAction(const vector<int>& availableActions) {
    return availableActions[rand() % availableActions.size()];
}

// Q-value update function using cost minimization (i.e., reward = -cost)
double updateQ(int currentState, int action, vector<vector<double>>& rMatrix, vector<vector<double>>& qMatrix, int goal) {
    double cost = rMatrix[currentState][action];
    if (cost <= 0 || cost == std::numeric_limits<double>::infinity())
        return qMatrix[currentState][action];

    double reward = -cost; // Negative cost to represent reward

    double maxQ = -std::numeric_limits<double>::infinity();
    auto nextActions = getAvailableActions(action, rMatrix);
    if (!nextActions.empty()) {
        for (int next : nextActions) {
            maxQ = std::max(maxQ, qMatrix[action][next]);
        }
    } else {
        maxQ = 0.0; // Dead-end
    }

    double target = reward + gammaLR * maxQ;
    qMatrix[currentState][action] += learningRate * (target - qMatrix[currentState][action]); // Use learning rate

    return qMatrix[currentState][action];
}

// Epsilon-greedy action selection
int epsilonGreedyAction(int state, const vector<vector<double>>& qMatrix, const vector<int>& availableActions) {
    if ((double)rand() / RAND_MAX < epsilon) {
        return getRandomAction(availableActions);
    } else {
        int best_action = availableActions[0];
        double best_value = qMatrix[state][best_action];
        for (int a : availableActions) {
            if (qMatrix[state][a] > best_value) {
                best_action = a;
                best_value = qMatrix[state][a];
            }
        }
        return best_action;
    }
}

// Extract and print the optimal path using max-Q traversal (Pure Q-Learning Path Extraction)
pair<vector<int>, double> detectOptimalPath(int start, int goal, const vector<vector<double>>& rMatrix, const vector<vector<double>>& qMatrix) {
    vector<int> path;
    double totalCost = 0.0;
    int currentState = start;
    path.push_back(currentState);
    vector<bool> visited(rMatrix.size(), false);
    visited[currentState] = true;
    int step_count = 0;
    const int max_steps = rMatrix.size() * 2;  // Prevent infinite loops

    while (currentState != goal && step_count++ < max_steps) {
        vector<int> actions = getAvailableActions(currentState, rMatrix);
        if (actions.empty()) {
            return {{}, std::numeric_limits<double>::infinity()}; // No path found
        }

        int nextState = -1;
        double maxQ = -std::numeric_limits<double>::infinity();

        for (int action : actions) {
            if (!visited[action] && qMatrix[currentState][action] > maxQ && rMatrix[currentState][action] != std::numeric_limits<double>::infinity()) {
                nextState = action;
                maxQ = qMatrix[currentState][action];
            }
        }

        if (nextState == -1) {
            // No valid next state found, try unvisited ones even with lower Q values
            for (int action : actions) {
                if (qMatrix[currentState][action] > maxQ && rMatrix[currentState][action] != std::numeric_limits<double>::infinity()) {
                    nextState = action;
                    maxQ = qMatrix[currentState][action];
                }
            }
            if (nextState == -1)
                return {{}, std::numeric_limits<double>::infinity()}; // Total failure
        }

        totalCost += rMatrix[currentState][nextState];
        currentState = nextState;
        path.push_back(currentState);
        visited[currentState] = true;
    }

    if (currentState == goal) {
        return {path, totalCost};
    } else {
        return {{}, std::numeric_limits<double>::infinity()}; // Path not found
    }
}

// Main Q-learning driver
void qLearning(int n, vector<vector<double>>& rMatrix, int goal, int max_iterations, ofstream& logStateSpace, ofstream& logOptimalPath) {
    vector<vector<double>> qMatrix(n, vector<double>(n, 0.0)); // Initialize Q-matrix
    double previousCost = std::numeric_limits<double>::infinity();
    int stableCostCount = 0; // Counter for consecutive stable costs

    for (int ep = 0; ep < max_iterations; ep++) {
        int state = 0; // Start state
        auto actions = getAvailableActions(state, rMatrix);
        if (actions.empty()) continue;

        while (state != goal) {
            int action = epsilonGreedyAction(state, qMatrix, actions);
            updateQ(state, action, rMatrix, qMatrix, goal);
            state = action;
            actions = getAvailableActions(state, rMatrix);
            if (actions.empty()) break;
        }

        // Decay epsilon after each episode
        if (epsilon > min_epsilon) {
            epsilon *= decay_rate;
        }

        // Detect the optimal path and its cost
        auto [optimalPath, totalCost] = detectOptimalPath(0, goal, rMatrix, qMatrix);

        // Log the optimal path every 10 episodes
        if (ep % 10 == 0) {
            logOptimalPath << "[Episode " << ep << "] ";
            if (!optimalPath.empty()) {
                for (int node : optimalPath) {
                    logOptimalPath << node << " ";
                }
                logOptimalPath << " | Cost: " << totalCost << "\n";
            } else {
                logOptimalPath << "No path found\n";
            }
        }

        // Check if the cost has stabilized
        if (totalCost == previousCost) {
            stableCostCount++;
        } else {
            stableCostCount = 0; // Reset the counter if the cost changes
        }
        previousCost = totalCost;

        // Terminate early if the cost is stable for 10 consecutive iterations
        if (stableCostCount >= 10) {
            logOptimalPath << "[Converged at Episode " << ep << "] ";
            if (!optimalPath.empty()) {
                for (int node : optimalPath) {
                    logOptimalPath << node << " ";
                }
                logOptimalPath << " | Cost: " << totalCost << "\n";
            } else {
                logOptimalPath << "No path found\n";
            }
            break;
        }
    }

    // Log final optimal path after training
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
