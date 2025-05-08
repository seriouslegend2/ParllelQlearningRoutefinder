#include "qlearning.h"

// Define global variables
double gammaLR = 0.95;                           // Discount factor
double learningRate = 0.1;                      // Learning rate
double epsilon = 1.0, min_epsilon = 0.01;       // Exploration settings
double decay_rate = 0.999;                      // Epsilon decay per episode

// Function definitions
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

int getRandomAction(const vector<int>& availableActions) {
    return availableActions[rand() % availableActions.size()];
}

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

int qLearning(int n, vector<vector<double>>& rMatrix, int goal, int max_iterations, ofstream& logStateSpace, ofstream& logOptimalPath) {
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

        if (epsilon > min_epsilon) {
            epsilon *= decay_rate;
        }

        auto result = detectOptimalPath(0, goal, rMatrix, qMatrix);
        vector<int> optimalPath = result.first;
        double totalCost = result.second;

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

        if (totalCost == previousCost) {
            stableCostCount++;
        } else {
            stableCostCount = 0;
        }
        previousCost = totalCost;

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
            return ep;  // Return the convergence iteration
        }
    }

    auto result = detectOptimalPath(0, goal, rMatrix, qMatrix);
    vector<int> optimalPath = result.first;
    double totalCost = result.second;
    logOptimalPath << "[Final] ";
    if (!optimalPath.empty()) {
        for (int node : optimalPath) {
            logOptimalPath << node << " ";
        }
        logOptimalPath << " | Cost: " << totalCost << "\n";
    } else {
        logOptimalPath << "No path found\n";
    }

    return max_iterations;  // Return max_iterations if no convergence
}
