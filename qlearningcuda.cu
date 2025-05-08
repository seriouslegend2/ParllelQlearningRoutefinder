#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <limits>
#include <cmath>
#include <time.h>
#include <algorithm>

using namespace std;

// Device-specific constants
#if defined(__CUDA_ARCH__)
    __device__ const double DEVICE_DOUBLE_INFINITY = HUGE_VAL;
    __device__ const double DEVICE_DOUBLE_NEGATIVE_INFINITY = -HUGE_VAL;
#else
    const double DEVICE_DOUBLE_INFINITY = std::numeric_limits<double>::infinity();
    const double DEVICE_DOUBLE_NEGATIVE_INFINITY = -std::numeric_limits<double>::infinity();
#endif

// Global variables
double gammaLR_global = 0.95;
double learningRate_global = 0.1;
double epsilon_global = 1.0;
double min_epsilon_global = 0.01;
double decay_rate_global = 0.999;

// CUDA error checking
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

// CPU Function: getAvailableActions - matches your original implementation
vector<int> getAvailableActions(int state, const vector<vector<double>>& rMatrix, int n_states) {
    vector<int> actions;
    if (state < 0 || state >= n_states || state >= rMatrix.size()) {
        return actions;
    }
    for (int j = 0; j < rMatrix[state].size(); j++) {
        if (rMatrix[state][j] > 0 && rMatrix[state][j] != std::numeric_limits<double>::infinity()) {
            actions.push_back(j);
        }
    }
    return actions;
}

// CUDA Kernel: Initialize cuRAND states
__global__ void initializeCurandStates(curandState* states, unsigned long long seed_base, int num_states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_states) {
        curand_init(seed_base + idx, idx, 0, &states[idx]);
    }
}

// CUDA Kernel: Update Q-values - modified to match sequential logic
__global__ void updateQKernel(double* qMatrix_dev, double* rMatrix_dev, int* availableActions_dev, 
                             int n_states, int goal, double gammaLR, double learningRate, 
                             double current_epsilon, curandState* states_dev) {
    int state = blockIdx.x * blockDim.x + threadIdx.x;
    if (state >= n_states) return;
    
    // Skip goal state
    if (state == goal) return;
    
    curandState localState = states_dev[state];
    int actionCount = availableActions_dev[state * n_states];
    
    if (actionCount == 0) {
        states_dev[state] = localState;
        return;
    }

    // Choose action using epsilon-greedy
    int action_to_take;
    if (curand_uniform_double(&localState) < current_epsilon) {
        // Exploration: random action
        int randIndex = curand(&localState) % actionCount;
        action_to_take = availableActions_dev[state * n_states + 1 + randIndex];
    } else {
        // Exploitation: best action
        int bestAction = availableActions_dev[state * n_states + 1];
        double bestQ = qMatrix_dev[state * n_states + bestAction];
        
        for (int i = 1; i < actionCount; ++i) {
            int currentAction = availableActions_dev[state * n_states + 1 + i];
            double currentQ = qMatrix_dev[state * n_states + currentAction];
            if (currentQ > bestQ) {
                bestQ = currentQ;
                bestAction = currentAction;
            }
        }
        action_to_take = bestAction;
    }

    // Get reward (negative cost)
    double cost = rMatrix_dev[state * n_states + action_to_take];
    if (cost <= 0 || cost == DEVICE_DOUBLE_INFINITY) {
        states_dev[state] = localState;
        return;
    }
    double reward = -cost;

    // Find max Q-value for next state
    int next_state = action_to_take;
    int next_action_count = availableActions_dev[next_state * n_states];
    double max_next_Q = DEVICE_DOUBLE_NEGATIVE_INFINITY;

    if (next_action_count > 0) {
        for (int i = 0; i < next_action_count; ++i) {
            int next_possible_action = availableActions_dev[next_state * n_states + 1 + i];
            max_next_Q = fmax(max_next_Q, qMatrix_dev[next_state * n_states + next_possible_action]);
        }
    } else {
        max_next_Q = 0.0;  // Dead-end
    }

    // Update Q-value using Bellman equation
    double target = reward + gammaLR * max_next_Q;
    qMatrix_dev[state * n_states + action_to_take] += 
        learningRate * (target - qMatrix_dev[state * n_states + action_to_take]);
    
    states_dev[state] = localState;
}

// CPU Function: Prepare available actions for CUDA kernel
vector<int> prepareAvailableActions(int n_states, const vector<vector<double>>& rMatrix_host) {
    vector<int> availableActions_flat(n_states * n_states, 0);
    
    for (int i = 0; i < n_states; ++i) {
        vector<int> actions_for_state = getAvailableActions(i, rMatrix_host, n_states);
        availableActions_flat[i * n_states] = static_cast<int>(actions_for_state.size());
        
        for (size_t j = 0; j < actions_for_state.size(); ++j) {
            if (j < static_cast<size_t>(n_states - 1)) {
                availableActions_flat[i * n_states + 1 + j] = actions_for_state[j];
            }
        }
    }
    return availableActions_flat;
}

// CPU Function: Detect Optimal Path - modified to match sequential logic
pair<vector<int>, double> detectOptimalPath(int start, int goal, 
                                          const vector<vector<double>>& rMatrix_host, 
                                          const vector<double>& qMatrix_flat_host, int n_states) {
    vector<int> path;
    double totalCost = 0.0;
    int currentState = start;

    if (start < 0 || start >= n_states || goal < 0 || goal >= n_states) {
        cerr << "Error: Invalid start or goal state." << endl;
        return {{}, std::numeric_limits<double>::infinity()};
    }

    path.push_back(currentState);
    vector<bool> visited(n_states, false);
    visited[currentState] = true;
    int step_count = 0;
    const int max_steps = n_states * 2;

    while (currentState != goal && step_count++ < max_steps) {
        vector<int> actions = getAvailableActions(currentState, rMatrix_host, n_states);
        if (actions.empty()) {
            return {{}, std::numeric_limits<double>::infinity()};
        }

        int nextState = -1;
        double currentMaxQ = -std::numeric_limits<double>::infinity();

        // First try unvisited states
        for (int action : actions) {
            if (action < 0 || action >= n_states) continue;
            
            double q_value = qMatrix_flat_host[currentState * n_states + action];
            if (!visited[action] && q_value > currentMaxQ &&
                rMatrix_host[currentState][action] > 0 &&
                rMatrix_host[currentState][action] != std::numeric_limits<double>::infinity()) {
                nextState = action;
                currentMaxQ = q_value;
            }
        }

        // If no unvisited state, pick best Q-value
        if (nextState == -1) {
            for (int action : actions) {
                if (action < 0 || action >= n_states) continue;
                
                double q_value = qMatrix_flat_host[currentState * n_states + action];
                if (q_value > currentMaxQ &&
                    rMatrix_host[currentState][action] > 0 &&
                    rMatrix_host[currentState][action] != std::numeric_limits<double>::infinity()) {
                    nextState = action;
                    currentMaxQ = q_value;
                }
            }
            
            if (nextState == -1) {
                return {{}, std::numeric_limits<double>::infinity()};
            }
        }
        
        if (nextState < 0 || nextState >= n_states) {
            cerr << "Error: Determined nextState " << nextState << " is out of bounds in detectOptimalPath." << endl;
            return {{}, std::numeric_limits<double>::infinity()};
        }
        
        if (currentState >= rMatrix_host.size() || nextState >= rMatrix_host[currentState].size()) {
            cerr << "Error: rMatrix_host access out of bounds in detectOptimalPath." << endl;
            return {{}, std::numeric_limits<double>::infinity()};
        }
        
        totalCost += rMatrix_host[currentState][nextState];
        currentState = nextState;
        path.push_back(currentState);
        
        if (currentState < n_states) {
            visited[currentState] = true;
        } else {
            cerr << "Error: Path went out of bounds in detectOptimalPath." << endl;
            return {{}, std::numeric_limits<double>::infinity()};
        }
    }

    if (currentState == goal) {
        return {path, totalCost};
    } else {
        return {{}, std::numeric_limits<double>::infinity()};
    }
}

// Main CUDA Q-learning function
int qLearningCUDA(int n_states, vector<vector<double>>& rMatrix_host, int goal, 
                 int max_iterations, ofstream& logOptimalPath) {
    // Add GPU monitoring
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Prepare data for GPU
    vector<int> availableActions_flat_host = prepareAvailableActions(n_states, rMatrix_host);
    vector<double> rMatrix_flat_host(n_states * n_states);
    vector<double> qMatrix_flat_host(n_states * n_states, 0.0);

    // Flatten R-matrix for GPU
    for (int i = 0; i < n_states; ++i) {
        for (int j = 0; j < n_states; ++j) {
            if (i < rMatrix_host.size() && j < rMatrix_host[i].size()) {
                rMatrix_flat_host[i * n_states + j] = rMatrix_host[i][j];
            } else {
                rMatrix_flat_host[i * n_states + j] = std::numeric_limits<double>::infinity();
            }
        }
    }

    // Allocate GPU memory
    double* d_qMatrix;
    double* d_rMatrix;
    int* d_availableActions;
    curandState* d_states;

    gpuErrchk(cudaMalloc(&d_qMatrix, n_states * n_states * sizeof(double)));
    gpuErrchk(cudaMalloc(&d_rMatrix, n_states * n_states * sizeof(double)));
    gpuErrchk(cudaMalloc(&d_availableActions, n_states * n_states * sizeof(int)));
    gpuErrchk(cudaMalloc(&d_states, n_states * sizeof(curandState)));

    // Copy data to GPU
    gpuErrchk(cudaMemcpy(d_qMatrix, qMatrix_flat_host.data(), 
                        n_states * n_states * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_rMatrix, rMatrix_flat_host.data(), 
                        n_states * n_states * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_availableActions, availableActions_flat_host.data(), 
                        n_states * n_states * sizeof(int), cudaMemcpyHostToDevice));

    // Initialize random states
    int threadsPerBlockInit = 256;
    int numBlocksInit = (n_states + threadsPerBlockInit - 1) / threadsPerBlockInit;
    initializeCurandStates<<<numBlocksInit, threadsPerBlockInit>>>(
        d_states, (unsigned long long)time(NULL), n_states);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());

    // Training parameters
    int threadsPerBlockQ = 256;
    int numBlocksQ = (n_states + threadsPerBlockQ - 1) / threadsPerBlockQ;
    double current_epsilon = epsilon_global;
    double previousCost = std::numeric_limits<double>::infinity();
    int stableCostCount = 0;

    // Main training loop
    for (int ep = 0; ep < max_iterations; ep++) {
        // Record start time
        cudaEventRecord(start);
        
        // Update Q-values in parallel
        updateQKernel<<<numBlocksQ, threadsPerBlockQ>>>(
            d_qMatrix, d_rMatrix, d_availableActions, n_states, goal,
            gammaLR_global, learningRate_global, current_epsilon, d_states);
        gpuErrchk(cudaGetLastError());
        gpuErrchk(cudaDeviceSynchronize());
        
        // Record stop time and calculate elapsed time
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        
        if (ep % 100 == 0) {
            cout << "Episode " << ep << " kernel execution time: " << milliseconds << " ms" << endl;
        }

        // Decay epsilon
        if (current_epsilon > min_epsilon_global) {
            current_epsilon *= decay_rate_global;
        } else {
            current_epsilon = min_epsilon_global;
        }

        // Check for convergence every 10 episodes
        if (ep % 10 == 0 || ep == max_iterations - 1) {
            // Copy Q-matrix back to host for path detection
            gpuErrchk(cudaMemcpy(qMatrix_flat_host.data(), d_qMatrix, 
                                n_states * n_states * sizeof(double), cudaMemcpyDeviceToHost));
            
            // Detect optimal path
            auto result = detectOptimalPath(0, goal, rMatrix_host, qMatrix_flat_host, n_states);
            vector<int> optimalPath = result.first;
            double totalCost = result.second;
            
            // Log current path
            logOptimalPath << "[CUDA Episode " << ep << "] ";
            if (!optimalPath.empty()) {
                for (int node : optimalPath) {
                    logOptimalPath << node << " ";
                }
                logOptimalPath << "| Cost: " << totalCost << "\n";
            } else {
                logOptimalPath << "No valid path found\n";
            }
            
            // Check for convergence
            if (totalCost == previousCost) {
                stableCostCount++;
                if (stableCostCount >= 10) {
                    logOptimalPath << "[CUDA Converged at Episode " << ep << "]\n";
                    break;
                }
            } else {
                stableCostCount = 0;
            }
            previousCost = totalCost;
        }
    }

    // Get final Q-matrix and path
    gpuErrchk(cudaMemcpy(qMatrix_flat_host.data(), d_qMatrix, 
                        n_states * n_states * sizeof(double), cudaMemcpyDeviceToHost));
    
    pair<vector<int>, double> result = detectOptimalPath(0, goal, rMatrix_host, qMatrix_flat_host, n_states);
    vector<int> optimalPath = result.first;
    double totalCost = result.second;

    // Log final path
    logOptimalPath << "[CUDA Final Path] ";
    if (!optimalPath.empty()) {
        for (int node : optimalPath) {
            logOptimalPath << node << " ";
        }
        logOptimalPath << "| Cost: " << totalCost << "\n";
    } else {
        logOptimalPath << "No valid path found\n";
    }

    // Log final Q-matrix
    logOptimalPath << "[CUDA Final Q-Matrix]\n";
    for (int i = 0; i < n_states; i++) {
        for (int j = 0; j < n_states; j++) {
            logOptimalPath << qMatrix_flat_host[i * n_states + j] << " ";
        }
        logOptimalPath << "\n";
    }

    // Clean up CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Free GPU memory
    gpuErrchk(cudaFree(d_qMatrix));
    gpuErrchk(cudaFree(d_rMatrix));
    gpuErrchk(cudaFree(d_availableActions));
    gpuErrchk(cudaFree(d_states));

    return max_iterations;
}
