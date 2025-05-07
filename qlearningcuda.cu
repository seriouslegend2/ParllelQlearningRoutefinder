#include "qlearningcuda.h"
#include "qlearning.h" // Include for global variables and shared functions
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <random> // For random number generation on the CPU

// Redefine CUDA_CHECK macro
#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t error = call;                                                 \
        if (error != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s:%d, code %d (%s)\n", __FILE__,         \
                    __LINE__, error, cudaGetErrorString(error));                  \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

// Kernel to update Q-matrix
__global__ void updateQMatrixKernel(double* qMatrix, const double* rMatrix, int n, double learningRate, double gammaLR) {
    int state = blockIdx.x * blockDim.x + threadIdx.x;
    int action = blockIdx.y * blockDim.y + threadIdx.y;

    if (state < n && action < n) {
        double cost = rMatrix[state * n + action];  // rMatrix is flattened in row-major order

        if (cost > 0 && cost != INFINITY) {
            double reward = -cost;

            // Find max Q-value for the next state
            double maxQ = -INFINITY;
            for (int next = 0; next < n; ++next) {
                maxQ = max(maxQ, qMatrix[action * n + next]);
            }

            // Update Q-value
            double target = reward + gammaLR * maxQ;
            qMatrix[state * n + action] += learningRate * (target - qMatrix[state * n + action]);
        }
    }
}

// Kernel for epsilon-greedy action selection using parallel reduction
__global__ void epsilonGreedyActionKernel(double* qMatrix, int* actions, int n, double epsilon, curandState* randState) {
    int state = blockIdx.x * blockDim.x + threadIdx.x;

    if (state < n) {
        curandState localState = randState[state];
        double randVal = curand_uniform(&localState);

        if (randVal < epsilon) {
            // Take random action
            actions[state] = state % n; // Simplified random action
        } else {
            // Parallel reduction to find the best action
            __shared__ double sharedMaxQ[32];
            __shared__ int sharedBestAction[32];

            int threadId = threadIdx.x;
            double maxQ = -INFINITY;
            int bestAction = 0;

            for (int action = threadId; action < n; action += blockDim.x) {
                double qValue = qMatrix[state * n + action];
                if (qValue > maxQ) {
                    maxQ = qValue;
                    bestAction = action;
                }
            }

            sharedMaxQ[threadId] = maxQ;
            sharedBestAction[threadId] = bestAction;
            __syncthreads();

            // Reduce within the block
            for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
                if (threadId < stride) {
                    if (sharedMaxQ[threadId + stride] > sharedMaxQ[threadId]) {
                        sharedMaxQ[threadId] = sharedMaxQ[threadId + stride];
                        sharedBestAction[threadId] = sharedBestAction[threadId + stride];
                    }
                }
                __syncthreads();
            }

            if (threadId == 0) {
                actions[state] = sharedBestAction[0];
            }
        }

        randState[state] = localState;
    }
}

// Kernel to initialize curand states
__global__ void initializeCurandStates(curandState* randState, unsigned long seed, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        curand_init(seed, id, 0, &randState[id]);
    }
}

// Function to generate a random graph (R-matrix) on the CPU
void generateRandomGraphCPU(vector<vector<double>>& rMatrix, int n, double maxWeight, double sparsity) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> weightDist(1.0, maxWeight); // Random weights
    std::uniform_real_distribution<> sparsityDist(0.0, 1.0);    // Sparsity control

    rMatrix.resize(n, vector<double>(n, std::numeric_limits<double>::infinity()));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i != j && sparsityDist(gen) > sparsity) {
                rMatrix[i][j] = weightDist(gen); // Assign random weight
            }
        }
    }
}

// Kernel to generate a random graph (R-matrix) on the GPU
__global__ void generateRandomGraphKernel(double* rMatrix, int n, double maxWeight, double sparsity, curandState* randState) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < n && col < n) {
        if (row == col) {
            rMatrix[row * n + col] = INFINITY; // No self-loops
        } else {
            curandState localState = randState[row * n + col];
            double randWeight = curand_uniform(&localState) * maxWeight;
            double randSparsity = curand_uniform(&localState);

            if (randSparsity > sparsity) {
                rMatrix[row * n + col] = randWeight; // Assign random weight
            } else {
                rMatrix[row * n + col] = INFINITY; // No edge
            }

            randState[row * n + col] = localState;
        }
    }
}

// Function to generate a random graph (R-matrix) on the GPU
void generateRandomGraphGPU(double* rMatrix_d, int n, double maxWeight, double sparsity, curandState* randState_d) {
    dim3 blockDim(16, 16);
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x, (n + blockDim.y - 1) / blockDim.y);

    generateRandomGraphKernel<<<gridDim, blockDim>>>(rMatrix_d, n, maxWeight, sparsity, randState_d);
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Main CUDA Q-learning function
void qLearningCUDA(int n, vector<vector<double>>& rMatrix, int goal, int max_iterations,
                   ofstream& logStateSpace, ofstream& logOptimalPath) {
    // Flatten rMatrix for CUDA
    vector<double> rMatrixFlat;
    for (const auto& row : rMatrix) {
        rMatrixFlat.insert(rMatrixFlat.end(), row.begin(), row.end());
    }

    // Allocate CUDA memory
    double *qMatrix_d, *rMatrix_d;
    int *actions_d;
    curandState *randState_d;

    CUDA_CHECK(cudaMalloc((void**)&qMatrix_d, n * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&rMatrix_d, n * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&actions_d, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&randState_d, n * sizeof(curandState)));

    // Initialize Q-matrix on host and copy to device
    vector<double> qMatrix(n * n, 0.0);
    CUDA_CHECK(cudaMemcpy(qMatrix_d, qMatrix.data(), n * n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(rMatrix_d, rMatrixFlat.data(), n * n * sizeof(double), cudaMemcpyHostToDevice));

    // Initialize random number generator state
    dim3 blockDim(32);
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x);
    initializeCurandStates<<<gridDim, blockDim>>>(randState_d, time(NULL), n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Main Training Loop
    for (int ep = 0; ep < max_iterations; ep++) {
        // Epsilon-Greedy Action Selection
        epsilonGreedyActionKernel<<<gridDim, blockDim>>>(qMatrix_d, actions_d, n, epsilon, randState_d);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Q-Matrix Update
        dim3 blockDimUpdate(16, 16);
        dim3 gridDimUpdate((n + blockDimUpdate.x - 1) / blockDimUpdate.x,
                           (n + blockDimUpdate.y - 1) / blockDimUpdate.y);

        updateQMatrixKernel<<<gridDimUpdate, blockDimUpdate>>>(qMatrix_d, rMatrix_d, n, learningRate, gammaLR);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Decay epsilon
        if (epsilon > min_epsilon) {
            epsilon *= decay_rate;
        }

        // Log optimal path every 10 episodes
        if (ep % 10 == 0) {
            CUDA_CHECK(cudaMemcpy(qMatrix.data(), qMatrix_d, n * n * sizeof(double), cudaMemcpyDeviceToHost));
            auto result = detectOptimalPath(0, goal, rMatrixFlat, qMatrix, n);
            vector<int> optimalPath = result.first;
            double totalCost = result.second;
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
    }

    // Clean up CUDA memory
    CUDA_CHECK(cudaFree(qMatrix_d));
    CUDA_CHECK(cudaFree(rMatrix_d));
    CUDA_CHECK(cudaFree(actions_d));
    CUDA_CHECK(cudaFree(randState_d));
}

pair<vector<int>, double> detectOptimalPath(int start, int goal, const vector<double>& rMatrixFlat,
                                            const vector<double>& qMatrixFlat, int n) {
    vector<int> path;
    double totalCost = 0.0;
    int currentState = start;
    path.push_back(currentState);
    vector<bool> visited(n, false);
    visited[currentState] = true;
    int step_count = 0;
    const int max_steps = n * 2;  // Prevent infinite loops

    while (currentState != goal && step_count++ < max_steps) {
        vector<int> actions;
        for (int j = 0; j < n; j++) {
            if (rMatrixFlat[currentState * n + j] > 0 &&
                rMatrixFlat[currentState * n + j] != std::numeric_limits<double>::infinity()) {
                actions.push_back(j);
            }
        }

        if (actions.empty()) {
            return {{}, std::numeric_limits<double>::infinity()}; // No path found
        }

        int nextState = -1;
        double maxQ = -std::numeric_limits<double>::infinity();

        for (int action : actions) {
            if (!visited[action] && qMatrixFlat[currentState * n + action] > maxQ &&
                rMatrixFlat[currentState * n + action] != std::numeric_limits<double>::infinity()) {
                nextState = action;
                maxQ = qMatrixFlat[currentState * n + action];
            }
        }

        if (nextState == -1) {
            // No valid next state found, try unvisited ones even with lower Q values
            for (int action : actions) {
                if (qMatrixFlat[currentState * n + action] > maxQ &&
                    rMatrixFlat[currentState * n + action] != std::numeric_limits<double>::infinity()) {
                    nextState = action;
                    maxQ = qMatrixFlat[currentState * n + action];
                }
            }
            if (nextState == -1)
                return {{}, std::numeric_limits<double>::infinity()}; // Total failure
        }

        totalCost += rMatrixFlat[currentState * n + nextState];
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