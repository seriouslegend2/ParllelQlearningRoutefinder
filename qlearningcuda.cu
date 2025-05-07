#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <limits>
#include <cmath>

using namespace std;

// CUDA kernel for Q-matrix update
__global__ void updateQKernel(double* rMatrix, double* qMatrix, int n, double gammaLR, double learningRate) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * n) return;

    int currentState = idx / n;
    int action = idx % n;

    double cost = rMatrix[currentState * n + action];
    if (cost <= 0 || cost == INFINITY) return;

    double reward = -cost;
    double maxQ = -INFINITY;

    for (int next = 0; next < n; next++) {
        maxQ = max(maxQ, qMatrix[action * n + next]);
    }

    double target = reward + gammaLR * maxQ;
    qMatrix[currentState * n + action] += learningRate * (target - qMatrix[currentState * n + action]);
}

// CUDA kernel for epsilon-greedy action selection using parallel reduction
__global__ void epsilonGreedyKernel(double* qMatrix, int* availableActions, int n, double epsilon, int* selectedActions, curandState* states) {
    int state = blockIdx.x;
    if (state >= n) return;

    curandState localState = states[state];
    int actionCount = availableActions[state * n];
    if (actionCount == 0) return;

    if (curand_uniform(&localState) < epsilon) {
        selectedActions[state] = availableActions[state * n + (curand(&localState) % actionCount)];
    } else {
        __shared__ double sharedMaxQ[256];
        __shared__ int sharedBestAction[256];

        int tid = threadIdx.x;
        int actionIdx = tid;
        double maxQ = -INFINITY;
        int bestAction = -1;

        while (actionIdx < actionCount) {
            int action = availableActions[state * n + actionIdx];
            double qValue = qMatrix[state * n + action];
            if (qValue > maxQ) {
                maxQ = qValue;
                bestAction = action;
            }
            actionIdx += blockDim.x;
        }

        sharedMaxQ[tid] = maxQ;
        sharedBestAction[tid] = bestAction;
        __syncthreads();

        // Parallel reduction to find the best action
        for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if (tid < stride) {
                if (sharedMaxQ[tid + stride] > sharedMaxQ[tid]) {
                    sharedMaxQ[tid] = sharedMaxQ[tid + stride];
                    sharedBestAction[tid] = sharedBestAction[tid + stride];
                }
            }
            __syncthreads();
        }

        if (tid == 0) {
            selectedActions[state] = sharedBestAction[0];
        }
    }

    states[state] = localState;
}

// Q-learning function
void qLearningCUDA(int n, vector<vector<double>>& rMatrix, int goal, int max_iterations, ofstream& logOptimalPath) {
    // Flatten matrices for CUDA
    vector<double> rMatrixFlat(n * n, 0.0), qMatrixFlat(n * n, 0.0);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            rMatrixFlat[i * n + j] = rMatrix[i][j];
        }
    }

    double *d_rMatrix, *d_qMatrix;
    cudaMalloc(&d_rMatrix, n * n * sizeof(double));
    cudaMalloc(&d_qMatrix, n * n * sizeof(double));
    cudaMemcpy(d_rMatrix, rMatrixFlat.data(), n * n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_qMatrix, qMatrixFlat.data(), n * n * sizeof(double), cudaMemcpyHostToDevice);

    // Initialize random states for curand
    curandState* d_states;
    cudaMalloc(&d_states, n * sizeof(curandState));

    int threadsPerBlock = 256;
    int blocks = (n * n + threadsPerBlock - 1) / threadsPerBlock;

    for (int ep = 0; ep < max_iterations; ep++) {
        updateQKernel<<<blocks, threadsPerBlock>>>(d_rMatrix, d_qMatrix, n, 0.95, 0.1);
        // Removed unnecessary cudaDeviceSynchronize() here
    }

    cudaMemcpy(qMatrixFlat.data(), d_qMatrix, n * n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_rMatrix);
    cudaFree(d_qMatrix);
    cudaFree(d_states);

    // Log final Q-matrix
    logOptimalPath << "[CUDA Final Q-Matrix]\n";
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            logOptimalPath << qMatrixFlat[i * n + j] << " ";
        }
        logOptimalPath << "\n";
    }
}
