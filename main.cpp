#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include "qlearning.h"
#include "qlearning_openMP.h"
#include "other_algorithms.h"

// Declare CUDA function
void qLearningCUDA(int n, std::vector<std::vector<double>>& rMatrix, int goal, int max_iterations, std::ofstream& logOptimalPath);

// Function to generate a random connected graph
void generateRandomGraph(int n, std::vector<std::vector<double>>& rMatrix, int goal) {
    // Initialize all edges to -1.0 (indicating no connection)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            rMatrix[i][j] = -1.0;
        }
    }

    // Initialize the diagonal to 0.0 to avoid self-loops with negative weights
    for (int i = 0; i < n; i++) {
        rMatrix[i][i] = 0.0;
    }

    // Randomly assign edges with strictly non-negative weights
    for (int i = 0; i < n; i++) {
        int num_edges = rand() % (n / 2 + 1) + 1; // Ensure at least one edge
        for (int j = 0; j < num_edges; j++) {
            int dest = rand() % n;
            if (dest != i) {
                double reward = (dest == goal) ? 100.0 : (rand() % 100 + 1); // Ensure strictly non-negative weights
                rMatrix[i][dest] = reward;
                rMatrix[dest][i] = reward; // Undirected graph
            }
        }
    }

    // Ensure the graph is connected
    for (int i = 0; i < n - 1; i++) {
        rMatrix[i][i + 1] = rand() % 100 + 1; // Ensure strictly non-negative weights
        rMatrix[i + 1][i] = rMatrix[i][i + 1];
    }

    // Final validation: Ensure no invalid weights remain
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (rMatrix[i][j] == -1.0 && i != j) {
                rMatrix[i][j] = INFINITY; // Mark unconnected edges with infinity
            }
        }
    }

    // Debug: Log generated weights
    std::cout << "Generated graph weights:" << std::endl;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (rMatrix[i][j] != INFINITY) {
                std::cout << "Edge (" << i << ", " << j << ") -> Weight: " << rMatrix[i][j] << std::endl;
            }
        }
    }
}

int main() {
    srand(time(NULL));

    int n, iterations, goal;
    std::cout << "Enter the number of fog nodes: ";
    std::cin >> n;

    std::cout << "Enter the number of iterations (recommended: 5000 or more): ";
    std::cin >> iterations;

    std::cout << "Enter the goal state (0 to " << n - 1 << "): ";
    std::cin >> goal;

    // Validate user inputs
    if (n <= 0) {
        std::cout << "Error: Number of fog nodes must be greater than 0." << std::endl;
        return -1;
    }
    if (iterations <= 0) {
        std::cout << "Error: Number of iterations must be greater than 0." << std::endl;
        return -1;
    }
    if (goal < 0 || goal >= n) {
        std::cout << "Error: Goal state must be between 0 and " << n - 1 << "." << std::endl;
        return -1;
    }

    std::vector<std::vector<double>> rMatrix(n, std::vector<double>(n, -1.0)); // Reward matrix (graph)
    
    // Debug: Log graph generation
    std::cout << "Generating random graph..." << std::endl;
    generateRandomGraph(n, rMatrix, goal);
    std::cout << "Graph generated successfully." << std::endl;

    // Create log files
    std::ofstream logStateSpace("qlearning_results.csv");
    std::ofstream logOptimalPath("optimal_path_results.csv");
    std::ofstream logCUDA("cuda_results.csv");
    std::ofstream logTimeResults("time_results.csv");
    std::ofstream logOpenMP("openmp_results.csv");

    // Sequential Q-learning
    std::cout << "Starting Sequential Q-learning..." << std::endl;
    auto startSeq = std::chrono::high_resolution_clock::now();
    int convergence_iteration = qLearning(n, rMatrix, goal, iterations, logStateSpace, logOptimalPath);
    auto endSeq = std::chrono::high_resolution_clock::now();
    double timeSeq = std::chrono::duration<double>(endSeq - startSeq).count();
    std::cout << "Sequential Q-learning completed. Converged at iteration: " << convergence_iteration << std::endl;

    // Calculate CUDA iterations (130% of convergence iteration)
    int cuda_iterations = static_cast<int>(convergence_iteration * 2.3);
    std::cout << "Using " << cuda_iterations << " iterations for CUDA Q-learning..." << std::endl;

    // OpenMP Q-learning
    std::cout << "Starting OpenMP Q-learning..." << std::endl;
    auto startOpenMP = std::chrono::high_resolution_clock::now();
    qLearningOpenMP(n, rMatrix, goal, iterations, logStateSpace, logOpenMP);
    auto endOpenMP = std::chrono::high_resolution_clock::now();
    double timeOpenMP = std::chrono::duration<double>(endOpenMP - startOpenMP).count();
    std::cout << "OpenMP Q-learning completed." << std::endl;

    // CUDA Q-learning
    std::cout << "Starting CUDA Q-learning..." << std::endl;
    auto startCUDA = std::chrono::high_resolution_clock::now();
    qLearningCUDA(n, rMatrix, goal, cuda_iterations, logCUDA);
    auto endCUDA = std::chrono::high_resolution_clock::now();
    double timeCUDA = std::chrono::duration<double>(endCUDA - startCUDA).count();
    std::cout << "CUDA Q-learning completed." << std::endl;

    // Dijkstra's Algorithm
    std::cout << "Starting Dijkstra's Algorithm..." << std::endl;
    auto startDijkstra = std::chrono::high_resolution_clock::now();
    dijkstra(n, rMatrix, 0, goal, logOptimalPath);
    auto endDijkstra = std::chrono::high_resolution_clock::now();
    double timeDijkstra = std::chrono::duration<double>(endDijkstra - startDijkstra).count();
    std::cout << "Dijkstra's Algorithm completed." << std::endl;

    // A* Search Algorithm
    std::cout << "Starting A* Search Algorithm..." << std::endl;
    auto startAStar = std::chrono::high_resolution_clock::now();
    aStar(n, rMatrix, 0, goal, logOptimalPath);
    auto endAStar = std::chrono::high_resolution_clock::now();
    double timeAStar = std::chrono::duration<double>(endAStar - startAStar).count();
    std::cout << "A* Search Algorithm completed." << std::endl;

    // Save time comparison
    logTimeResults << "Implementation,Time (s)\n";
    logTimeResults << "Sequential Q-learning," << timeSeq << "\n";
    logTimeResults << "OpenMP Q-learning," << timeOpenMP << "\n";
    logTimeResults << "CUDA Q-learning," << timeCUDA << "\n";
    logTimeResults << "Dijkstra's Algorithm," << timeDijkstra << "\n";
    logTimeResults << "A* Search Algorithm," << timeAStar << "\n";

    std::cout << "Time comparison saved to time_results.csv\n";

    return 0;
}
