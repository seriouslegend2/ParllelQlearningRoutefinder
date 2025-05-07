#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <chrono>
#include <sys/resource.h> // For memory usage tracking
#include "qlearning_openMP.h"
#include "other_algorithms.h"
#include <fstream> // For reading memory usage from /proc/self/statm
#include <unistd.h> // For sysconf and _SC_PAGESIZE

using namespace std;
using namespace chrono;

// Function to generate a random connected graph
void generateRandomGraph(int n, vector<vector<double>>& rMatrix, int goal) {
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
    cout << "Generated graph weights:" << endl;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (rMatrix[i][j] != INFINITY) {
                cout << "Edge (" << i << ", " << j << ") -> Weight: " << rMatrix[i][j] << endl;
            }
        }
    }
}

// Function to get memory usage in kilobytes from /proc/self/statm
long getMemoryUsage() {
    std::ifstream statm("/proc/self/statm");
    if (!statm.is_open()) {
        cerr << "Error: Unable to read memory usage from /proc/self/statm." << endl;
        return 0;
    }
    long pages;
    statm >> pages; // Read the number of memory pages
    statm.close();
    return pages * sysconf(_SC_PAGESIZE) / 1024; // Convert pages to kilobytes
}

int main() {
    srand(time(NULL));

    int n, iterations, goal;
    cout << "Enter the number of fog nodes: ";
    cin >> n;

    cout << "Enter the number of iterations (recommended: 5000 or more): ";
    cin >> iterations;

    cout << "Enter the goal state (0 to " << n - 1 << "): ";
    cin >> goal;

    // Validate user inputs
    if (n <= 0) {
        cout << "Error: Number of fog nodes must be greater than 0." << endl;
        return -1;
    }
    if (iterations <= 0) {
        cout << "Error: Number of iterations must be greater than 0." << endl;
        return -1;
    }
    if (goal < 0 || goal >= n) {
        cout << "Error: Goal state must be between 0 and " << n - 1 << "." << endl;
        return -1;
    }

    vector<vector<double>> rMatrix(n, vector<double>(n, -1.0)); // Reward matrix (graph)
    
    // Debug: Log graph generation
    cout << "Generating random graph..." << endl;
    generateRandomGraph(n, rMatrix, goal);
    cout << "Graph generated successfully." << endl;

    // Create log files
    ofstream logStateSpace("qlearning_results.csv");
    ofstream logOptimalPath("optimal_path_results.csv");
    ofstream logTimeResults("time_results.csv");

    logStateSpace << "Iteration,State,Q-Values\n";
    logOptimalPath << "Algorithm,Path\n";
    logTimeResults << "Algorithm,Time(ms),Memory(KB)\n";

    // Debug: Log algorithm execution
    cout << "Starting Q-learning with OpenMP..." << endl;
    auto start_time = high_resolution_clock::now();
    long start_memory = getMemoryUsage();
    qLearningOpenMP(n, rMatrix, goal, iterations, logStateSpace, logOptimalPath);
    auto end_time = high_resolution_clock::now();
    long end_memory = getMemoryUsage();
    long qlearning_openmp_time = duration_cast<milliseconds>(end_time - start_time).count();
    long qlearning_openmp_memory = end_memory - start_memory;
    logTimeResults << "Q-learning (OpenMP)," << qlearning_openmp_time << "," << qlearning_openmp_memory << "\n";
    cout << "Q-learning with OpenMP completed in " << qlearning_openmp_time << " ms, memory used: " << qlearning_openmp_memory << " KB." << endl;

    // Debug: Log Q-learning completion
    cout << "Q-learning results logged to qlearning_results.csv and optimal_path_results.csv." << endl;

    // Debug: Check graph connectivity
    cout << "Checking graph connectivity..." << endl;
    bool is_connected = false;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (rMatrix[i][j] != -1.0) {
                is_connected = true;
                break;
            }
        }
        if (is_connected) break;
    }
    if (!is_connected) {
        cerr << "Error: The generated graph is not connected. Dijkstra's Algorithm cannot proceed." << endl;
        return -1;
    }
    cout << "Graph connectivity check passed." << endl;

    // Debug: Check for negative weights
    cout << "Checking for negative weights in the graph..." << endl;
    bool has_negative_weights = false;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (rMatrix[i][j] < 0) {
                has_negative_weights = true;
                cout << "Negative weight detected on edge (" << i << ", " << j << ") -> Weight: " << rMatrix[i][j] << endl;
                break;
            }
        }
        if (has_negative_weights) break;
    }
    if (has_negative_weights) {
        cerr << "Error: The generated graph contains negative weights. Exiting." << endl;
        return -1;
    }
    cout << "Graph weight check passed. No negative weights found." << endl;

    // Start Dijkstra's Algorithm
    cout << "Starting Dijkstra's Algorithm..." << endl;
    start_time = high_resolution_clock::now();
    start_memory = getMemoryUsage();
    dijkstra(n, rMatrix, 0, goal, logOptimalPath);
    end_time = high_resolution_clock::now();
    end_memory = getMemoryUsage();
    long dijkstra_time = duration_cast<microseconds>(end_time - start_time).count(); // Use microseconds for higher resolution
    long dijkstra_memory = max(0L, end_memory - start_memory); // Ensure non-negative memory usage
    logTimeResults << "Dijkstra," << dijkstra_time << "," << dijkstra_memory << "\n";
    cout << "Dijkstra's Algorithm completed in " << dijkstra_time << " µs, memory used: " << dijkstra_memory << " KB." << endl;

    // Start A* Search Algorithm
    cout << "Starting A* Search Algorithm..." << endl;
    start_time = high_resolution_clock::now();
    start_memory = getMemoryUsage();
    aStar(n, rMatrix, 0, goal, logOptimalPath);
    end_time = high_resolution_clock::now();
    end_memory = getMemoryUsage();
    long astar_time = duration_cast<microseconds>(end_time - start_time).count(); // Use microseconds for higher resolution
    long astar_memory = max(0L, end_memory - start_memory); // Ensure non-negative memory usage
    logTimeResults << "A*," << astar_time << "," << astar_memory << "\n";
    cout << "A* Search Algorithm completed in " << astar_time << " µs, memory used: " << astar_memory << " KB." << endl;

    // Debug: Confirm comparison
    cout << "Comparison of Q-learning, Dijkstra's, and A* results logged to time_results.csv." << endl;

    // Ensure log files are flushed and closed
    logStateSpace.close();
    logOptimalPath.close();
    logTimeResults.close();

    cout << "Results have been logged to qlearning_results.csv, optimal_path_results.csv, and time_results.csv." << endl;

    return 0;
}
