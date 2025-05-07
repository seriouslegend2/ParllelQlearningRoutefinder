#ifndef OTHER_ALGORITHMS_H
#define OTHER_ALGORITHMS_H

#include <iostream>
#include <vector>
#include <queue>
#include <cmath>
#include <functional>
#include <algorithm>
#include <fstream>

using namespace std;

// Simple heuristic function (Manhattan distance)
double heuristic(int node, int goal) {
    return abs(goal - node);
}

// Dijkstra's algorithm
void dijkstra(int n, vector<vector<double>>& rMatrix, int start, int goal, ofstream& logOptimalPath) {
    cout << "Running Dijkstra's Algorithm..." << endl; // Debug log
    vector<double> dist(n, INFINITY); // Distance vector initialized to infinity
    vector<int> prev(n, -1);          // Previous node vector initialized to -1
    priority_queue<pair<double, int>, vector<pair<double, int>>, greater<pair<double, int>>> pq;

    dist[start] = 0;  // Start node distance is 0
    pq.push({0, start});  // Push the start node to the priority queue

    int iteration_count = 0; // Safeguard to prevent infinite loops
    const int max_iterations = n * n; // Arbitrary upper limit for iterations

    while (!pq.empty()) {
        if (++iteration_count > max_iterations) {
            cerr << "Error: Dijkstra's Algorithm exceeded maximum iterations. Possible infinite loop." << endl;
            logOptimalPath << "Dijkstra's Path: Algorithm terminated due to excessive iterations.\n";
            return;
        }

        int u = pq.top().second;  // Get the node with the smallest distance
        pq.pop();

        cout << "Processing node " << u << " with current distance " << dist[u] << endl; // Debug log

        if (u == goal) break;  // Exit early if goal is reached

        for (int v = 0; v < n; v++) {
            if (rMatrix[u][v] != -1.0) {  // Valid edge
                double alt = dist[u] + rMatrix[u][v];  // Alternative distance
                if (alt < dist[v]) {  // Relaxation step
                    dist[v] = alt;
                    prev[v] = u;
                    pq.push({dist[v], v});  // Push updated distance to priority queue
                    cout << "Updated distance of node " << v << " to " << dist[v] << endl; // Debug log
                }
            }
        }
    }

    if (dist[goal] == INFINITY) {
        logOptimalPath << "Dijkstra's Path: No path found to goal " << goal << ".\n";
        cout << "Dijkstra's Algorithm: No path found to goal " << goal << "." << endl; // Debug log
        return;
    }

    // Reconstruct the optimal path from the previous nodes
    vector<int> path;
    for (int u = goal; u != -1; u = prev[u]) {
        path.push_back(u);
    }
    reverse(path.begin(), path.end());

    // Log the optimal path and its cost
    logOptimalPath << "Dijkstra's Path: ";
    for (int node : path) {
        logOptimalPath << node << " ";
    }
    logOptimalPath << " | Cost: " << dist[goal] << "\n";
    cout << "Dijkstra's Algorithm: Path found and logged with cost " << dist[goal] << "." << endl; // Debug log
}

// A* Search Algorithm
void aStar(int n, vector<vector<double>>& rMatrix, int start, int goal, ofstream& logOptimalPath) {
    cout << "Running A* Search Algorithm..." << endl; // Debug log
    vector<double> dist(n, INFINITY);  // Distance vector initialized to infinity
    vector<double> heuristic_cost(n, INFINITY);  // Heuristic cost vector initialized to infinity
    vector<int> prev(n, -1);  // Previous node vector initialized to -1
    priority_queue<pair<double, int>, vector<pair<double, int>>, greater<pair<double, int>>> pq;

    dist[start] = 0;  // Start node distance is 0
    heuristic_cost[start] = heuristic(start, goal);  // Calculate the heuristic cost for start node
    pq.push({dist[start] + heuristic_cost[start], start});  // Push the start node to the priority queue

    int iteration_count = 0; // Safeguard to prevent infinite loops
    const int max_iterations = n * n; // Arbitrary upper limit for iterations

    while (!pq.empty()) {
        if (++iteration_count > max_iterations) {
            cerr << "Error: A* Search Algorithm exceeded maximum iterations. Possible infinite loop." << endl;
            logOptimalPath << "A* Path: Algorithm terminated due to excessive iterations.\n";
            return;
        }

        int u = pq.top().second;  // Get the node with the smallest f_score
        pq.pop();

        cout << "Processing node " << u << " with current f_score " << dist[u] + heuristic_cost[u] << endl; // Debug log

        if (u == goal) break;  // Exit early if goal is reached

        for (int v = 0; v < n; v++) {
            if (rMatrix[u][v] != -1.0) {  // Valid edge
                double alt = dist[u] + rMatrix[u][v];  // Alternative distance
                if (alt < dist[v]) {  // Relaxation step
                    dist[v] = alt;
                    prev[v] = u;
                    heuristic_cost[v] = heuristic(v, goal);  // Calculate heuristic for v
                    pq.push({dist[v] + heuristic_cost[v], v});  // Push updated f_score to priority queue
                    cout << "Updated f_score of node " << v << " to " << dist[v] + heuristic_cost[v] << endl; // Debug log
                }
            }
        }
    }

    if (dist[goal] == INFINITY) {
        logOptimalPath << "A* Path: No path found to goal " << goal << ".\n";
        cout << "A* Search Algorithm: No path found to goal " << goal << "." << endl; // Debug log
        return;
    }

    // Reconstruct the optimal path from the previous nodes
    vector<int> path;
    for (int u = goal; u != -1; u = prev[u]) {
        path.push_back(u);
    }
    reverse(path.begin(), path.end());

    // Log the optimal path
    logOptimalPath << "A* Path: ";
    for (int node : path) {
        logOptimalPath << node << " ";
    }
    logOptimalPath << " | Cost: " << dist[goal] << "\n";
    cout << "A* Search Algorithm: Path found and logged with cost " << dist[goal] << "." << endl; // Debug log
}

#endif
