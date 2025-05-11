# High-Performance Computing (HPC) Project

This project demonstrates high-performance computing techniques, including parallel algorithms, performance optimizations, and benchmarking tools.

## Prerequisites

Ensure the following are installed:
- Python 3.8 or higher
- MPI (Message Passing Interface) library (e.g., OpenMPI or MPICH)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd final HPC
   ```

2. Verify MPI installation:
   ```bash
   mpirun --version
   ```

## Running the Code

1. Navigate to the project directory:
   ```bash
   cd c:\Users\varun\Downloads\final HPC
   ```

2. Compile and link the program:
   ```bash
   nvcc -c qlearningcuda.cu -o qlearningcuda.o -std=c++11 && \
   g++ -c qlearning_openMP.cpp -o qlearning_openMP.o -fopenmp -std=c++11 && \
   g++ -c qlearning.cpp -o qlearning.o -std=c++11 && \
   g++ -c main.cpp -o main.o -std=c++11 && \
   g++ main.o qlearning.o qlearning_openMP.o qlearningcuda.o -o main -fopenmp -L/usr/local/cuda/lib64 -lcuda -lcudart -lcurand
   ```

3. Run the compiled program:
   ```bash
   ./main
   ```

4. Check the terminal for output and the `results` folder for saved data.

## Results

The program outputs:
- Performance metrics (e.g., execution time, speedup, efficiency)
- Computation results from parallel algorithms

## Contributing

Contributions are welcome! Submit a pull request or open an issue.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
