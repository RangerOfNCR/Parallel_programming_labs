#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <chrono>
#include <iomanip>    
#include <algorithm>  
#include <cuda_runtime.h>

using namespace std;
using namespace std::chrono;

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
             << cudaGetErrorString(err) << endl; \
        exit(1); \
    } \
}

// Шаблонизированное ядро для поддержки динамических размеров блоков при компиляции
template <int BLOCK_SIZE>
__global__ void tiledMultiplyKernel(int* A, int* B, int* C, int SIZE) {
    
    __shared__ int tileA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int tileB[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    int sum = 0;

    for (int t = 0; t < (SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE; t++) {
        int tiledCol = t * BLOCK_SIZE + threadIdx.x;
        int tiledRow = t * BLOCK_SIZE + threadIdx.y;

        if (row < SIZE && tiledCol < SIZE)
            tileA[threadIdx.y][threadIdx.x] = A[row * SIZE + tiledCol];
        else
            tileA[threadIdx.y][threadIdx.x] = 0;

        if (tiledRow < SIZE && col < SIZE)
            tileB[threadIdx.y][threadIdx.x] = B[tiledRow * SIZE + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0;

        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; k++) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < SIZE && col < SIZE) {
        C[row * SIZE + col] = sum;
    }
}

bool readMatrix(const string& path, vector<int>& mat, int SIZE) {
    ifstream fin(path);
    if (!fin) {
        cerr << "Ошибка открытия файла " << path << "!" << endl;
        return false;
    }
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            fin >> mat[i * SIZE + j];
        }
    }
    fin.close();
    return true;
}

bool writeMatrix(const string& path, const vector<int>& mat, int SIZE) {
    ofstream fout(path);
    if (!fout) {
        cerr << "Ошибка записи файла " << path << "!" << endl;
        return false;
    }
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            fout << mat[i * SIZE + j] << " ";
        }
        fout << endl;
    }
    fout.close();
    return true;
}

struct BenchmarkResult {
    string description;
    int blockX;
    int blockY;
    double avgTimeMs;
    double minTimeMs;
    double maxTimeMs;
    double gflops;
    int totalBlocks;
};

int main() {
#if _MSC_VER
    system("chcp 65001 > nul");
#endif
    ios_base::sync_with_stdio(false);

    const int SIZE = 100;
    size_t bytes = SIZE * SIZE * sizeof(int);

    struct BlockConfig {
        int blockSize;
        string description;
    };

    
    vector<BlockConfig> configs = {
        {4,  "4x4 (16 threads/block)"},
        {8,  "8x8 (64 threads/block)"},
        {16, "16x16 (256 threads/block)"},
        {32, "32x32 (1024 threads/block)"}
    };

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));

    cout << "=== MATRIX MULTIPLICATION BENCHMARK ===" << endl;
    cout << "GPU: " << prop.name << endl;
    cout << "Matrix size: " << SIZE << "x" << SIZE << endl;
    cout << "=========================================" << endl << endl;

    vector<int> A(SIZE * SIZE);
    vector<int> B(SIZE * SIZE);
    vector<int> C(SIZE * SIZE);
    vector<int> bestC(SIZE * SIZE);

    cout << "Loading matrices..." << endl;
#ifndef SRC_DIR
#define SRC_DIR "."
#endif
    if (!readMatrix(string(SRC_DIR) + "/Matrix_A.txt", A, SIZE) ||
        !readMatrix(string(SRC_DIR) + "/Matrix_B.txt", B, SIZE)) {
        return 1;
    }
    cout << "Matrices loaded successfully!" << endl << endl;

    int* d_A; int* d_B; int* d_C;
    CHECK_CUDA(cudaMalloc(&d_A, bytes));
    CHECK_CUDA(cudaMalloc(&d_B, bytes));
    CHECK_CUDA(cudaMalloc(&d_C, bytes));

    CHECK_CUDA(cudaMemcpy(d_A, A.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, B.data(), bytes, cudaMemcpyHostToDevice));

    vector<BenchmarkResult> results;
    const int WARMUP_ITER = 2;
    const int TEST_ITER = 5;
    double bestTimeTotal = 1e9; // Переменная для корректного отслеживания лучшего времени

    for (const auto& cfg : configs) {
        int threadsPerBlock = cfg.blockSize * cfg.blockSize;

        if (threadsPerBlock > prop.maxThreadsPerBlock) {
            cout << "Skipping " << cfg.description << " (exceeds max threads/block)" << endl;
            continue;
        }

        cout << "\nTesting: " << cfg.description << endl;

        dim3 threads(cfg.blockSize, cfg.blockSize);
        dim3 blocks((SIZE + cfg.blockSize - 1) / cfg.blockSize, (SIZE + cfg.blockSize - 1) / cfg.blockSize);

        // Прогрев
        for (int i = 0; i < WARMUP_ITER; i++) {
            if (cfg.blockSize == 4) tiledMultiplyKernel<4><<<blocks, threads>>>(d_A, d_B, d_C, SIZE);
            else if (cfg.blockSize == 8) tiledMultiplyKernel<8><<<blocks, threads>>>(d_A, d_B, d_C, SIZE);
            else if (cfg.blockSize == 16) tiledMultiplyKernel<16><<<blocks, threads>>>(d_A, d_B, d_C, SIZE);
            else if (cfg.blockSize == 32) tiledMultiplyKernel<32><<<blocks, threads>>>(d_A, d_B, d_C, SIZE);
            CHECK_CUDA(cudaDeviceSynchronize());
        }

        vector<double> times;
        times.reserve(TEST_ITER);

        for (int iter = 0; iter < TEST_ITER; iter++) {
            

            auto start = high_resolution_clock::now();

            if (cfg.blockSize == 4) tiledMultiplyKernel<4><<<blocks, threads>>>(d_A, d_B, d_C, SIZE);
            else if (cfg.blockSize == 8) tiledMultiplyKernel<8><<<blocks, threads>>>(d_A, d_B, d_C, SIZE);
            else if (cfg.blockSize == 16) tiledMultiplyKernel<16><<<blocks, threads>>>(d_A, d_B, d_C, SIZE);
            else if (cfg.blockSize == 32) tiledMultiplyKernel<32><<<blocks, threads>>>(d_A, d_B, d_C, SIZE);
            
            CHECK_CUDA(cudaDeviceSynchronize());

            auto end = high_resolution_clock::now();
            double timeMs = duration_cast<microseconds>(end - start).count() / 1000.0;
            times.push_back(timeMs);

            cout << "    Run " << (iter + 1) << ": " << fixed << setprecision(2) << timeMs << " ms" << endl;
        }

        double sum = 0.0;
        double minTime = times[0];
        double maxTime = times[0];
        for (double t : times) {
            sum += t;
            if (t < minTime) minTime = t;
            if (t > maxTime) maxTime = t;
        }

        double avgTime = sum / TEST_ITER;
        double gflops = (2.0 * SIZE * SIZE * SIZE / 1e9) / (avgTime / 1000.0);

        results.push_back({
            cfg.description, cfg.blockSize, cfg.blockSize,
            avgTime, minTime, maxTime, gflops, (int)(blocks.x * blocks.y)
        });

        
        if (avgTime < bestTimeTotal) {
            bestTimeTotal = avgTime;
            CHECK_CUDA(cudaMemcpy(bestC.data(), d_C, bytes, cudaMemcpyDeviceToHost));
        }

        cout << "  * Average time: " << fixed << setprecision(2) << avgTime << " ms" << endl;
    }

    // Вывод итоговой таблицы
    cout << "\n\n" << string(80, '=') << endl;
    cout << "BENCHMARK RESULTS SUMMARY" << endl;
    cout << string(80, '=') << endl;

    sort(results.begin(), results.end(), [](const BenchmarkResult& a, const BenchmarkResult& b) {
        return a.avgTimeMs < b.avgTimeMs;
    });

    cout << left << setw(30) << "Configuration"
         << setw(15) << "Block Size"
         << setw(15) << "Avg Time (ms)"
         << setw(15) << "GFLOPS"
         << setw(15) << "Blocks" << endl;
    cout << string(80, '-') << endl;

    for (size_t i = 0; i < results.size(); i++) {
        const auto& r = results[i];
        string blockSizeStr = to_string(r.blockX) + "x" + to_string(r.blockY);

        cout << left << setw(30) << r.description
             << setw(15) << blockSizeStr
             << setw(15) << fixed << setprecision(2) << r.avgTimeMs
             << setw(15) << fixed << setprecision(2) << r.gflops
             << setw(15) << r.totalBlocks;

        if (i == 0) cout << " \x1b[32m← FASTEST\x1b[0m"; 
        cout << endl;
    }
    cout << string(80, '=') << endl;

    cout << "\nSaving result matrix (using best configuration)..." << endl;
    writeMatrix(string(SRC_DIR) + "/Matrix_C.txt", bestC, SIZE);

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    return 0;
}