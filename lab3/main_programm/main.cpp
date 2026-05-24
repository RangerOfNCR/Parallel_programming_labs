#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <chrono>
#include <mpi.h>

#ifndef SRC_DIR
#define SRC_DIR "."
#endif

using namespace std;
using namespace std::chrono;

// --- Вспомогательные функции ---

bool readMatrix(const string& path, vector<vector<int>>& mat) {
    ifstream fin(path);
    if (!fin) {
        cerr << "Не удалось открыть файл " << path << " для чтения!" << endl;
        return false;
    }
    for (auto& row : mat)
        for (auto& val : row)
            fin >> val;
    fin.close();
    return true;
}

bool writeMatrix(const string& path, const vector<vector<int>>& mat) {
    ofstream fout(path);
    if (!fout) {
        cerr << "Не удалось открыть файл " << path << " для записи!" << endl;
        return false;
    }
    for (const auto& row : mat) {
        for (const auto& val : row)
            fout << val << " ";
        fout << endl;
    }
    fout.close();
    return true;
}

vector<int> flatten(const vector<vector<int>>& mat) {
    int n = mat.size();
    vector<int> flat(n * n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            flat[i * n + j] = mat[i][j];
    return flat;
}

vector<vector<int>> unflatten(const vector<int>& flat, int n) {
    vector<vector<int>> mat(n, vector<int>(n));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            mat[i][j] = flat[i * n + j];
    return mat;
}

// --- Основная функция MPI ---

vector<vector<int>> multiplyMatrices_MPI(const vector<vector<int>>& A, const vector<vector<int>>& B, long long& total_operations, int rank, int size) {
    int N = A.size(); 

    vector<int> flatA, flatB;
    if (rank == 0) {
        flatA = flatten(A);
        flatB = flatten(B);
    }

    int rows_per_proc = N / size;
    int remainder = N % size;
    int local_rows = rows_per_proc + (rank < remainder ? 1 : 0);

    vector<int> localA(local_rows * N);
    vector<int> sendcounts(size);
    vector<int> displs(size);
    int offset = 0;

    for (int i = 0; i < size; i++) {
        int rows = rows_per_proc + (i < remainder ? 1 : 0);
        sendcounts[i] = rows * N;
        displs[i] = offset;
        offset += sendcounts[i];
    }

    // Раздаем части матрицы A всем процессам
    MPI_Scatterv(
        rank == 0 ? flatA.data() : nullptr,
        sendcounts.data(),
        displs.data(),
        MPI_INT,
        localA.data(),
        local_rows * N,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    
    if (rank != 0) flatB.resize(N * N);
    MPI_Bcast(flatB.data(), N * N, MPI_INT, 0, MPI_COMM_WORLD);

    vector<int> localC(local_rows * N, 0);
    long long local_ops = 0; 

    for (int i = 0; i < local_rows; i++) {
        for (int j = 0; j < N; j++) {
            int sum = 0;
            for (int k = 0; k < N; k++) {
                sum += localA[i * N + k] * flatB[k * N + j];
                local_ops += 2; 
            }
            localC[i * N + j] = sum;
        }
    }

    
    vector<int> flatC;
    if (rank == 0) flatC.resize(N * N);

    MPI_Gatherv(
        localC.data(),
        local_rows * N,
        MPI_INT,
        rank == 0 ? flatC.data() : nullptr,
        sendcounts.data(),
        displs.data(),
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    // Суммируем количество операций со всех процессов на нулевой
    MPI_Reduce(&local_ops, &total_operations, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) return unflatten(flatC, N);
    else return {};
}

// --- Main ---

int main(int argc, char** argv)
{
#if _MSC_VER
    system("chcp 65001 > nul");
#endif
    ios_base::sync_with_stdio(false);

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int SIZE = 2000;

    // Инициализируем A и B на ВСЕХ процессах, чтобы A.size() везде равнялось 200.
    // Это важно для правильного подсчета N внутри multiplyMatrices_MPI.
    vector<vector<int>> A(SIZE, vector<int>(SIZE));
    vector<vector<int>> B(SIZE, vector<int>(SIZE));

    int read_success = 1;
    if (rank == 0) {
        if (!readMatrix(string(SRC_DIR) + "/Matrix_A.txt", A) ||
            !readMatrix(string(SRC_DIR) + "/Matrix_B.txt", B)) {
            read_success = 0;
        }
    }

    // Если файлы не прочитались, завершаем все процессы, чтобы избежать зависания
    MPI_Bcast(&read_success, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (read_success == 0) {
        MPI_Finalize();
        return 1;
    }

    auto start = high_resolution_clock::now();
    long long operations = 0;

    
    vector<vector<int>> C = multiplyMatrices_MPI(A, B, operations, rank, size);

    auto end = high_resolution_clock::now();
    float duration = duration_cast<nanoseconds>(end - start).count() / 1000000;

    if (rank == 0) {
        if (!writeMatrix(string(SRC_DIR) + "/Matrix_C.txt", C)) {
            MPI_Finalize();
            return 1;
        }
        cout << "Перемноженная матрица сохранена в файл" << endl;
        cout << "Время выполнения: " << duration << " мс" << endl;
        cout << "Объем задачи (кол-во операций): " << operations << endl;
    }

    MPI_Finalize(); 
    return 0;
}