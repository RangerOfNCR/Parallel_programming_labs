#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <chrono>
#include <omp.h>

using namespace std;
using namespace std::chrono;

// Функция чтения матрицы из файла
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

// Функция записи матрицы в файл
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

// Функция умножения матриц и подсчета объема задачи
vector<vector<int>> multiplyMatrices(const vector<vector<int>>& A,const vector<vector<int>>& B, long long& operations) {
    int SIZE = A.size();
    vector<vector<int>> C(SIZE, vector<int>(SIZE, 0));
    operations = 0;

    // Убираем reduction и операции из цикла
    #pragma omp parallel for
    for (int i = 0; i < SIZE; i++) {
        for (int k = 0; k < SIZE; k++) {
            int aik = A[i][k];
            for (int j = 0; j < SIZE; j++) {
                C[i][j] += aik * B[k][j];
            }
        }
    }
    operations = 2LL * SIZE * SIZE * SIZE;
    return C;
}

// Функция вывода матрицы
void printMatrix(const vector<vector<int>>& mat) {
    for (const auto& row : mat) {
        for (const auto& val : row)
            cout << val << " ";
        cout << endl;
    }
}

//Всего 16 потоков, 8 ядер

int main() {
#if _MSC_VER
    system("chcp 65001 > nul");
#endif
    ios_base::sync_with_stdio(false);

    const int SIZE = 1600;
    vector<vector<int>> A(SIZE, vector<int>(SIZE));
    vector<vector<int>> B(SIZE, vector<int>(SIZE));

    if (!readMatrix(string(SRC_DIR) + "/Matrix_A.txt", A) ||
        !readMatrix(string(SRC_DIR) + "/Matrix_B.txt", B)) {
        return 1;
    }

    vector<int> threadCounts = { 1, 2, 4, 8, 16 };

    for (int threads : threadCounts) {
        omp_set_num_threads(threads);

        long long operations = 0;

        auto start = high_resolution_clock::now();
        vector<vector<int>> C = multiplyMatrices(A, B, operations);
        auto end = high_resolution_clock::now();

        int duration = duration_cast<nanoseconds>(end - start).count() / 1000000;

        cout << "Потоки: " << threads << endl;
        cout << "Время: " << duration << " мс" << endl;
        cout << "Операции: " << operations << endl;
        cout << "------------------------" << endl;

        // Запишем только один раз (например для максимального числа потоков)
        if (threads == threadCounts.back()) {
            if (!writeMatrix(string(SRC_DIR) + "/Matrix_C.txt", C)) {
                return 1;
            }
        }
    }

    return 0;
}