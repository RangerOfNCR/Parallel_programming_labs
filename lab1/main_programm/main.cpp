#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <chrono>

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
vector<vector<int>> multiplyMatrices(const vector<vector<int>>& A, const vector<vector<int>>& B, long long& operations) {
    int SIZE = A.size();
    vector<vector<int>> C(SIZE, vector<int>(SIZE, 0));
    operations = 0;
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            for (int k = 0; k < SIZE; k++) {
                C[i][j] += A[i][k] * B[k][j];
                operations += 2; // одно умножение + одно сложение
            }
        }
    }
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

int main() {
#if _MSC_VER
    system("chcp 65001 > nul");
#endif
    ios_base::sync_with_stdio(false);

    const int SIZE = 10;
    vector<vector<int>> A(SIZE, vector<int>(SIZE));
    vector<vector<int>> B(SIZE, vector<int>(SIZE));

    // Чтение матриц
    if (!readMatrix(string(SRC_DIR) + "/Matrix_A.txt", A) ||
        !readMatrix(string(SRC_DIR) + "/Matrix_B.txt", B)) {
        return 1;
    }

    // Измерение времени умножения
    auto start = high_resolution_clock::now();
    long long operations = 0;
    vector<vector<int>> C = multiplyMatrices(A, B, operations);
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start).count();

    // Вывод результата
    cout << "Matrix C:\n";
    printMatrix(C);

    if (!writeMatrix(string(SRC_DIR) + "/Matrix_C.txt", C)) {
        return 1;
    }

    cout << "Перемноженная матрица сохранена в файл" << endl;
    cout << "Время выполнения: " << duration << " мс" << endl;
    cout << "Объем задачи (кол-во операций): " << operations << endl;

    return 0;
}