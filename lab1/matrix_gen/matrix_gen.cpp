#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <string>

using namespace std;

int main() {
#if _MSC_VER
    system("chcp 65001 > nul");
#endif
    ios_base::sync_with_stdio(false);
    const int SIZE = 10;

    vector<vector<int>> A(SIZE, vector<int>(SIZE));
    vector<vector<int>> B(SIZE, vector<int>(SIZE));

    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dist(0, 10);

    // Заполняем матрицы
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            A[i][j] = dist(gen);
            B[i][j] = dist(gen);
        }
    }

    // Используем SRC_DIR из CMake для пути
    ofstream foutA(string(SRC_DIR) + "/Matrix_A.txt");
    if (!foutA) {
        cerr << "Не удалось открыть файл Matrix_A.txt для записи!" << endl;
        return 1;
    }
    for (const auto& row : A) {
        for (int val : row)
            foutA << val << " ";
        foutA << endl;
    }
    foutA.close();

    ofstream foutB(string(SRC_DIR) + "/Matrix_B.txt");
    if (!foutB) {
        cerr << "Не удалось открыть файл Matrix_B.txt для записи!" << endl;
        return 1;
    }
    for (const auto& row : B) {
        for (int val : row)
            foutB << val << " ";
        foutB << endl;
    }
    foutB.close();

    cout << "Матрицы успешно сохранены в папке исходников!" << endl;
    return 0;
}