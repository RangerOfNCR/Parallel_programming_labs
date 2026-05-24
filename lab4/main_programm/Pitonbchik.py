import numpy as np
from pathlib import Path

BASE = Path(__file__).parent  # Папка, где лежит скрипт

A = np.loadtxt(BASE / "Matrix_A.txt", dtype=int)
B = np.loadtxt(BASE / "Matrix_B.txt", dtype=int)
C = np.loadtxt(BASE / "Matrix_C.txt", dtype=int)

# Перемножение с помощью numpy
C_ref = np.dot(A, B)

# Проверка совпадения
if np.array_equal(C, C_ref):
    print("Верификация пройдена: результаты совпадают")
else:
    print("Ошибка: результаты НЕ совпадают")
    # Можно вывести различия
    diff = C - C_ref
    print("Различия:\n", diff)