import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Данные из таблицы
data = {
    'Размер матрицы': [100, 100, 100, 100, 100,
                       200, 200, 200, 200, 200,
                       400, 400, 400, 400, 400,
                       800, 800, 800, 800, 800,
                       1200, 1200, 1200, 1200, 1200,
                       1600, 1600, 1600, 1600, 1600,
                       2000, 2000, 2000, 2000, 2000],
    'Количество процессов': [1, 2, 4, 8, 16] * 7,
    'Время перемножения (мс)': [
        0.69, 0.63, 0.75, 1.3, 1.5,
        4.8, 2.6, 1.8, 2.5, 3.3,
        46, 24, 13, 14, 10,
        397, 204, 104, 79, 63,
        1446, 761, 464, 346, 296,
        4237, 2257, 1243, 886, 822,
        8939, 4812, 2548, 1673, 1590
    ]
}

df = pd.DataFrame(data)
sizes = df['Размер матрицы'].unique()

# 1. График: время от количества процессов
plt.figure(figsize=(10, 6))
for size in sizes:
    subset = df[df['Размер матрицы'] == size]
    plt.plot(subset['Количество процессов'], subset['Время перемножения (мс)'], 
             marker='o', label=f'Размер {size}')
    # Подписи значений над точками
    for x, y in zip(subset['Количество процессов'], subset['Время перемножения (мс)']):
        plt.text(x, y + max(0.05 * y, 0.5), f'{y:.1f}', 
                 ha='center', va='bottom', fontsize=8)

plt.xlabel('Количество процессов')
plt.ylabel('Время перемножения (мс)')
plt.title('Зависимость времени перемножения от количества процессов\n(для разных размеров матриц)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks([1, 2, 4, 8, 16])
plt.tight_layout()
plt.show()

# 2. График: ускорение (Speedup) = T1 / Tp
plt.figure(figsize=(10, 6))
for size in sizes:
    subset = df[df['Размер матрицы'] == size]
    t1 = subset[subset['Количество процессов'] == 1]['Время перемножения (мс)'].values[0]
    speedup = t1 / subset['Время перемножения (мс)']
    plt.plot(subset['Количество процессов'], speedup, 
             marker='s', label=f'Размер {size}')
    # Подписи значений над точками
    for x, y in zip(subset['Количество процессов'], speedup):
        plt.text(x, y + 0.05, f'{y:.2f}', 
                 ha='center', va='bottom', fontsize=8)

# Линия идеального ускорения
procs = [1, 2, 4, 8, 16]
plt.plot(procs, procs, 'k--', label='Идеальное ускорение', alpha=0.5)

plt.xlabel('Количество процессов')
plt.ylabel('Ускорение (Speedup)')
plt.title('Ускорение параллельного умножения матриц')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks([1, 2, 4, 8, 16])
plt.tight_layout()
plt.show()

