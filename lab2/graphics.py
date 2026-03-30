import matplotlib.pyplot as plt
import numpy as np

# Данные
matrix_sizes = [100, 200, 400, 800, 1200, 1600, 2000]
threads = [1, 2, 4, 8, 16]

times = [
    [0.9, 0.9, 0.7, 1.2, 2.9],    # 100x100
    [7, 4, 3, 3, 5],               # 200x200
    [61, 32, 22, 13, 14],          # 400x400
    [567, 264, 138, 87, 93],       # 800x800
    [1628, 817, 441, 304, 219],    # 1200x1200
    [3698, 1906, 1002, 639, 472],  # 1600x1600
    [7250, 3700, 1977, 1310, 1017] # 2000x2000
]

# Цвета
colors = plt.cm.viridis(np.linspace(0, 1, len(matrix_sizes)))

# ==================== ГРАФИК ВРЕМЕНИ ====================
plt.figure(figsize=(14, 9))

for i, (size, time_vals, color) in enumerate(zip(matrix_sizes, times, colors)):
    # Построение линии
    plt.plot(threads, time_vals, marker='o', linewidth=2, markersize=8, 
             label=f'Матрица {size}x{size}', color=color)
    
    # Добавление значений над точками
    for x, y in zip(threads, time_vals):
        if y < 10:
            offset = y * 1.25
        elif y < 100:
            offset = y * 1.1
        else:
            offset = y * 1.05
            
        plt.text(x, offset, str(y), ha='center', va='bottom', 
                fontsize=9, fontweight='bold', color=color,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor='none'))

plt.xlabel('Количество потоков', fontsize=12, fontweight='bold')
plt.ylabel('Время перемножения (мс)', fontsize=12, fontweight='bold')
plt.title('Зависимость времени перемножения матриц от количества потоков', fontsize=14, fontweight='bold')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
plt.grid(True, alpha=0.3, linestyle='--')
plt.xticks(threads)
plt.yscale('log')
plt.xscale('log', base=2)
plt.tight_layout()
plt.show()

# ==================== ГРАФИК УСКОРЕНИЯ ====================
plt.figure(figsize=(14, 9))

for i, (size, time_vals, color) in enumerate(zip(matrix_sizes, times, colors)):
    # Вычисление ускорения
    speedup = [time_vals[0] / t for t in time_vals]
    
    # Построение линии
    plt.plot(threads, speedup, marker='s', linewidth=2, markersize=8, 
             label=f'Матрица {size}x{size}', color=color)
    
    # Добавление значений ускорения
    for x, y in zip(threads, speedup):
        offset = y * 1.05
        plt.text(x, offset, f'{y:.2f}', ha='center', va='bottom', 
                fontsize=9, fontweight='bold', color=color,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor='none'))

# Идеальное ускорение
plt.plot([1, 16], [1, 16], 'k--', alpha=0.5, label='Идеальное ускорение', linewidth=1.5)

plt.xlabel('Количество потоков', fontsize=12, fontweight='bold')
plt.ylabel('Ускорение (относительно 1 потока)', fontsize=12, fontweight='bold')
plt.title('Ускорение перемножения матриц при увеличении количества потоков', fontsize=14, fontweight='bold')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
plt.grid(True, alpha=0.3, linestyle='--')
plt.xticks(threads)
plt.xscale('log', base=2)
plt.tight_layout()
plt.show()