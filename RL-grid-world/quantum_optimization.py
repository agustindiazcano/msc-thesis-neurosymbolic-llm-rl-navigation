import json
import numpy as np
import matplotlib.pyplot as plt
import dimod # Librería de simulación cuántica local

# --- 1. CARGAR TU FÓRMULA GANADORA ---
# Si no tienes el archivo, usamos los valores hardcodeados de tu último éxito
weights = {
    "euclid_weight": 0.4217, 
    "chebyshev_weight": 0.0230, 
    "manhattan_weight": 0.1006, 
    "conic_weight": 0.4547
}
print(f"Fórmula Cargada: Conic {weights['conic_weight']:.0%} | Euclid {weights['euclid_weight']:.0%}")

# --- 2. GENERAR CANDIDATOS (El Caos Inicial) ---
print("Generando 200 neuronas candidatas al azar...")
np.random.seed(42)
candidates = np.random.uniform(-1, 1, (200, 2)) # 2D para poder visualizarlo

# --- 3. CONSTRUIR QUBO (La Física VNN) ---
print("Construyendo Matriz de Energía...")
Q = {}
N = len(candidates)
TARGET_K = 20 # Queremos las 20 mejores

# Penalización por no tener K neuronas (Linear)
lagrange = 10.0
for i in range(N):
    Q[(i,i)] = -lagrange * (2 * TARGET_K - 1)

# Repulsión Híbrida (Quadratic)
def get_hybrid_dist(p1, p2):
    diff = np.abs(p1 - p2)
    d_euc = np.linalg.norm(diff)
    d_che = np.max(diff)
    d_man = np.sum(diff)
    # TU FORMULA:
    return (weights['euclid_weight']*d_euc + 
            weights['chebyshev_weight']*d_che + 
            weights['manhattan_weight']*d_man)

avg_dist = 0.4 # Radio de repulsión
for i in range(N):
    for j in range(i+1, N):
        d = get_hybrid_dist(candidates[i], candidates[j])
        if d < avg_dist:
            # Repulsión modulada por el peso Cónico (Tu hallazgo)
            force = (1.0 - d/avg_dist) * weights['conic_weight'] * 20.0
            Q[(i,j)] = force + (2 * lagrange)
        else:
            Q[(i,j)] = 2 * lagrange

# --- 4. RESOLVER (SIMULACIÓN CLÁSICA) ---
print("Simulando Recocido Cuántico en CPU...")
sampler = dimod.SimulatedAnnealingSampler()
response = sampler.sample_qubo(Q, num_reads=50) # 50 intentos

best_sample = response.first.sample
selected_indices = [k for k,v in best_sample.items() if v == 1]

print(f"Neuronas Seleccionadas: {len(selected_indices)}")

# --- 5. VISUALIZACIÓN ---
plt.figure(figsize=(10, 10))
# Dibujar las descartadas (Gris)
rejected = [i for i in range(N) if i not in selected_indices]
cand_r = candidates[rejected]
plt.scatter(cand_r[:,0], cand_r[:,1], c='lightgray', s=30, alpha=0.5, label='Candidatos (Descartados)')

# Dibujar las ELEGIDAS por tu fórmula (Rojo)
cand_s = candidates[selected_indices]
plt.scatter(cand_s[:,0], cand_s[:,1], c='red', s=150, edgecolors='black', label='Neuronas Óptimas (VNN)')

# Dibujar el radio de influencia (Geometría)
for p in cand_s:
    circle = plt.Circle(p, 0.15, color='red', fill=False, linestyle='--', alpha=0.3)
    plt.gca().add_patch(circle)

plt.title(f"Optimización Topológica VNN (Simulación QUBO)\nFórmula: 45% Cónica / 42% Euclídea")
plt.legend()
plt.grid(True)
plt.show()