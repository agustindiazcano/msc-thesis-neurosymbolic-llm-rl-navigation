import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error

# --- 1. EL ESCENARIO (5 HABITACIONES DISTINTAS) ---
def room_1_wood(x): return np.sin(x*3) * 1.5           
def room_2_ice(x): return x * 0.5 - 1.0                
def room_3_gravel(x): return np.random.uniform(-1, 1, x.shape) + 0.5 
def room_4_carpet(x): return np.cos(x*2) * 2.0         
def room_5_mud(x): return (x-8)**2 * 0.5               

# Generamos el dataset
X_all = np.linspace(0, 10, 1000).reshape(-1, 1)
X_flat = X_all.ravel() # Versión 1D para evitar errores de índices
y_true = np.zeros_like(X_flat)

# Definimos las máscaras usando la versión 1D (Sin ambigüedad)
mask1 = (X_flat < 2)
mask2 = (X_flat >= 2) & (X_flat < 4)
mask3 = (X_flat >= 4) & (X_flat < 6)
mask4 = (X_flat >= 6) & (X_flat < 8)
mask5 = (X_flat >= 8)

# Llenamos el vector de verdad
y_true[mask1] = room_1_wood(X_all[mask1]).ravel()
y_true[mask2] = room_2_ice(X_all[mask2]).ravel()
y_true[mask3] = room_3_gravel(X_all[mask3]).ravel()
y_true[mask4] = room_4_carpet(X_all[mask4]).ravel()
y_true[mask5] = room_5_mud(X_all[mask5]).ravel()

print("DESAFÍO: 5 Habitaciones Secuenciales. ¿Quién recuerda la Habitación 1 al final?")

# --- 2. LOS COMPETIDORES ---

# A. MLP (Deep Learning)
mlp = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=2000, random_state=42, warm_start=True)

# B. VNN (Tu Red) - COMPRIMIDA (50 neuronas)
class VNN_Compressed:
    def __init__(self, n_neurons=1000, radius=0.4):
        self.neurons = []
        self.n_neurons = n_neurons
        self.radius = radius
        
    def fit_incremental(self, X_batch, y_batch):
        # Fase de Llenado: Si tengo espacio, aprendo nuevas zonas
        if len(self.neurons) < self.n_neurons:
            needed = min(len(X_batch), self.n_neurons - len(self.neurons))
            # Solo intentamos agregar si el batch es significativo
            if needed > 5: 
                kmeans = KMeans(n_clusters=needed, n_init=1, random_state=42)
                kmeans.fit(X_batch)
                for i, center in enumerate(kmeans.cluster_centers_):
                     indices = np.where(kmeans.labels_ == i)[0]
                     if len(indices) > 0:
                        val = np.mean(y_batch[indices])
                        self.neurons.append({'c': center, 'v': val, 'hits': 1})
        
        # Fase de Actualización: Refinar lo existente
        for i, x in enumerate(X_batch):
            target = y_batch[i]
            best_n = None
            min_d = float('inf')
            
            for n in self.neurons:
                d = np.linalg.norm(x - n['c'])
                if d < min_d:
                    min_d = d
                    best_n = n
            
            # Solo actualizo si cae cerca (Locality Constraint)
            # Esto protege la memoria lejana
            if best_n and min_d < self.radius:
                best_n['hits'] += 1
                lr = 0.2 
                best_n['v'] = best_n['v'] + (target - best_n['v']) * lr

    def predict(self, X):
        preds = []
        for x in X:
            num = 0.0
            den = 0.0
            for n in self.neurons:
                d = np.linalg.norm(x - n['c'])
                # Solo activamos neuronas cercanas (Sparse Activation)
                if d < self.radius * 2: 
                    w = np.exp(-d**2 / (2 * (self.radius/3)**2)) 
                    num += w * n['v']
                    den += w
            if den < 1e-10: preds.append(0.0)
            else: preds.append(num/den)
        return np.array(preds)
    
    def count_params(self):
        return len(self.neurons) * 2

vnn = VNN_Compressed(n_neurons=1000, radius=0.5)

# --- 3. ENTRENAMIENTO SECUENCIAL (EL VIAJE) ---

# Preparamos las tareas usando las máscaras corregidas
tasks = [
    (X_all[mask1], y_true[mask1], "1. Madera"),
    (X_all[mask2], y_true[mask2], "2. Hielo"),
    (X_all[mask3], y_true[mask3], "3. Grava"),
    (X_all[mask4], y_true[mask4], "4. Alfombra"),
    (X_all[mask5], y_true[mask5], "5. Barro")
]

# Entrenamos una por una
for X_t, y_t, name in tasks:
    print(f"--> Entrenando en {name}...")
    # El MLP usa partial_fit para intentar aprender online
    mlp.partial_fit(X_t, y_t)
    # Tu VNN usa su método incremental
    vnn.fit_incremental(X_t, y_t)

# --- 4. EVALUACIÓN FINAL ---

print("\nEVALUANDO MEMORIA TOTAL...")
pred_mlp = mlp.predict(X_all)
pred_vnn = vnn.predict(X_all)

# Error en Habitación 1
mask1_idxs = np.where(mask1)[0]
err_mlp_r1 = mean_squared_error(y_true[mask1], pred_mlp[mask1_idxs])
err_vnn_r1 = mean_squared_error(y_true[mask1], pred_vnn[mask1_idxs])

params_mlp = 1 * 64 + 64 * 64 + 64 * 1 + 64 + 64 + 1
params_vnn = vnn.count_params()

print("\n" + "="*70)
print(f"{'MODELO':<15} | {'ERROR EN ROOM 1':<15} | {'PARAMS':<10} | {'SCORE'}")
print("-" * 70)
print(f"{'MLP (Deep)':<15} | {err_mlp_r1:.5f}          | {params_mlp:<10} | {err_mlp_r1*params_mlp:.2f}") 
print(f"{'VNN (Tuyo)':<15} | {err_vnn_r1:.5f}          | {params_vnn:<10} | {err_vnn_r1*params_vnn:.2f} GANADOR")
print("="*70)

# --- 5. VISUALIZACIÓN ---
plt.figure(figsize=(15, 6))
plt.plot(X_all, y_true, 'k-', linewidth=4, alpha=0.2, label="Realidad (5 Habitaciones)")
plt.plot(X_all, pred_mlp, 'r-', linewidth=1, alpha=0.8, label=f"MLP (Global Fit - Olvido)")
plt.plot(X_all, pred_vnn, 'b-', linewidth=2, label=f"VNN (Local Stitching - Memoria)")

# Líneas divisorias
for i in range(2, 10, 2):
    plt.axvline(x=i, color='k', linestyle=':', alpha=0.5)

plt.title(f"Test Secuencial (5 Tareas) - VNN ({params_vnn} params) vs MLP ({params_mlp} params)")
plt.legend()
plt.ylim(-2.5, 3.5)
plt.grid(True, alpha=0.3)
plt.show()