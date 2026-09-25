import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from collections import deque

# --- 1. GENERADOR DE CAOS CAMBIANTE ---
def get_lorenz_stream(n_steps=20000, rho=28.0):
    state = np.array([1.0, 1.0, 1.0])
    dt = 0.01
    sigma, beta = 10.0, 8.0/3.0
    
    X, y = [], []
    for _ in range(n_steps):
        x_val, y_val, z_val = state
        dx = sigma * (y_val - x_val)
        dy = x_val * (rho - z_val) - y_val
        dz = x_val * y_val - beta * z_val
        
        state_next = state + np.array([dx, dy, dz]) * dt
        
        X.append(state)
        y.append(state_next[0]) # Predecir siguiente X
        
        state = state_next
    return np.array(X), np.array(y)

# --- 2. LOS COMPETIDORES ---

# A. MLP (Deep Learning con Online Learning)
mlp = MLPRegressor(hidden_layer_sizes=(100, 100), learning_rate_init=0.01, random_state=42)

# B. VNN ADAPTATIVA (Tu Modelo Ágil)
class AdaptiveVNN:
    def __init__(self, n_neurons=300, weights=[0.29, 0.01, 0.15, 0.55]):
        self.n_neurons = n_neurons
        self.weights = np.array(weights)
        self.centers = None; self.values = None; self.radii = None
        self.counts = None

    def fit_batch(self, X, y):
        # Inicialización fuerte (K-Means)
        kmeans = KMeans(n_clusters=self.n_neurons, n_init=1).fit(X)
        self.centers = kmeans.cluster_centers_
        self.values = np.zeros(self.n_neurons)
        self.counts = np.ones(self.n_neurons)
        
        # Radios heurísticos
        from sklearn.metrics.pairwise import euclidean_distances
        dists = euclidean_distances(self.centers)
        np.fill_diagonal(dists, np.inf)
        self.radii = np.min(dists, axis=1) * 2.0 # Cobertura amplia
        
        labels = kmeans.predict(X)
        for i in range(self.n_neurons):
            idxs = np.where(labels==i)[0]
            if len(idxs) > 0: self.values[i] = np.mean(y[idxs])

    def predict_one(self, x):
        # Inferencia para un solo punto (rápida)
        diff = np.abs(x - self.centers)
        d_euc = np.linalg.norm(diff, axis=1)
        
        # Optimización: Solo calcular activaciones cercanas
        mask = d_euc < (self.radii * 1.5)
        if not np.any(mask): return 0.0 # Nadie cerca
        
        # Cálculo solo en subset activo
        active_indices = np.where(mask)[0]
        
        # Recalcular distancias solo para activos
        diff_act = diff[active_indices]
        r_act = self.radii[active_indices]
        d_euc_act = d_euc[active_indices]
        d_che_act = np.max(diff_act, axis=1)
        d_man_act = np.sum(diff_act, axis=1)
        
        act_euc = np.exp(-(d_euc_act/r_act)**2)
        act_che = np.exp(-(d_che_act/r_act)**2)
        act_man = np.exp(-(d_man_act/(r_act*2))**2)
        act_con = np.maximum(0, 1 - (d_euc_act/r_act))
        
        combined = (self.weights[0]*act_euc + self.weights[1]*act_che + 
                    self.weights[2]*act_man + self.weights[3]*act_con)
        
        w_final = combined ** 4
        denom = np.sum(w_final) + 1e-9
        
        return np.sum(w_final * self.values[active_indices]) / denom

    def update_online(self, x, target):
        # Plasticidad: Ajustar la neurona ganadora
        dists = np.linalg.norm(self.centers - x, axis=1)
        winner_idx = np.argmin(dists)
        
        # Solo actualizamos si está razonablemente cerca
        if dists[winner_idx] < self.radii[winner_idx] * 2.0:
            # Learning rate decay (1/n) para estabilidad
            lr = 0.5 # Tasa agresiva para adaptación rápida
            self.values[winner_idx] += lr * (target - self.values[winner_idx])
            
            # Opcional: Mover el centro hacia el nuevo dato (Tracking)
            self.centers[winner_idx] += 0.05 * (x - self.centers[winner_idx])

# --- 3. EL EXPERIMENTO ---

# Fase 1: Entrenamiento Base (Rho=28)
print("1. Entrenando en Régimen A (Rho=28)...")
X_base, y_base = get_lorenz_stream(2000, rho=28.0)
scaler = StandardScaler().fit(X_base)
X_base = scaler.transform(X_base)

mlp.fit(X_base, y_base) # Entrenamiento Batch
vnn = AdaptiveVNN(n_neurons=200)
vnn.fit_batch(X_base, y_base)

# Fase 2: EL CAMBIO DE CAOS (Rho=45)
print("2. ¡CAMBIO DE FÍSICA! Entrando en Régimen B (Rho=45)...")
X_new, y_new = get_lorenz_stream(1000, rho=45.0)
X_new = scaler.transform(X_new) # Usamos el mismo scaler (simulamos sensor calibrado)

errors_mlp = []
errors_vnn = []

print("3. Ejecutando Adaptación Online (Punto a Punto)...")
# Bucle de Streaming
for i in range(len(X_new)):
    x_curr = X_new[i]
    y_true = y_new[i]
    
    # A. Predecir (Antes de ver la respuesta)
    pred_mlp = mlp.predict([x_curr])[0]
    pred_vnn = vnn.predict_one(x_curr)
    
    # Guardar error
    errors_mlp.append(abs(y_true - pred_mlp))
    errors_vnn.append(abs(y_true - pred_vnn))
    
    # B. Aprender (Después de ver el error)
    # MLP partial fit (muy costoso para 1 sample, pero lo intentamos)
    mlp.partial_fit([x_curr], [y_true])
    
    # VNN update (muy barato)
    vnn.update_online(x_curr, y_true)

# --- 4. RESULTADOS ---
avg_err_mlp = np.mean(errors_mlp)
avg_err_vnn = np.mean(errors_vnn)

print("\n" + "="*60)
print(f"{'MODELO':<10} | {'ERROR MEDIO (Adaptación)':<25} | {'ESTADO'}")
print("-" * 60)
print(f"{'MLP':<10} | {avg_err_mlp:.5f}                    | {'¿Lento?'}")
print(f"{'VNN':<10} | {avg_err_vnn:.5f}                    | {'¿Rápido?'}")
print("="*60)

# Gráfica de error acumulado (Curva de Aprendizaje)
plt.figure(figsize=(12, 5))
plt.plot(np.cumsum(errors_mlp), 'r-', label=f'MLP (Total Err: {sum(errors_mlp):.1f})')
plt.plot(np.cumsum(errors_vnn), 'g-', linewidth=2, label=f'VNN (Total Err: {sum(errors_vnn):.1f})')
plt.title("Adaptación Online ante Cambio de Régimen (Chaos Shift)")
plt.xlabel("Tiempo (Steps)")
plt.ylabel("Error Acumulado")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()