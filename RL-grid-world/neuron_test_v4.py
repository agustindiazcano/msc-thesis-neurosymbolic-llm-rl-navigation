import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from sklearn.neural_network import MLPRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
import time

# --- 1. FÍSICA CAMBIANTE ---
def lorenz_physics(state, t, sigma, rho, beta):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return [dx, dy, dz]

# Fase 1: Entrenamiento (Mundo Normal)
t_train = np.linspace(0, 40, 2000)
data_train = odeint(lorenz_physics, [1,1,1], t_train, args=(10.0, 28.0, 8.0/3.0))

# Fase 2: Test (Mundo Roto - "Broken Wing")
t_test = np.linspace(0, 20, 1000)
start_test = data_train[-1] 
data_test = odeint(lorenz_physics, start_test, t_test, args=(15.0, 20.0, 8.0/3.0)) 

X_train, y_train = data_train[:-1], data_train[1:]
X_test, y_test = data_test[:-1], data_test[1:]

print(f"Dataset: {len(X_train)} train, {len(X_test)} test (Broken Physics).")

# --- 2. COMPETIDORES ---

# A. MLP (Online Learning Wrapper)
class OnlineMLP:
    def __init__(self):
        self.name = "MLP (Deep)"
        self.model = MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=1, warm_start=True, random_state=42)
        self.buffer_X = []
        self.buffer_y = []
        
    def fit(self, X, y):
        self.model.max_iter = 500 # Fase inicial offline
        self.model.fit(X, y)
        self.model.max_iter = 1   # Modo online activado
        
    def predict(self, x):
        return self.model.predict([x])[0]
    
    def learn(self, x, target):
        self.buffer_X.append(x)
        self.buffer_y.append(target)
        # Re-entrena con ventana reciente (Last 50)
        window = 50
        if len(self.buffer_X) > window:
            X_batch = self.buffer_X[-window:]
            y_batch = self.buffer_y[-window:]
            self.model.partial_fit(X_batch, y_batch)

# B. VNN BASE (Para herencia)
class VNN_Base:
    def __init__(self, n_neurons=100):
        self.n_neurons = n_neurons
        self.neurons = []
        self.kmeans = None
        
    def fit(self, X, y):
        self.kmeans = KMeans(n_clusters=self.n_neurons, n_init=10, random_state=42)
        self.kmeans.fit(X)
        centers = self.kmeans.cluster_centers_
        
        for i, center in enumerate(centers):
            indices = np.where(self.kmeans.labels_ == i)[0]
            if len(indices) > 0:
                vectors = y[indices] - X[indices]
                v = np.mean(vectors, axis=0) # Vector promedio
                radius = np.mean(np.linalg.norm(X[indices] - center, axis=1)) * 1.5
            else:
                v = np.zeros(3); radius = 1.0
            
            self.neurons.append({'c': center, 'v': v, 'r': radius, 'conf': 1.0})
            
    def learn_step(self, x, target):
        # Update simple de Hebbian/Kalman local
        # Busca la neurona más cercana y actualiza su vector
        true_vector = target - x
        
        dists = np.linalg.norm([n['c'] for n in self.neurons] - x, axis=1)
        idx = np.argmin(dists)
        n = self.neurons[idx]
        
        if dists[idx] < n['r'] * 2:
            # Update local
            lr = 0.5 # Tasa de aprendizaje agresiva para adaptación rápida
            n['v'] = (1 - lr) * n['v'] + lr * true_vector
            n['c'] += 0.1 * (x - n['c']) # Drift espacial

# C. SMART TRACKER (Baseline Hard)
class SmartTracker(VNN_Base):
    def __init__(self, n_neurons=100):
        super().__init__(n_neurons)
        self.name = "Smart (Hard)"
        
    def predict(self, x):
        # Nearest Neighbor Hard
        dists = np.linalg.norm([n['c'] for n in self.neurons] - x, axis=1)
        idx = np.argmin(dists)
        n = self.neurons[idx]
        return x + n['v']

# D. SMART+SOFT TRACKER (Tu Campeón)
class SmartSoftTracker(VNN_Base):
    def __init__(self, n_neurons=100):
        super().__init__(n_neurons)
        self.name = "Smart+Soft"
        
    def predict(self, x):
        # Soft Voting (k-NN Gaussian)
        total_v = np.zeros(3)
        total_w = 0.0
        
        dists = np.linalg.norm([n['c'] for n in self.neurons] - x, axis=1)
        nearest_indices = np.argsort(dists)[:5] # Top 5 vecinos
        
        for idx in nearest_indices:
            n = self.neurons[idx]
            d = dists[idx]
            
            # Activación Gaussiana (Soft)
            if d < n['r'] * 3: 
                sigma = n['r'] / 1.5
                w = np.exp(-d**2 / (2 * sigma**2))
                total_v += n['v'] * w
                total_w += w
        
        if total_w > 0:
            final_v = total_v / total_w
        else:
            final_v = self.neurons[nearest_indices[0]]['v']
            
        return x + final_v

# --- 3. EJECUCIÓN ONLINE ---

models = [OnlineMLP(), SmartTracker(120), SmartSoftTracker(120)]

# Pre-entrenamiento (Mundo Normal)
print("=== FASE 1: ENTRENAMIENTO OFFLINE ===")
for m in models:
    print(f"Entrenando {m.name}...")
    m.fit(X_train, y_train)

# Test Online (Mundo Roto)
print("\n=== FASE 2: TEST ONLINE (ADAPTACIÓN) ===")
print(f"{'STEP':<5} | {'MLP Err':<10} | {'Smart Err':<10} | {'S+Soft Err':<10}")
print("-" * 50)

errors = {m.name: [] for m in models}
preds_log = {m.name: [] for m in models}

for t in range(len(X_test)):
    curr_x = X_test[t]
    true_y = y_test[t]
    
    step_errs = []
    
    for m in models:
        # 1. Predecir
        pred = m.predict(curr_x)
        preds_log[m.name].append(pred)
        
        # 2. Medir Error
        err = np.linalg.norm(pred - true_y)
        errors[m.name].append(err)
        step_errs.append(err)
        
        # 3. APRENDER (ONLINE)
        if hasattr(m, 'learn'): m.learn(curr_x, true_y)
        elif hasattr(m, 'learn_step'): m.learn_step(curr_x, true_y)
    
    if t % 100 == 0:
        print(f"{t:<5} | {step_errs[0]:.4f}     | {step_errs[1]:.4f}     | {step_errs[2]:.4f}")

# Calcular MSE Final
print("\n--- RESULTADOS FINALES (Online) ---")
for m in models:
    mean_err = np.mean(errors[m.name])
    print(f"{m.name:<12} | Mean Error: {mean_err:.5f}")

# --- 4. GRÁFICA ---
plt.figure(figsize=(15, 6))
# Solo graficamos la dimensión X para claridad
plt.plot(y_test[:, 0], 'k-', alpha=0.2, linewidth=3, label="Realidad (Broken)")

colores = {'MLP (Deep)': 'red', 'Smart (Hard)': 'orange', 'Smart+Soft': 'green'}

for m in models:
    p = np.array(preds_log[m.name])
    # Suavizado para la gráfica
    loss_smooth = np.convolve(errors[m.name], np.ones(20)/20, mode='valid')
    plt.plot(p[:, 0], linestyle='--', color=colores[m.name], label=f"{m.name}", alpha=0.8)

plt.title("Adaptación Online: Recuperación tras el Colapso Físico")
plt.legend()
plt.show()