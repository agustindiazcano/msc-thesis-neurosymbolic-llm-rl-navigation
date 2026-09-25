import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import time

# --- 1. GENERADOR DE CAOS (LORENZ ATTRACTOR) ---
def lorenz_system(current_state, t):
    x, y, z = current_state
    sigma = 10.0
    rho = 28.0
    beta = 8.0 / 3.0
    
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return [dx_dt, dy_dt, dz_dt]

# Generar datos de entrenamiento (La Historia)
t = np.linspace(0, 50, 2000)
initial_state = [1.0, 1.0, 1.0]
chaos_data = odeint(lorenz_system, initial_state, t)

# Preparar Dataset: Input(t) -> Target(t+1)
X = chaos_data[:-1] # Estado actual
y = chaos_data[1:]  # Estado futuro (Lo que hay que predecir)

# Separar Train (Pasado) y Test (Futuro desconocido)
split = int(len(X) * 0.7)
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

print(f"Dataset Caótico Generado: {len(X)} muestras 3D.")

# --- 2. LOS COMPETIDORES (V2 SUITE) ---

# A. DEEP LEARNING (MLP Regressor)
mlp = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)

# B. VOLUMETRIC LOGIC (All Variants from V2, adapted for 3D)
class VNN_VectorField:
    def __init__(self, model_type, n_neurons=100):
        self.type = model_type
        self.name = model_type
        self.neurons = [] 
        self.n_neurons = n_neurons
        self.base_r = 3.0 # Radio base para el caos de Lorenz
        
    def _star_activation(self, dx, dy, dz, points, amp, r):
        # Activación Estelar generalizada a 3D (Cilindrica/Esférica simple)
        # Usamos proyección XY para los 'spikes' por simplicidad visual
        rad_xy = np.sqrt(dx**2 + dy**2)
        total_dist = np.sqrt(dx**2 + dy**2 + dz**2)
        theta = np.arctan2(dy, dx)
        
        # Modulamos el radio efectivo
        r_eff = r * (1.0 - amp * np.abs(np.sin(points/2 * theta)))
        
        return 1.0 if total_dist < r_eff else 0.0

    def _get_activation(self, n, x):
        dx = x[0] - n['c'][0]
        dy = x[1] - n['c'][1]
        dz = x[2] - n['c'][2]
        
        act = 0.0
        
        # --- GEOMETRÍAS ---
        if "Diamond" in self.type:
            d = abs(dx) + abs(dy) + abs(dz) # L1 3D
            if d < n['r']: act = 1.0
            
        elif "Hard Sphere" in self.type:
            d2 = dx**2 + dy**2 + dz**2
            if d2 < n['r']**2: act = 1.0
            
        elif "Soft Fuzzy" in self.type:
            d2 = dx**2 + dy**2 + dz**2
            sigma = n['r'] / 1.5
            act = np.exp(-d2 / (2 * sigma**2))
            
        elif "Smart+Soft" in self.type:
             d2 = dx**2 + dy**2 + dz**2
             sigma = n['r'] / 1.5
             act = np.exp(-d2 / (2 * sigma**2)) * n.get('confidence', 1.0)

        elif "Smart" in self.type:
            d = np.sqrt(dx**2 + dy**2 + dz**2)
            if d < n['r']: act = n.get('confidence', 1.0)
            
        elif "Star" in self.type or "Mine" in self.type:
            # Proyección pseudo-3D
            points = 24 if "Mine" in self.type else (14 if "14" in self.type else 8)
            amp = 0.5 if "Mine" in self.type else 0.3
            act = self._star_activation(dx, dy, dz, points, amp, n['r'])
            
        return act

    def fit(self, X, y):
        # 1. K-Means Inicialización (Distribución de neuronas)
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=self.n_neurons, n_init=10, random_state=42)
        kmeans.fit(X)
        
        # 2. Aprendizaje de Vectores Locales
        for center in kmeans.cluster_centers_:
            dists = np.linalg.norm(X - center, axis=1)
            radius = self.base_r
            
            # Puntos cercanos (Contexto Local)
            nearby_indices = np.where(dists < radius)[0]
            
            if len(nearby_indices) > 0:
                # Vector promedio local
                local_deltas = y[nearby_indices] - X[nearby_indices]
                velocity_vector = np.mean(local_deltas, axis=0)
                
                # Cálculo de confianza (Coherencia del flujo)
                # Si todos los vectores apuntan igual, alta confianza. Si hay turbulencia, baja.
                coherence = np.linalg.norm(velocity_vector) / (np.mean(np.linalg.norm(local_deltas, axis=1)) + 1e-6)
                confidence = 0.5 + coherence # Base 0.5 + Coherencia
            else:
                velocity_vector = np.zeros(3)
                confidence = 0.1
                
            self.neurons.append({
                'c': center,
                'v': velocity_vector,
                'r': radius,
                'confidence': confidence
            })

    def predict(self, X):
        predictions = []
        for x in X:
            active_vectors = []
            total_weight = 0
            
            for n in self.neurons:
                w = self._get_activation(n, x)
                if w > 0:
                    # Decay por distancia real 3D
                    dist = np.linalg.norm(x - n['c'])
                    w *= (1.0 / (dist + 0.1))
                    
                    active_vectors.append(n['v'] * w)
                    total_weight += w
            
            if total_weight > 0:
                avg_velocity = np.sum(active_vectors, axis=0) / total_weight
                predictions.append(x + avg_velocity)
            else:
                predictions.append(x) # Inercia cero si no sabe
                
        return np.array(predictions)

# --- 3. LA BATALLA ---

models = [
    ("MLP (Deep)", mlp),
    ("Hard Sphere", VNN_VectorField("Hard Sphere (L2)")),
    ("Soft Fuzzy", VNN_VectorField("Soft Fuzzy")),
    ("Smart+Soft", VNN_VectorField("Smart+Soft")),
    ("Marine Mine", VNN_VectorField("Marine Mine")),
    ("Diamond", VNN_VectorField("Diamond"))
]

results = []

print("\n--- INICIANDO COMPETENCIA ---")
for name, model in models:
    t0 = time.time()
    
    # Entrenar
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    
    # Predecir
    pred = model.predict(X_test)
    
    # Evaluar
    mse = mean_squared_error(y_test, pred)
    results.append({'name': name, 'mse': mse, 'time': train_time, 'pred': pred})
    
    print(f"{name:<15} | MSE: {mse:.5f} | Time: {train_time:.3f}s")

# --- 4. VISUALIZACIÓN 3D ---
fig = plt.figure(figsize=(18, 10))

# Plot Realidad
ax_real = fig.add_subplot(2, 4, 1, projection='3d')
ax_real.plot(y_test[:,0], y_test[:,1], y_test[:,2], 'k-', alpha=0.4, label='Realidad')
ax_real.set_title("LORENZ ATTRACTOR (Real)")

# Plots Modelos
for i, res in enumerate(results):
    ax = fig.add_subplot(2, 4, i + 2, projection='3d')
    pred = res['pred']
    
    # Realidad tenue de fondo
    ax.plot(y_test[:,0], y_test[:,1], y_test[:,2], 'k-', alpha=0.1)
    
    # Predicción del modelo
    color = 'r' if "MLP" in res['name'] else 'b'
    if "Smart+Soft" in res['name']: color = 'g' # Destacar al ganador
    
    ax.plot(pred[:,0], pred[:,1], pred[:,2], linestyle='--', color=color, alpha=0.7)
    ax.set_title(f"{res['name']}\nMSE: {res['mse']:.4f}")

plt.tight_layout()
plt.show()