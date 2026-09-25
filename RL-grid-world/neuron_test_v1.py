
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import time

# --- CONFIGURACIÓN ---
N_SAMPLES = 600
NOISE = 0.25
N_NEURONS = 16  # Neuronas totales (8 por clase aprox)

X, y = make_moons(n_samples=N_SAMPLES, noise=NOISE, random_state=42)

# ==========================================
# 1. BASELINE: MLP (Deep Learning Clásico)
# ==========================================
class BenchmarkMLP:
    def __init__(self):
        self.model = MLPClassifier(hidden_layer_sizes=(N_NEURONS,), max_iter=2000, random_state=42)
        self.type = "Baseline MLP (Deep)"
        
    def fit(self, X, y):
        self.model.fit(X, y)
        
    def predict(self, X):
        return self.model.predict(X)

# ==========================================
# 2. ARQUITECTURA: Volumetric Logic (Todas las variantes)
# ==========================================
class VolumetricNetwork:
    def __init__(self, model_type, base_radius=0.35):
        self.type = model_type
        self.base_radius = base_radius
        self.neurons = []
        
    def _star_activation(self, dx, dy, points, amp, r):
        # Cálculo polar para formas complejas
        rad = np.sqrt(dx**2 + dy**2)
        theta = np.arctan2(dy, dx)
        # R efectivo modulado por una onda senoidal sobre el ángulo
        r_eff = r * (1.0 - amp * np.abs(np.sin(points/2 * theta)))
        return 1.0 if rad < r_eff else 0.0

    def fit(self, X, y):
        self.neurons = []
        for target_class in [0, 1]:
            X_c = X[y == target_class] # Clase propia
            X_e = X[y != target_class] # Clase enemiga
            if len(X_c) == 0: continue
            
            # A. Inicialización (KMeans)
            n_c = N_NEURONS // 2
            kmeans = KMeans(n_clusters=n_c, n_init=10, random_state=42)
            kmeans.fit(X_c)
            centers = kmeans.cluster_centers_
            
            # B. Refinamiento por neurona
            for center in centers:
                radius = self.base_radius
                weight = 1.0
                
                # --- LÓGICA SMART (Solo para el modelo "Smart") ---
                if "Smart" in self.type:
                    # 1. KALMAN-STYLE (Centrado Dinámico)
                    for _ in range(5):
                        dists = np.linalg.norm(X_c - center, axis=1)
                        neighbors = X_c[dists < radius]
                        if len(neighbors) > 0:
                            center = 0.7 * center + 0.3 * np.mean(neighbors, axis=0)

                    # 2. LAGRANGE CONSTRAINT (No tocar enemigos)
                    dists_enemy = np.linalg.norm(X_e - center, axis=1)
                    if len(dists_enemy) > 0:
                        closest_enemy = np.min(dists_enemy)
                        # Margen de seguridad: radio debe ser menor a la distancia al enemigo
                        radius = min(closest_enemy * 0.9, 1.2) 
                    
                    # 3. BAYES WEIGHT (Densidad = Confianza)
                    dists_friend = np.linalg.norm(X_c - center, axis=1)
                    hits = np.sum(dists_friend < radius)
                    # Peso logarítmico basado en densidad
                    weight = np.log(1 + hits) + 0.5

                # Guardamos la neurona configurada
                self.neurons.append({'c': center, 'r': radius, 'lbl': target_class, 'w': weight})

    def predict(self, X):
        preds = []
        for x in X:
            s0, s1 = 0.0, 0.0
            
            for n in self.neurons:
                dx = x[0] - n['c'][0]
                dy = x[1] - n['c'][1]
                act = 0.0
                
                # --- SELECTOR DE GEOMETRÍA ---
                
                if "Diamond" in self.type:
                    # L1 Norm (Manhattan)
                    d = abs(dx) + abs(dy)
                    if d < n['r']: act = 1.0
                
                elif "Hard Sphere" in self.type:
                    # L2 Norm (Euclidiana cuadrada optimizada)
                    d2 = dx**2 + dy**2
                    if d2 < n['r']**2: act = 1.0
                    
                elif "Soft Fuzzy" in self.type:
                    # Gaussiana simple
                    d2 = dx**2 + dy**2
                    sigma = n['r'] / 1.5
                    act = np.exp(-d2 / (2 * sigma**2))
                    
                elif "Smart" in self.type:
                    # Híbrido: Hard limit (Lagrange) + Soft interior (Bayes/Fuzzy)
                    d = np.sqrt(dx**2 + dy**2)
                    if d < n['r']: 
                        # Dentro del radio seguro, activación ponderada por peso (Bayes)
                        act = n['w']
                
                elif "Star 8" in self.type:
                    act = self._star_activation(dx, dy, points=8, amp=0.3, r=n['r'])
                    
                elif "Star 14" in self.type:
                    act = self._star_activation(dx, dy, points=14, amp=0.2, r=n['r'])
                    
                elif "Marine Mine" in self.type:
                    # Forma agresiva con muchos picos
                    act = self._star_activation(dx, dy, points=24, amp=0.5, r=n['r'])

                # Acumulación de señal
                if n['lbl'] == 0: s0 += act
                else: s1 += act
            
            # Voto final (Argmax)
            preds.append(0 if s0 >= s1 else 1)
        return np.array(preds)

# --- LISTA COMPLETA DE MODELOS ---
models_list = [
    BenchmarkMLP(),
    VolumetricNetwork('Hard Sphere (L2)'),
    VolumetricNetwork('Soft Fuzzy'),
    VolumetricNetwork('Diamond (L1)'),
    VolumetricNetwork('Star 8-Point'),
    VolumetricNetwork('Star 14-Point'),
    VolumetricNetwork('Marine Mine'),
    VolumetricNetwork('Smart (Bayes+Lagr+MK)')
]

# --- VISUALIZACIÓN ---
# Configurar plot (2 filas, 4 columnas)
fig = plt.figure(figsize=(20, 10))
plt.subplots_adjust(hspace=0.3, wspace=0.2)
h = .02
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

print(f"{'MODELO':<30} | {'ACCURACY':<10} | {'TIEMPO (ms)':<12} | {'HARDWARE NOTE'}")
print("-" * 80)

for i, model in enumerate(models_list):
    # Train
    t0 = time.time()
    model.fit(X, y)
    dt = (time.time() - t0) * 1000
    
    # Predict
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    
    # Hardware Note
    hw_note = "FLOAT (GPU)"
    if "MLP" in model.type: hw_note = "MATRIX MUL (GPU)"
    elif "Diamond" in model.type: hw_note = "INT ADC (CPU/BitNet)"
    elif "Sphere" in model.type: hw_note = "INT MULT (CPU)"
    elif "Smart" in model.type: hw_note = "MIXED (High Reliability)"
    
    print(f"{model.type:<30} | {acc:.4f}     | {dt:.2f} ms     | {hw_note}")
    
    # Plot
    ax = plt.subplot(2, 4, i + 1)
    
    # Contour de decisión
    # Para dibujar rápido, predecimos sobre la malla
    if hasattr(model, 'predict'):
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=plt.cm.bwr, alpha=0.3)
    
    # Puntos originales
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.bwr, edgecolors='k', s=20, alpha=0.6)
    
    # Centros de neuronas (si existen)
    if hasattr(model, 'neurons'):
        for n in model.neurons:
            if "Smart" in model.type:
                 # Círculo real
                 circle = plt.Circle(n['c'], n['r'], color='k', fill=False, alpha=0.3, linestyle='--')
                 ax.add_artist(circle)
            else:
                 # Punto simple
                 ax.scatter(n['c'][0], n['c'][1], c='k', s=10, marker='x', alpha=0.5)

    ax.set_title(f"{model.type}\nAcc: {acc:.3f}", fontsize=10, fontweight='bold')
    ax.set_xticks(()); ax.set_yticks(())

plt.suptitle("Benchmark VNN: Geometrías de Activación Neuronal", fontsize=16)
plt.tight_layout()
plt.show()