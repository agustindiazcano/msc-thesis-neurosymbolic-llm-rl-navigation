import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import time

# --- CONFIGURACIÓN DEL EXPERIMENTO ---
STEPS = 400
TRAP_START = 200 # A mitad de camino cambiamos la física

def generate_dynamic_path(steps):
    """Genera una trayectoria que cambia de reglas a la mitad"""
    t = np.linspace(0, 8*np.pi, steps)
    path = np.zeros((steps, 2))
    
    # Fase 1: Espiral Logarítmica Suave (Physics A)
    path[:TRAP_START, 0] = 0.5 * t[:TRAP_START] * np.cos(t[:TRAP_START])
    path[:TRAP_START, 1] = 0.5 * t[:TRAP_START] * np.sin(t[:TRAP_START])
    
    # Fase 2: LA TRAMPA (Invierte gravedad + ZigZag Caótico) (Physics B)
    t2 = t[TRAP_START:]
    path[TRAP_START:, 0] = path[TRAP_START-1, 0] - 0.5 * (t2 - t[TRAP_START]) * np.cos(2*t2)
    path[TRAP_START:, 1] = path[TRAP_START-1, 1] + 0.5 * (t2 - t[TRAP_START]) * np.sin(2*t2) + np.sin(10*t2)
    
    return path

# ==========================================
# 1. BASELINE: MLP (Online Learning)
# ==========================================
class TrackerMLP:
    def __init__(self):
        self.model = MLPRegressor(hidden_layer_sizes=(64, 64), 
                                  activation='relu', solver='adam', 
                                  max_iter=1, warm_start=True, learning_rate_init=0.01)
        self.name = "MLP (Online)"
        self.buffer_X = []
        self.buffer_y = []
        
    def predict(self, x):
        if len(self.buffer_X) < 10: return x 
        return self.model.predict([x])[0]
        
    def learn(self, x_t, x_next):
        self.buffer_X.append(x_t)
        self.buffer_y.append(x_next)
        window = 20 
        X_train = np.array(self.buffer_X[-window:])
        y_train = np.array(self.buffer_y[-window:])
        self.model.fit(X_train, y_train)

# ==========================================
# 2. VNN: Vector Field Tracker (All Variants)
# ==========================================
class VNN_VectorField:
    def __init__(self, model_type, base_radius=1.0):
        self.name = model_type
        self.type = model_type
        self.neurons = [] 
        self.base_r = base_radius
        
    def _star_activation(self, dx, dy, points, amp, r):
        rad = np.sqrt(dx**2 + dy**2)
        theta = np.arctan2(dy, dx)
        r_eff = r * (1.0 - amp * np.abs(np.sin(points/2 * theta)))
        return 1.0 if rad < r_eff else 0.0

    def _get_activation(self, n, x):
        dx = x[0] - n['c'][0]
        dy = x[1] - n['c'][1]
        act = 0.0
        
        # --- GEOMETRÍAS ---
        if "Diamond" in self.type:
            d = abs(dx) + abs(dy)
            if d < n['r']: act = 1.0
            
        elif "Hard Sphere" in self.type:
            d2 = dx**2 + dy**2
            if d2 < n['r']**2: act = 1.0
            
        elif "Soft Fuzzy" in self.type:
            d2 = dx**2 + dy**2
            sigma = n['r'] / 1.5
            act = np.exp(-d2 / (2 * sigma**2))
            
        elif "Smart+Soft" in self.type:
             # Combina Smart (Confianza) con Soft (Gaussiana)
             # Permite transiciones más suaves en el campo vectorial
             d2 = dx**2 + dy**2
             sigma = n['r'] / 1.5
             # Gaussiana ponderada por confianza
             act = np.exp(-d2 / (2 * sigma**2)) * n['confidence_weight']

        elif "Smart" in self.type:
            d = np.sqrt(dx**2 + dy**2)
            if d < n['r']: act = n['confidence_weight'] # Ponderado por confianza
            
        elif "Star 8" in self.type:
            act = self._star_activation(dx, dy, points=8, amp=0.3, r=n['r'])
            
        elif "Star 14" in self.type:
            act = self._star_activation(dx, dy, points=14, amp=0.2, r=n['r'])
            
        elif "Marine Mine" in self.type:
            act = self._star_activation(dx, dy, points=24, amp=0.5, r=n['r'])
            
        return act

    def predict(self, x):
        active_vectors = []
        total_weight = 0
        
        for n in self.neurons:
            w = self._get_activation(n, x)
            # Distance decay para todos (más cerca del centro = más influencia)
            if w > 0:
                dist = np.linalg.norm(x - n['c'])
                w *= (1.0 / (dist + 0.1)) 
                
                active_vectors.append(n['v'] * w)
                total_weight += w
                
        if total_weight > 0:
            avg_velocity = np.sum(active_vectors, axis=0) / total_weight
            return x + avg_velocity
        else:
            return x 
            
    def learn(self, x_t, x_next):
        velocity = x_next - x_t
        
        # Buscar mejor neurona (mantenemos simple Euclidean lookup para update)
        best_n = None
        min_dist = float('inf')
        
        for n in self.neurons:
            dist = np.linalg.norm(x_t - n['c'])
            if dist < min_dist:
                min_dist = dist
                best_n = n
        
        # Update o Create
        if best_n and min_dist < best_n['r']:
            # SMART UPDATE para todos (adaptabilidad básica)
            learning_rate = 0.3
            if "Smart" in self.type: learning_rate = 0.5 # Smart aprende más rápido
            
            best_n['v'] = (1 - learning_rate) * best_n['v'] + learning_rate * velocity
            best_n['c'] += 0.1 * (x_t - best_n['c']) # Kohonen drift
            
            if "Smart" in self.type:
                best_n['confidence_weight'] = min(best_n['confidence_weight'] + 0.1, 2.0)
                # Penalizar radio si hay error grande (Refinamiento)
                # (Simulado simple aquí)
                
        else:
            self.neurons.append({
                'c': x_t,
                'v': velocity,
                'r': self.base_r,
                'confidence_weight': 1.0
            })

# --- EJECUCIÓN ---

path = generate_dynamic_path(STEPS)

models = [
    TrackerMLP(),
    VNN_VectorField("Hard Sphere (L2)"),
    VNN_VectorField("Soft Fuzzy"),
    VNN_VectorField("Smart+Soft (Hybrid)"), # <--- NUEVO
    VNN_VectorField("Diamond (L1)"),
    VNN_VectorField("Star 8-Point"),
    VNN_VectorField("Star 14-Point"),
    VNN_VectorField("Marine Mine"),
    VNN_VectorField("Smart (Bayes+Lagr+MK)")
]

errors = {m.name: [] for m in models} # Acumulador de errores

print(f"{'STEP':<5} | {'EVENT':<20} | {'MLP':<8} | {'SPHERE':<8} | {'S+SOFT':<8} | {'SMART':<8}")
print("-" * 80)

for t in range(STEPS - 1):
    current_pos = path[t]
    actual_next = path[t+1]
    
    event_msg = ">>> PHYSICS CHANGE <<<" if t == TRAP_START else ""
    
    step_errs = {}
    
    for model in models:
        # Predict
        pred = model.predict(current_pos)
        
        # Error
        err = np.linalg.norm(pred - actual_next)
        errors[model.name].append(err)
        step_errs[model.name] = err
        
        # Learn
        model.learn(current_pos, actual_next)

    if t % 50 == 0 or t == TRAP_START:
        print(f"{t:<5} | {event_msg:<20} | {step_errs['MLP (Online)']:.4f}   | {step_errs['Hard Sphere (L2)']:.4f}   | {step_errs['Smart+Soft (Hybrid)']:.4f}   | {step_errs['Smart (Bayes+Lagr+MK)']:.4f}")

# --- VISUALIZACIÓN ---
plt.figure(figsize=(12, 6))
cols = float(len(models))
for i, m in enumerate(models):
    # Suavizar
    smooth = np.convolve(errors[m.name], np.ones(15)/15, mode='valid')
    plt.plot(smooth, label=m.name, alpha=0.8)

plt.axvline(x=TRAP_START, color='k', linestyle='--', label='Trap Start', linewidth=2)
plt.title("Adaptabilidad Dinámica: Error vs Tiempo")
plt.xlabel("Tiempo (t)")
plt.ylabel("Error de Rastreo")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
