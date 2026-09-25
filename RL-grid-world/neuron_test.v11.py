import numpy as np
import optuna
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings("ignore")

# --- 1. ENTORNO Y DATOS DIVERSOS ---
class CartPoleEnv:
    def __init__(self):
        self.gravity = 9.8; self.masscart = 1.0; self.masspole = 0.1
        self.total_mass = 1.1; self.length = 0.5; self.polemass_length = 0.05
        self.tau = 0.02

    def step(self, state, force):
        x, x_dot, theta, theta_dot = state
        costheta = np.cos(theta); sintheta = np.sin(theta)
        temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0/3.0 - self.masspole * costheta**2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        x += self.tau * x_dot; x_dot += self.tau * xacc
        theta += self.tau * theta_dot; theta_dot += self.tau * thetaacc
        return np.array([x, x_dot, theta, theta_dot])

def get_diverse_expert_data(n=3000):
    X, y = [], []
    for _ in range(n):
        # Muestreo Uniforme para evitar el error de clusters
        state = np.array([
            np.random.uniform(-2.4, 2.4),
            np.random.uniform(-3.0, 3.0),
            np.random.uniform(-0.25, 0.25),
            np.random.uniform(-3.0, 3.0)
        ])
        force = -(1.0*state[0] + 1.5*state[1] + 15.0*state[2] + 2.0*state[3])
        X.append(state); y.append(force)
    return np.array(X), np.array(y)

X_train, y_train = get_diverse_expert_data()

# --- 2. LA NEURONA "MOTOR DE INFERENCIA" (4 CABEZAS) ---
class FusionVNN:
    def __init__(self, weights, n_neurons=40):
        # weights: [w_euclid, w_chebyshev, w_manhattan, w_conic]
        self.weights = np.array(weights)
        self.weights /= np.sum(self.weights) # Normalizar para que sumen 1
        
        self.n_neurons = n_neurons
        self.centers = None; self.values = None; self.radii = None

    def fit(self, X, y):
        kmeans = KMeans(n_clusters=self.n_neurons, n_init=1).fit(X)
        self.centers = kmeans.cluster_centers_
        self.values = np.zeros(self.n_neurons)
        self.radii = np.full(self.n_neurons, 2.5) # Radio base amplio
        
        labels = kmeans.predict(X)
        for i in range(self.n_neurons):
            idxs = np.where(labels==i)[0]
            if len(idxs)>0: self.values[i] = np.mean(y[idxs])

    def predict(self, state):
        # Calculamos la diferencia vectorial una sola vez
        diff = np.abs(state - self.centers) # (N, 4)
        
        # --- CABEZA 1: EUCLÍDEA (Física Real) ---
        d_euclid = np.linalg.norm(diff, axis=1)
        act_euclid = np.exp(-(d_euclid/self.radii)**2)
        
        # --- CABEZA 2: CHEBYSHEV (Límites Robóticos) ---
        d_cheby = np.max(diff, axis=1)
        act_cheby = np.exp(-(d_cheby/self.radii)**2)
        
        # --- CABEZA 3: MANHATTAN (Eficiencia de Costo) ---
        d_man = np.sum(diff, axis=1)
        # Ajustamos radio para Manhattan porque las distancias suman más
        act_man = np.exp(-(d_man/(self.radii*1.5))**2) 
        
        # --- CABEZA 4: CÓNICA/LAGRANGE (Precisión Dura) ---
        # Usamos Euclídea pero con activación lineal (triangular)
        x_norm = d_euclid / self.radii
        act_conic = np.maximum(0, 1 - x_norm)

        # --- FUSIÓN DE INFERENCIA ---
        # La activación final es la suma ponderada de las 4 opiniones
        # w[0]*Euclid + w[1]*Cheby + w[2]*Man + w[3]*Conic
        combined_activation = (
            self.weights[0] * act_euclid +
            self.weights[1] * act_cheby +
            self.weights[2] * act_man +
            self.weights[3] * act_conic
        )
        
        # --- AGREGACIÓN FINAL ---
        if np.sum(combined_activation) < 1e-6: return 0.0
        
        # Usamos Softmax Aggregation (Temperatura fija)
        # Esto permite que la neurona más activa domine
        w_powered = combined_activation ** 2.0 
        
        return np.average(self.values, weights=w_powered)

# --- 3. OPTIMIZAR LA MEZCLA CON OPTUNA ---
def objective(trial):
    # Optuna busca cuánto peso darle a cada lógica
    w_e = trial.suggest_float("w_euclid", 0.0, 1.0)
    w_c = trial.suggest_float("w_chebyshev", 0.0, 1.0)
    w_m = trial.suggest_float("w_manhattan", 0.0, 1.0)
    w_k = trial.suggest_float("w_conic", 0.0, 1.0)
    
    # Crear VNN con estos pesos
    vnn = FusionVNN([w_e, w_c, w_m, w_k], n_neurons=30)
    vnn.fit(X_train, y_train)
    
    # Test de Supervivencia
    env = CartPoleEnv()
    state = np.array([0,0,0.1,0])
    steps = 0
    total_stab = 0
    
    for _ in range(800): # Max 800 steps
        force = vnn.predict(state)
        state = env.step(state, force)
        total_stab += abs(state[2]) # Acumular error de ángulo
        if abs(state[2]) > 0.20 or abs(state[0]) > 2.4: break
        steps += 1
        
    # Fitness: Durar mucho penalizando el tambaleo
    return steps - (total_stab * 0.5)

# --- 4. EJECUCIÓN ---
print("Iniciando Calibración del Motor de Inferencia Híbrido...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

print("\n" + "="*60)
print("LA MEZCLA MAESTRA DE TU MOTOR ES:")
bp = study.best_params
total = sum(bp.values())
print(f"1. Euclídea (Física):    {bp['w_euclid']/total*100:.1f}%")
print(f"2. Chebyshev (Límites):  {bp['w_chebyshev']/total*100:.1f}%")
print(f"3. Manhattan (Costos):   {bp['w_manhattan']/total*100:.1f}%")
print(f"4. Cónica (Precisión):   {bp['w_conic']/total*100:.1f}%")
print("-" * 60)
print(f"SCORE FINAL: {study.best_value:.2f}")
print("="*60)