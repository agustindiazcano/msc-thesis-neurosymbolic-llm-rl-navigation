import numpy as np
import optuna
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings("ignore") 

# --- 1. ENTORNO Y DATOS (CON EL ARREGLO APLICADO) ---
class CartPoleEnv:
    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = 1.1
        self.length = 0.5 
        self.polemass_length = 0.05
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
        # MUESTREO UNIFORME (CRÍTICO PARA QUE FUNCIONE)
        state = np.array([
            np.random.uniform(-2.0, 2.0),
            np.random.uniform(-2.0, 2.0),
            np.random.uniform(-0.2, 0.2),
            np.random.uniform(-2.0, 2.0)
        ])
        # PID Experto
        force = -(1.0*state[0] + 1.5*state[1] + 15.0*state[2] + 2.0*state[3])
        X.append(state); y.append(force)
    return np.array(X), np.array(y)

# Generamos datos BUENOS una sola vez
X_train, y_train = get_diverse_expert_data()

# --- 2. LA VNN CONFIGURABLE ---
class ParametricVNN:
    def __init__(self, p_dist, sharpness, agg_mode, kernel_type, n_neurons=30):
        self.p_dist = p_dist       # L-p metric
        self.sharpness = sharpness
        self.agg_mode = agg_mode   # 0=Avg, 1=Max
        self.kernel_type = kernel_type # 'gauss', 'conic', 'inv'
        self.n_neurons = n_neurons
        self.centers = None; self.values = None; self.radii = None

    def fit(self, X, y):
        kmeans = KMeans(n_clusters=self.n_neurons, n_init=1).fit(X)
        self.centers = kmeans.cluster_centers_
        self.values = np.zeros(self.n_neurons)
        self.radii = np.full(self.n_neurons, 2.0) # Radio fijo amplio
        
        labels = kmeans.predict(X)
        for i in range(self.n_neurons):
            idxs = np.where(labels==i)[0]
            if len(idxs)>0: self.values[i] = np.mean(y[idxs])

    def predict(self, state):
        # 1. Distancia Minkowski generalizada
        diff = np.abs(state - self.centers)
        dist_pow = np.sum(diff ** self.p_dist, axis=1)
        dists = dist_pow ** (1.0 / self.p_dist)
        
        # 2. Kernel
        x_norm = (dists * self.sharpness) / self.radii
        if self.kernel_type == 'gauss':
            activations = np.exp(-x_norm**2)
        elif self.kernel_type == 'conic':
            activations = np.maximum(0, 1 - x_norm)
        else: # inv
            activations = 1.0 / (1.0 + x_norm**2)

        # 3. Agregación Híbrida
        if np.sum(activations) < 1e-6: return 0.0
        
        # Interpolar entre Average (0) y Max (1) usando Softmax con temperatura
        # agg_mode 0 -> T=1, agg_mode 1 -> T=20 (casi max)
        T = 1.0 + (self.agg_mode * 20.0)
        w_powered = activations ** T
        
        return np.average(self.values, weights=w_powered)

# --- 3. OBJETIVO DE OPTUNA ---
def objective(trial):
    # Definir el espacio de búsqueda de la fórmula
    p_dist = trial.suggest_float("p_dist", 1.0, 3.0) # L1 a L3
    sharpness = trial.suggest_float("sharpness", 0.5, 5.0)
    agg_mode = trial.suggest_float("agg_mode", 0.0, 1.0) # 0=Avg, 1=Max
    kernel_type = trial.suggest_categorical("kernel", ["gauss", "conic", "inv"])
    
    # Instanciar y Entrenar
    vnn = ParametricVNN(p_dist, sharpness, agg_mode, kernel_type)
    vnn.fit(X_train, y_train)
    
    # Evaluar en CartPole (Fitness = Tiempo de vida)
    env = CartPoleEnv()
    state = np.array([0,0,0.1,0])
    steps = 0
    for _ in range(500): # Max 500 steps
        force = vnn.predict(state)
        state = env.step(state, force)
        if abs(state[2]) > 0.20 or abs(state[0]) > 2.4: break
        steps += 1
        
    return steps # Optuna intentará maximizar esto

# --- 4. EJECUCIÓN ---
if __name__ == "__main__":
    print("Buscando la Fórmula Perfecta con Optuna...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50) # 50 intentos inteligentes

    print("\n" + "="*60)
    print("LA FÓRMULA DOMADORA ES:")
    best = study.best_params
    print(f"1. Kernel:    {best['kernel']}")
    print(f"2. Distancia: L-{best['p_dist']:.2f}")
    print(f"3. Nitidez:   {best['sharpness']:.2f}")
    print(f"4. Agregación:{best['agg_mode']:.2f} (Cerca de 1.0 = Teoría Confirmada)")
    print(f"SCORE FINAL:  {study.best_value} steps")
    print("="*60)
