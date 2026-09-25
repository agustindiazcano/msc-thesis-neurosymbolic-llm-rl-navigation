import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import copy
import time
import warnings
warnings.filterwarnings("ignore") # Silenciar alertas de convergencia
# --- 1. ENTORNO DE PRUEBA (CartPole) ---
# Usamos CartPole porque es sensible a la física y la latencia
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

# Dataset de experto para entrenar las VNNs candidatas
# --- REEMPLAZAR ESTA FUNCIÓN ---
# --- REEMPLAZAR ESTA FUNCIÓN ENTERA ---
def get_expert_data(n=2000):
    X, y = [], []
    print("Generando datos diversos (Uniform Sampling)...")
    
    for _ in range(n):
        # En lugar de simular una trayectoria continua que se puede atascar,
        # muestreamos estados aleatorios en todo el espacio de fases.
        # Esto OBLIGA a KMeans a encontrar clusters distintos.
        
        state = np.array([
            np.random.uniform(-2.4, 2.4),   # Posición (x) - Todo el riel
            np.random.uniform(-3.0, 3.0),   # Velocidad (x_dot)
            np.random.uniform(-0.25, 0.25), # Ángulo (theta) - Hasta 15 grados
            np.random.uniform(-3.0, 3.0)    # Vel. Angular (theta_dot)
        ])
        
        # Le preguntamos al Experto PID: "¿Qué harías aquí?"
        force = -(1.0*state[0] + 1.5*state[1] + 15.0*state[2] + 2.0*state[3])
        
        X.append(state)
        y.append(force)
        
    return np.array(X), np.array(y)

# --- 2. LA VNN MUTANTE (Variable Math) ---
class MetaVNN:
    def __init__(self, genome, n_neurons=30):
        self.genome = genome 
        # Decodificar Genes de la Fórmula
        self.kernel_type = int(genome[0]) # 0: Gauss, 1: Linear, 2: Inv
        self.dist_metric = genome[1]      # 1.0 a 3.0 (L1, L2, L3...)
        self.sharpness = genome[2]        # 0.1 a 5.0
        self.agg_mode = genome[3]         # 0.0 (Avg) a 1.0 (Max)
        
        self.n_neurons = n_neurons
        self.centers = None
        self.values = None
        self.radii = None

    def fit(self, X, y):
        # Entrenamiento rápido estándar (K-Means)
        kmeans = KMeans(n_clusters=self.n_neurons, n_init=1).fit(X)
        self.centers = kmeans.cluster_centers_
        self.values = np.zeros(self.n_neurons)
        # Radio base heurístico
        self.radii = np.full(self.n_neurons, 1.5) 
        
        labels = kmeans.predict(X)
        for i in range(self.n_neurons):
            idxs = np.where(labels==i)[0]
            if len(idxs)>0: self.values[i] = np.mean(y[idxs])

    def predict(self, state):
        # --- AQUI ESTA LA MAGIA DE LA FORMULA ---
        
        # 1. DISTANCIA GENERALIZADA (Minkowski variable)
        # d = (sum |x - c|^p)^(1/p)
        diff = np.abs(state - self.centers)
        dist_pow = np.sum(diff ** self.dist_metric, axis=1)
        dists = dist_pow ** (1.0 / self.dist_metric)
        
        # 2. KERNEL VARIABLE (La forma de la activación)
        # Normalizamos por radio y sharpness
        x_norm = (dists * self.sharpness) / self.radii
        
        if self.kernel_type == 0:   # Gaussian
            activations = np.exp(-x_norm**2)
        elif self.kernel_type == 1: # Triangular (Conic)
            activations = np.maximum(0, 1 - x_norm)
        else:                       # Inverse Quadratic (Gravity like)
            activations = 1.0 / (1.0 + x_norm**2)
            
        # 3. AGREGACIÓN HÍBRIDA (Avg vs Max)
        # Si agg_mode es 0 -> Weighted Avg
        # Si agg_mode es 1 -> Hard Max
        # Usamos Softmax con temperatura controlada por agg_mode
        
        if np.sum(activations) < 1e-5: return 0.0
        
        if self.agg_mode < 0.1: # Pure Average
            return np.average(self.values, weights=activations)
        elif self.agg_mode > 0.9: # Pure Max Winner
            idx = np.argmax(activations)
            return self.values[idx] * activations[idx]
        else:
            # Híbrido: Potenciar los ganadores (Softmax-like weighting)
            # Elevamos las activaciones para hacerlas más picudas
            power = 1.0 + (self.agg_mode * 10.0) # 1 a 11
            w_powered = activations ** power
            return np.average(self.values, weights=w_powered)

# --- 3. ALGORITMO GENÉTICO (Buscador de Fórmulas) ---
pop_size = 20
generations = 15
mutation_rate = 0.2

# Población inicial: [Kernel(0-2), Metric(1-3), Sharp(0.5-3), Agg(0-1)]
population = []
for _ in range(pop_size):
    gene = [
        np.random.randint(0, 3),      # Kernel
        np.random.uniform(1.0, 3.0),  # Metric
        np.random.uniform(0.5, 3.0),  # Sharpness
        np.random.uniform(0.0, 1.0)   # Aggregation
    ]
    population.append(gene)

print(f"Buscando la Fórmula Perfecta (Meta-Optimización)...")
print(f"Genes: [Kernel, Distancia(p), Sharpness, Agregacion]")

# --- GENERAR DATOS DE ENTRENAMIENTO ---
X_train, y_train = get_expert_data(n=2000)

history_best = []
best_gene_global = None
best_score_global = 0

for gen in range(generations):
    scores = []
    
    for gene in population:
        # 1. Instanciar VNN con esta fórmula
        vnn = MetaVNN(gene, n_neurons=20) # Red pequeña para forzar eficiencia
        vnn.fit(X_train, y_train)
        
        # 2. Test de Resistencia en CartPole (Fitness)
        env = CartPoleEnv()
        state = np.array([0,0,0.1,0]) # Empezar un poco caído
        steps = 0
        for _ in range(1000):
            force = vnn.predict(state)
            state = env.step(state, force)
            if abs(state[2]) > 0.20 or abs(state[0]) > 2.4: break
            steps += 1
        scores.append(steps)

    # Estadísticas
    scores = np.array(scores)
    best_idx = np.argmax(scores)
    best_score = scores[best_idx]
    best_gene = population[best_idx]
    
    if best_score > best_score_global:
        best_score_global = best_score
        best_gene_global = copy.deepcopy(best_gene)
    
    history_best.append(best_score)
    
    # Decodificar para print
    k_name = ["Gauss", "Conic", "InvQuad"][int(best_gene[0])]
    print(f"Gen {gen+1}: Mejor Score {best_score} | Formula: {k_name}, L-{best_gene[1]:.1f}, Sharp {best_gene[2]:.1f}, Agg {best_gene[3]:.2f}")

    # Selección y Mutación (Elitismo)
    # Ordenar por score
    sorted_idxs = np.argsort(scores)[::-1]
    survivors = [population[i] for i in sorted_idxs[:5]] # Top 5
    
    new_pop = []
    while len(new_pop) < pop_size:
        parent = survivors[np.random.randint(0, 5)]
        child = copy.deepcopy(parent)
        
        # Mutar
        if np.random.rand() < mutation_rate:
            choice = np.random.randint(0, 4)
            if choice == 0: child[0] = np.random.randint(0, 3)
            if choice == 1: child[1] = np.clip(child[1] + np.random.normal(0, 0.5), 1.0, 4.0)
            if choice == 2: child[2] = np.clip(child[2] + np.random.normal(0, 0.5), 0.1, 5.0)
            if choice == 3: child[3] = np.clip(child[3] + np.random.normal(0, 0.2), 0.0, 1.0)
        new_pop.append(child)
    
    population = new_pop

# --- 4. RESULTADO FINAL ---
print("\n" + "="*60)
print("LA FÓRMULA GANADORA ES:")
bg = best_gene_global
k_types = ["Gaussian (Soft)", "Conic (Hard)", "Inverse Quadratic (Gravity)"]
print(f"1. Forma del Kernel:     {k_types[int(bg[0])]}")
print(f"2. Métrica de Distancia: L-{bg[1]:.2f} (1=Manhattan, 2=Euclid)")
print(f"3. Nitidez (Sharpness):  {bg[2]:.2f}")
print(f"4. Agregación (Avg-Max): {bg[3]:.2f} (0=Avg, 1=Max)")
print("="*60)

# Visualizar convergencia
plt.plot(history_best)
plt.xlabel("Generación")
plt.ylabel("Tiempo de Supervivencia (Frames)")
plt.title("Evolución de la Arquitectura Matemática VNN")
plt.grid(True)
plt.show()