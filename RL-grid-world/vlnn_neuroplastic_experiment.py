import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
import json

# ==========================================
# 0. CONFIGURACIÓN
# ==========================================
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

PARAMS = {
    'n_episodes': 800,    # Menos episodios necesarios porque VLNN aprende rápido
    'max_steps': 50,
    'gamma': 0.95,        # Descuento a futuro
    'epsilon_start': 1.0, # Exploración inicial
    'epsilon_end': 0.05,
    'epsilon_decay': 0.995,
    'hidden_dim': 64,     # Para la MLP
    'init_radius': 0.5,   # VLNN: Radio inicial ajustado para 6D (espacio normalizado)
    'min_radius': 0.1     # VLNN: Radio mínimo (refinamiento)
}

# ==========================================
# 1. GESTOR DE MAPAS (FAIRNESS)
# ==========================================
class MapManager:
    def __init__(self, n_episodes):
        self.map_seeds = [random.randint(0, 1000000) for _ in range(n_episodes)]
    def get_seed(self, episode_idx):
        return self.map_seeds[episode_idx]

# ==========================================
# 2. EL ENTORNO (GRIDWORLD + CURRICULUM + SHAPED REWARD)
# ==========================================
class GridWorld:
    def __init__(self, size=6, n_obstacles=5): # Empieza fácil (Curriculum)
        self.size = size
        self.n_obstacles = n_obstacles
        self.map = np.zeros((size, size))
        self.start = (0, 0)
        self.agent_pos = (0, 0)
        self.goal_pos = (size-1, size-1)
        self.obstacles = []
    
    def seed_map(self, seed, size, n_obstacles):
        self.size = size
        self.n_obstacles = n_obstacles
        random.seed(seed)
        np.random.seed(seed)
        
        self.map = np.zeros((self.size, self.size))
        self.obstacles = []
        
        candidates = []
        for i in range(self.size):
            for j in range(self.size):
                if (i,j) != (0,0) and (i,j) != (self.size-1, self.size-1):
                    candidates.append((i,j))
        
        # Safety check for obstacles
        actual_obs = min(len(candidates), self.n_obstacles)
        self.obstacles = random.sample(candidates, actual_obs)
        for obs in self.obstacles:
            self.map[obs] = 1
            
        self.goal_pos = (self.size-1, self.size-1)
        self.agent_pos = (0, 0)
        
    def reset(self, episode_seed, size, n_obstacles):
        self.seed_map(episode_seed, size, n_obstacles)
        return self.get_obs() # Returns raw obs, but wrapper will handle it

    def get_obs(self):
        # Patch 5x5 + Vector Goal (27 dims) -- Legacy, used by wrapper internally if needed or disregarded
        return np.zeros(27) 

    def step(self, action):
        moves = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1), 4: (0, 0)}
        dx, dy = moves[action]
        nx, ny = self.agent_pos[0] + dx, self.agent_pos[1] + dy
        
        gx, gy = self.goal_pos
        ax, ay = self.agent_pos
        dist_old = np.sqrt((gx - ax)**2 + (gy - ay)**2)
        
        reward = -0.1 # Costo de vida
        done = False
        collision = False
        reached_goal = False
        
        if not (0 <= nx < self.size and 0 <= ny < self.size) or ((nx, ny) in self.obstacles):
            reward = -2.0 # Castigo fuerte
            collision = True
        else:
            self.agent_pos = (nx, ny)
            dist_new = np.sqrt((gx - nx)**2 + (gy - ny)**2)
            reward += (dist_old - dist_new) * 3.0 # Shaped Reward fuerte
            
            if self.agent_pos == self.goal_pos:
                reward = 20.0 # Jackpot
                done = True
                reached_goal = True
                
        return self.get_obs(), reward, done, {'collision': collision, 'goal': reached_goal}

# ==========================================
# WRAPPER SENSORIAL (LIDAR - 6 Dims)
# ==========================================
class LidarWrapper:
    def __init__(self, env):
        self.env = env
        
    def get_lidar_obs(self):
        # En lugar de pixels, lanzamos 4 rayos (Up, Right, Down, Left)
        # Retorna: [dist_u, dist_r, dist_d, dist_l, goal_x, goal_y]
        ax, ay = self.env.agent_pos
        size = self.env.size
        obstacles = set(self.env.obstacles)
        
        sensors = []
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)] # U, R, D, L
        
        for dx, dy in directions:
            dist = 0
            for k in range(1, size): # Raycast
                nx, ny = ax + dx*k, ay + dy*k
                if not (0 <= nx < size and 0 <= ny < size) or (nx, ny) in obstacles:
                    break # Chocamos
                dist += 1
            # Normalizamos la distancia (0.0 a 1.0) para que la VNN coma rico
            sensors.append(dist / size)
            
        # Agregamos el vector goal (ya normalizado)
        # Recalculamos goal vector aqui para tenerlo a mano
        gx, gy = self.env.goal_pos
        dx, dy = gx - ax, gy - ay
        d = np.sqrt(dx**2 + dy**2) + 1e-5
        goal_vec = [dx/d, dy/d]
        
        return np.array(sensors + goal_vec, dtype=np.float32)

    # Delegamos el resto al env original
    def __getattr__(self, name):
        return getattr(self.env, name)

# ==========================================
# 3. EL CONTENDIENTE: MLP (Deep Q-Network Simple)
# ==========================================
class DQN_Agent:
    def __init__(self, input_dim, action_dim):
        self.model = nn.Sequential(
            nn.Linear(input_dim, PARAMS['hidden_dim']),
            nn.ReLU(),
            nn.Linear(PARAMS['hidden_dim'], PARAMS['hidden_dim']),
            nn.ReLU(),
            nn.Linear(PARAMS['hidden_dim'], action_dim)
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.epsilon = PARAMS['epsilon_start']
        
    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 4)
        state_t = torch.FloatTensor(state)
        with torch.no_grad():
            q_values = self.model(state_t)
        return torch.argmax(q_values).item()

    def update(self, state, action, reward, next_state, done):
        state_t = torch.FloatTensor(state)
        next_state_t = torch.FloatTensor(next_state)
        reward_t = torch.FloatTensor([reward])
        action_t = torch.LongTensor([action])
        
        # Q(s,a) actual
        q_vals = self.model(state_t)
        q_curr = q_vals[action_t]
        
        # Q target
        with torch.no_grad():
            q_next = self.model(next_state_t)
            max_q_next = torch.max(q_next)
            target = reward_t + (1 - int(done)) * PARAMS['gamma'] * max_q_next
            
        loss = self.criterion(q_curr, target)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.epsilon = max(PARAMS['epsilon_end'], self.epsilon * PARAMS['epsilon_decay'])

# ==========================================
# 4. EL HÉROE: VLNN COGNITIVE ENGINE (Con Neuroplasticidad)
# ==========================================

class CognitiveNeuron:
    def __init__(self, neuron_id, center, radius, n_actions=5):
        self.id = neuron_id
        self.c = np.array(center, dtype=np.float32)
        self.r = radius
        self.n_samples = 1
        self.q_values = np.zeros(n_actions) 
        # Añadimos contador de errores para saber cuándo achicarse
        self.confusion_count = 0 

    def match(self, x):
        return np.linalg.norm(x - self.c) <= self.r

    def update_bayes(self, x):
        # Ajuste suave del centro (Drift)
        self.n_samples += 1
        lr = 0.1 / np.sqrt(self.n_samples)
        self.c += lr * (x - self.c)

    def learn_value(self, action, reward, max_next_q):
        # Q-Learning clásico
        alpha = 0.2
        target = reward + 0.95 * max_next_q # Gamma hardcoded por simpleza
        prediction_error = target - self.q_values[action]
        self.q_values[action] += alpha * prediction_error
        
        # --- LA SOLUCIÓN: NEUROPLASTICIDAD POR DOLOR ---
        # Si recibimos un castigo fuerte (reward negativo) en una acción 
        # que creíamos que era buena (Q > 0), significa que esta esfera 
        # está mezclando situaciones seguras con peligrosas.
        if reward < -0.5 and self.q_values[action] > -0.1:
            self.shrink()

    def shrink(self):
        # "¡Auch! Me equivoqué. Me hago más chica para ser más precisa."
        self.r *= 0.8 # Reducir radio un 20%
        self.confusion_count += 1

class VolumetricCognitiveBrain:
    def __init__(self, input_dim, action_dim=5):
        self.neurons = [] # Lista de esferas
        self.action_dim = action_dim
        self.last_active_idx = None
        self.epsilon = PARAMS['epsilon_start']
        
    def get_active_neuron_idx(self, x):
        # 1. Buscar coincidencia geométrica
        # Prioridad: La más cercana que contenga al punto
        best_idx = None
        min_dist = float('inf')
        
        for i, neuron in enumerate(self.neurons):
            dist = np.linalg.norm(x - neuron.c)
            if dist <= neuron.r:
                # Estamos dentro
                if dist < min_dist:
                    min_dist = dist
                    best_idx = i
        
        # 2. Si no hay coincidencia, CREAR (One-Shot Learning)
        if best_idx is None:
            new_neuron = CognitiveNeuron(len(self.neurons), x, PARAMS['init_radius'], self.action_dim)
            self.neurons.append(new_neuron)
            best_idx = len(self.neurons) - 1
            
        return best_idx

    def get_action(self, state):
        # Fase Exploración
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        # Fase Explotación (Cognitiva)
        idx = self.get_active_neuron_idx(state)
        neuron = self.neurons[idx]
        
        # Retornar la acción con mayor valor Q en esta esfera
        # (Si todo es 0, elige random para romper simetría)
        if np.all(neuron.q_values == 0):
            return random.randint(0, self.action_dim - 1)
        return np.argmax(neuron.q_values)

    def update(self, state, action, reward, next_state, done):
        # 1. Identificar Dónde Estoy
        curr_idx = self.get_active_neuron_idx(state)
        curr_neuron = self.neurons[curr_idx]
        
        # 2. Refinar Geometría
        curr_neuron.update_bayes(state)
        
        # 3. Identificar Futuro
        next_val = 0
        if not done:
            next_idx = self.get_active_neuron_idx(next_state)
            next_neuron = self.neurons[next_idx]
            next_val = np.max(next_neuron.q_values)
            
        # 4. Aprender Valor y Posiblemente Achicarse (La Magia)
        curr_neuron.learn_value(action, reward, next_val)
        
        # Decay exploration
        self.epsilon = max(PARAMS['epsilon_end'], self.epsilon * PARAMS['epsilon_decay'])

    def get_parameter_count(self):
        # Cada neurona: Centro(6) + Radio(1) + Q_vals(5) + Stats(2) = ~14 floats
        return len(self.neurons) * (6 + 1 + 5 + 2)

# ==========================================
# 5. LOOP DE ENTRENAMIENTO UNIFICADO
# ==========================================
def train(agent_name, env, map_manager):
    # CAMBIO 1: Usar dimensiones reducidas
    # 4 sensores de distancia + 2 de goal = 6 dimensiones
    input_dim = 6
    action_dim = 5
    
    # Instanciar el Wrapper
    lidar_env = LidarWrapper(env)
    
    if agent_name == "MLP (Adam)":
        agent = DQN_Agent(input_dim, action_dim)
    else:
        agent = VolumetricCognitiveBrain(input_dim, action_dim)
        
    metrics = {'rewards': [], 'success': [], 'params': []}
    
    start_time = time.time()
    
    for episode in range(PARAMS['n_episodes']):
        # --- CURRICULUM ---
        if episode < 200:
            size, obs = 6, 3   # Jardín de Infantes
        elif episode < 500:
            size, obs = 9, 10  # Primaria
        else:
            size, obs = 12, 25 # Universidad (Hard)
            
        seed = map_manager.get_seed(episode)
        
        # CAMBIO 2: Usar la observación del LIDAR
        state_raw = env.reset(seed, size, obs) 
        state = lidar_env.get_lidar_obs()      
        
        total_reward = 0
        done = False
        
        for step in range(PARAMS['max_steps']):
            action = agent.get_action(state)
            next_state_raw, reward, done, info = env.step(action)
            next_state = lidar_env.get_lidar_obs() # Convertir a LIDAR
            
            # Learn Step
            agent.update(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            
            if done: break
            
        metrics['rewards'].append(total_reward)
        metrics['success'].append(1 if info['goal'] else 0)
        
        if agent_name != "MLP (Adam)":
            metrics['params'].append(agent.get_parameter_count())
        else:
            metrics['params'].append(64*6 + 64 + 64*64 + 64 + 64*5 + 5) # Fijo para 6D input

        if episode % 50 == 0:
            avg_win = np.mean(metrics['success'][-50:])
            print(f"[{agent_name}] Ep {episode} (Map {size}x{size}): WinRate {avg_win:.2f} | Params: {metrics['params'][-1]}")

    dt = time.time() - start_time
    return metrics, dt

# ==========================================
# 6. EJECUCIÓN
# ==========================================
if __name__ == "__main__":
    env = GridWorld()
    map_manager = MapManager(PARAMS['n_episodes'])
    
    print("--- 🧠 DUELO V4: NEUROPLASTICIDAD ADAPTATIVA (Smart Spheres) ---")
    
    print("\n>>> Entrenando Motor Cognitivo (VLNN - Smart)...")
    m_vlnn, t_vlnn = train("VLNN (Smart)", env, map_manager)
    
    print("\n>>> Entrenando Red Neuronal (MLP - Standard)...")
    m_mlp, t_mlp = train("MLP (Adam)", env, map_manager)
    
    # GRAFICAR
    def smooth(data, w=30):
        return np.convolve(data, np.ones(w)/w, mode='valid')

    plt.figure(figsize=(12, 6))
    
    # Win Rate
    plt.subplot(1, 2, 1)
    if len(m_vlnn['success']) > 30: plt.plot(smooth(m_vlnn['success']), label='VLNN (Smart)', color='blue', linewidth=2)
    if len(m_mlp['success']) > 30: plt.plot(smooth(m_mlp['success']), label='MLP (Adam)', color='magenta', linestyle='--')
    plt.title("Tasa de Éxito (Neuroplasticidad)")
    plt.axvline(x=200, color='gray', alpha=0.3, linestyle=':')
    plt.axvline(x=500, color='gray', alpha=0.3, linestyle=':')
    plt.xlabel("Episodios")
    plt.ylabel("Win Rate")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Parameters
    plt.subplot(1, 2, 2)
    plt.plot(m_vlnn['params'], label='VLNN Smart Params', color='blue')
    plt.plot(m_mlp['params'], label='MLP Fixed Params', color='magenta', linestyle='--')
    plt.title("Evolución de Complejidad (Parámetros)")
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('vnn_neuroplasticity_experiment.png')
    print("\nResultados guardados en 'vnn_neuroplasticity_experiment.png'")
    
    print(f"\nTIEMPOS: VLNN {t_vlnn:.2f}s | MLP {t_mlp:.2f}s")
    print(f"VLNN Final WinRate: {np.mean(m_vlnn['success'][-50:]):.2f}")
    print(f"MLP Final WinRate: {np.mean(m_mlp['success'][-50:]):.2f}")
    
    # JSON OUTPUT
    results = {
        "experiment": "Real Cognitive VLNN (Neuroplasticity) vs Deep MLP",
        "metrics": {
            "VLNN": {
                "training_time_sec": float(t_vlnn),
                "final_win_rate": float(np.mean(m_vlnn['success'][-50:])),
                "final_params": int(m_vlnn['params'][-1])
            },
            "MLP": {
                "training_time_sec": float(t_mlp),
                "final_win_rate": float(np.mean(m_mlp['success'][-50:])),
                "final_params": int(m_mlp['params'][-1])
            }
        },
        "config": PARAMS
    }
    print("\n--- EXPERIMENT RESULTS (JSON) ---")
    print(json.dumps(results, indent=4))
