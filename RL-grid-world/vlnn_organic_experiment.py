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
    'n_episodes': 800,
    'max_steps': 50,
    'gamma': 0.95,
    'epsilon_start': 1.0,
    'epsilon_end': 0.05,
    'epsilon_decay': 0.995,
    'hidden_dim': 64,
    'init_radius': 0.6,   
    'min_radius': 0.15     
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
# 2. EL ENTORNO (GRIDWORLD + WRAPPER)
# ==========================================
class GridWorld:
    def __init__(self, size=6, n_obstacles=5):
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
        
        actual_obs = min(len(candidates), self.n_obstacles)
        self.obstacles = random.sample(candidates, actual_obs)
        for obs in self.obstacles:
            self.map[obs] = 1
            
        self.goal_pos = (self.size-1, self.size-1)
        self.agent_pos = (0, 0)
        
    def reset(self, episode_seed, size, n_obstacles):
        self.seed_map(episode_seed, size, n_obstacles)
        return self.get_obs()

    def get_obs(self):
        return np.zeros(27) 

    def step(self, action):
        moves = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1), 4: (0, 0)}
        dx, dy = moves[action]
        nx, ny = self.agent_pos[0] + dx, self.agent_pos[1] + dy
        
        gx, gy = self.goal_pos
        ax, ay = self.agent_pos
        dist_old = np.sqrt((gx - ax)**2 + (gy - ay)**2)
        
        reward = -0.1
        done = False
        collision = False
        reached_goal = False
        
        if not (0 <= nx < self.size and 0 <= ny < self.size) or ((nx, ny) in self.obstacles):
            reward = -2.0 
            collision = True
        else:
            self.agent_pos = (nx, ny)
            dist_new = np.sqrt((gx - nx)**2 + (gy - ny)**2)
            reward += (dist_old - dist_new) * 3.0 
            
            if self.agent_pos == self.goal_pos:
                reward = 20.0 
                done = True
                reached_goal = True
                
        return self.get_obs(), reward, done, {'collision': collision, 'goal': reached_goal}

class LidarWrapper:
    def __init__(self, env):
        self.env = env
        
    def get_lidar_obs(self):
        ax, ay = self.env.agent_pos
        size = self.env.size
        obstacles = set(self.env.obstacles)
        
        sensors = []
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)] # U, R, D, L
        
        for dx, dy in directions:
            dist = 0
            for k in range(1, size):
                nx, ny = ax + dx*k, ay + dy*k
                if not (0 <= nx < size and 0 <= ny < size) or (nx, ny) in obstacles:
                    break 
                dist += 1
            sensors.append(dist / size)
            
        gx, gy = self.env.goal_pos
        dx, dy = gx - ax, gy - ay
        d = np.sqrt(dx**2 + dy**2) + 1e-5
        goal_vec = [dx/d, dy/d]
        
        return np.array(sensors + goal_vec, dtype=np.float32)

    def __getattr__(self, name):
        return getattr(self.env, name)

# ==========================================
# 3. EL CONTENDIENTE: MLP
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
        
        q_vals = self.model(state_t)
        q_curr = q_vals[action_t]
        
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
# 4. EL HÉROE: VLNN - MITOSIS FRACTAL + PLASTILINA FUSION
# ==========================================

class CognitiveNeuron:
    def __init__(self, neuron_id, center, radius, n_actions=5):
        self.id = neuron_id
        self.c = np.array(center, dtype=np.float32)
        self.r = radius
        self.n_samples = 1
        self.q_values = np.zeros(n_actions) 
        self.pain_accum = 0       
        self.pain_threshold = 8   
        self.last_used = 0  # Timestamp para apoptosis   

    def match(self, x):
        return np.linalg.norm(x - self.c) <= self.r

    def update_bayes(self, x):
        self.n_samples += 1
        lr = 0.1 / np.sqrt(self.n_samples)
        self.c += lr * (x - self.c)

    def learn_value(self, action, reward, max_next_q):
        alpha = 0.2
        target = reward + 0.95 * max_next_q 
        prediction_error = target - self.q_values[action]
        self.q_values[action] += alpha * prediction_error
        
        if reward < -0.5 and self.q_values[action] > -0.5:
            self.pain_accum += 1
            
    def should_mitose(self):
        return self.pain_accum >= self.pain_threshold and self.r > PARAMS['min_radius']

    def mitosis(self, n_daughters=4):
        daughters = []
        input_dim = self.c.shape[0]
        
        for _ in range(n_daughters):
            offset = np.random.randn(input_dim)
            if np.linalg.norm(offset) > 0:
                offset /= np.linalg.norm(offset) 
            
            dist_offset = np.random.uniform(0, self.r * 0.5)
            new_c = self.c + offset * dist_offset
            
            new_r = self.r * 0.6 
            
            d_neuron = CognitiveNeuron(None, new_c, new_r, len(self.q_values))
            # FIX 3: Herencia con Mutación/Duda (90% confianza)
            d_neuron.q_values = self.q_values.copy() * 0.9 
            d_neuron.pain_accum = 0 
            
            daughters.append(d_neuron)
            
        return daughters

class VolumetricCognitiveBrain:
    def __init__(self, input_dim, action_dim=5):
        self.neurons = [] 
        self.action_dim = action_dim
        self.epsilon = PARAMS['epsilon_start']
        
    def get_active_neuron_idx(self, x, current_step=0):
        best_idx = None
        min_dist = float('inf')
        
        for i, neuron in enumerate(self.neurons):
            dist = np.linalg.norm(x - neuron.c)
            if dist <= neuron.r:
                if dist < min_dist:
                    min_dist = dist
                    best_idx = i
        
        if best_idx is None:
            new_neuron = CognitiveNeuron(len(self.neurons), x, PARAMS['init_radius'], self.action_dim)
            new_neuron.last_used = current_step
            self.neurons.append(new_neuron)
            best_idx = len(self.neurons) - 1
        else:
            # Actualizar timestamp de uso
            self.neurons[best_idx].last_used = current_step
            
        return best_idx

    def get_action(self, state, current_step=0):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        idx = self.get_active_neuron_idx(state, current_step)
        neuron = self.neurons[idx]
        if np.all(neuron.q_values == 0):
            return random.randint(0, self.action_dim - 1)
        return np.argmax(neuron.q_values)

    def update(self, state, action, reward, next_state, done, current_step=0):
        curr_idx = self.get_active_neuron_idx(state, current_step)
        curr_neuron = self.neurons[curr_idx]
        
        curr_neuron.update_bayes(state)
        
        next_val = 0
        if not done:
            next_idx = self.get_active_neuron_idx(next_state, current_step)
            next_neuron = self.neurons[next_idx]
            next_val = np.max(next_neuron.q_values)
            
        curr_neuron.learn_value(action, reward, next_val)
        
        # --- MITOSIS FRACTAL ---
        if curr_neuron.should_mitose():
            daughters = curr_neuron.mitosis()
            self.neurons.pop(curr_idx)
            for d in daughters:
                d.id = len(self.neurons) 
                self.neurons.append(d)
        
        self.epsilon = max(PARAMS['epsilon_end'], self.epsilon * PARAMS['epsilon_decay'])

    def fuse_neurons(self):
        """
        FACTOR DE LIMITACIÓN DE CRECIMIENTO (PLASTILINA).
        Busca esferas adyacentes que "piensan igual" y las fusiona.
        Versión simplificada para evitar loops infinitos.
        """
        if len(self.neurons) < 2:
            return
            
        merged_count = 0
        to_remove = set()
        
        # Una sola pasada, marcamos para eliminar
        for i in range(len(self.neurons)):
            if i in to_remove:
                continue
                
            n1 = self.neurons[i]
            
            for j in range(i + 1, len(self.neurons)):
                if j in to_remove:
                    continue
                    
                n2 = self.neurons[j]
                
                # ¿Se tocan Y piensan igual?
                dist = np.linalg.norm(n1.c - n2.c)
                if dist < (n1.r + n2.r) * 1.2:  # 20% tolerancia "blanditas"
                    
                    # FIX 1: Solo fusionar si coinciden en la ACCIÓN GANADORA
                    best_action_1 = np.argmax(n1.q_values)
                    best_action_2 = np.argmax(n2.q_values)
                    
                    if best_action_1 != best_action_2:
                        continue  # No mezclar peras con manzanas
                    
                    q_diff = np.linalg.norm(n1.q_values - n2.q_values)
                    
                    if q_diff < 0.5:  # Conservador, no 1.0
                        # Fusionar n2 en n1
                        total_weight = n1.r + n2.r
                        n1.c = (n1.c * n1.r + n2.c * n2.r) / total_weight
                        n1.r = max(n1.r, n2.r) + dist * 0.3
                        n1.q_values = (n1.q_values + n2.q_values) / 2
                        n1.pain_accum = 0
                        
                        to_remove.add(j)
                        merged_count += 1
                        break  # Solo una fusión por n1 por iteración
        
        # Eliminar neuronas fusionadas (de atrás hacia adelante)
        for idx in sorted(to_remove, reverse=True):
            self.neurons.pop(idx)
    
    def prune_dead_neurons(self, current_step, max_age=2000):
        """
        APOPTOSIS: Elimina neuronas que no se activaron en los últimos 'max_age' pasos.
        Esto borra la memoria del mapa anterior que ya no sirve.
        """
        initial_count = len(self.neurons)
        self.neurons = [n for n in self.neurons if (current_step - n.last_used) < max_age]
        
        deleted = initial_count - len(self.neurons)
        # if deleted > 0:
        #     print(f"[LIMPIEZA] Se eliminaron {deleted} neuronas viejas.")

    def get_parameter_count(self):
        return len(self.neurons) * (6 + 1 + 5 + 2)

# ==========================================
# 5. LOOP DE ENTRENAMIENTO
# ==========================================
def train(agent_name, env, map_manager):
    input_dim = 6
    action_dim = 5
    lidar_env = LidarWrapper(env)
    
    if agent_name == "MLP (Adam)":
        agent = DQN_Agent(input_dim, action_dim)
    else:
        agent = VolumetricCognitiveBrain(input_dim, action_dim)
        
    metrics = {'rewards': [], 'success': [], 'params': []}
    start_time = time.time()
    
    global_step_counter = 0  # Contador de pasos global para apoptosis
    
    for episode in range(PARAMS['n_episodes']):
        # Track map size changes
        old_size = 0 if episode == 0 else size
        
        if episode < 200: size, obs = 6, 3
        elif episode < 500: size, obs = 9, 10
        else: size, obs = 12, 25
        
        # FIX 2: RESET EXPLORACIÓN al cambiar de mapa
        if agent_name != "MLP (Adam)" and size != old_size and episode > 0:
            agent.epsilon = 0.8  # Shock de adrenalina
            
        seed = map_manager.get_seed(episode)
        state_raw = env.reset(seed, size, obs) 
        state = lidar_env.get_lidar_obs()      
        
        total_reward = 0
        done = False
        
        for step in range(PARAMS['max_steps']):
            global_step_counter += 1
            
            if agent_name == "MLP (Adam)":
                action = agent.get_action(state)
            else:
                action = agent.get_action(state, global_step_counter)
                
            next_state_raw, reward, done, info = env.step(action)
            next_state = lidar_env.get_lidar_obs()
            
            if agent_name == "MLP (Adam)":
                agent.update(state, action, reward, next_state, done)
            else:
                agent.update(state, action, reward, next_state, done, global_step_counter)
            
            state = next_state
            total_reward += reward
            if done: break

        # PLASTILINA TIME + APOPTOSIS
        if agent_name != "MLP (Adam)":
            # Apoptosis: Olvidar neuronas viejas
            agent.prune_dead_neurons(global_step_counter, max_age=2000)
            
            # Fusión cada 10 episodios
            if episode % 10 == 0:
                agent.fuse_neurons()
            
        metrics['rewards'].append(total_reward)
        metrics['success'].append(1 if info['goal'] else 0)
        
        if agent_name != "MLP (Adam)":
            metrics['params'].append(agent.get_parameter_count())
        else:
            metrics['params'].append(64*6 + 64 + 64*64 + 64 + 64*5 + 5)

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
    
    print("--- 🧠 DUELO V6: FRACTAL + PLASTILINA (Organic Brain) ---")
    
    print("\n>>> Entrenando Motor Cognitivo (VLNN - Organic)...")
    m_vlnn, t_vlnn = train("VLNN (Organic)", env, map_manager)
    
    print("\n>>> Entrenando Red Neuronal (MLP - Baseline)...")
    m_mlp, t_mlp = train("MLP (Adam)", env, map_manager)
    
    # GRAFICAR
    def smooth(data, w=30):
        return np.convolve(data, np.ones(w)/w, mode='valid')

    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    if len(m_vlnn['success']) > 30: plt.plot(smooth(m_vlnn['success']), label='VLNN (Organic)', color='blue', linewidth=2)
    if len(m_mlp['success']) > 30: plt.plot(smooth(m_mlp['success']), label='MLP (Adam)', color='magenta', linestyle='--')
    plt.title("Tasa de Éxito (Organic)")
    plt.axvline(x=200, color='gray', alpha=0.3, linestyle=':')
    plt.axvline(x=500, color='gray', alpha=0.3, linestyle=':')
    plt.xlabel("Episodios")
    plt.ylabel("Win Rate")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(m_vlnn['params'], label='VLNN Organic Params', color='blue')
    plt.plot(m_mlp['params'], label='MLP Fixed Params', color='magenta', linestyle='--')
    plt.title("Respiración Cerebral (Params)")
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('vnn_organic_experiment.png')
    print("\nResultados guardados en 'vnn_organic_experiment.png'")
    
    print(f"\nTIEMPOS: VLNN {t_vlnn:.2f}s | MLP {t_mlp:.2f}s")
    
    results = {
        "experiment": "VLNN Organic Brain (Fractal + Plastilina)",
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
        }
    }
    print("\n--- EXPERIMENT RESULTS (JSON) ---")
    print(json.dumps(results, indent=4))
