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
# 1-6. [TODO EL CÓDIGO ANTERIOR SE MANTIENE IGUAL]
# (MapManager, GridWorld, LidarWrapper, DQN_Agent, CognitiveNeuron, VolumetricCognitiveBrain)
# ==========================================

class MapManager:
    def __init__(self, n_episodes):
        self.map_seeds = [random.randint(0, 1000000) for _ in range(n_episodes)]
    def get_seed(self, episode_idx):
        return self.map_seeds[episode_idx]

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
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        
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

class CognitiveNeuron:
    def __init__(self, neuron_id, center, radius, n_actions=5):
        self.id = neuron_id
        self.c = np.array(center, dtype=np.float32)
        self.r = radius
        self.n_samples = 1
        self.q_values = np.zeros(n_actions) 
        self.pain_accum = 0       
        self.pain_threshold = 8   
        self.last_used = 0

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
            d_neuron.q_values = self.q_values.copy() * 0.98  # Mejor herencia 
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
        
        # Variables para buscar al "Vecino más cercano" (Soft Match)
        nearest_neuron_idx = None
        nearest_dist = float('inf')
        
        for i, neuron in enumerate(self.neurons):
            dist = np.linalg.norm(x - neuron.c)
            
            # Rastrear al vecino más cercano ABSOLUTO
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_neuron_idx = i
            
            # Chequeo de contención (Hard Match - Estoy adentro)
            if dist <= neuron.r:
                if dist < min_dist:
                    min_dist = dist
                    best_idx = i
        
        if best_idx is None:
            # --- CASO DESCONOCIDO: CREACIÓN CON TRANSFERENCIA DE CONOCIMIENTO ---
            new_neuron = CognitiveNeuron(len(self.neurons), x, PARAMS['init_radius'], self.action_dim)
            new_neuron.last_used = current_step
            
            # LA MAGIA: Si hay un vecino, heredamos su conocimiento
            if nearest_neuron_idx is not None:
                neighbor = self.neurons[nearest_neuron_idx]
                
                # Heredamos Q-Values (Intuición) con un poco de duda (0.9)
                # "Si allá funcionó ir a la izquierda, probablemente aquí también"
                new_neuron.q_values = neighbor.q_values.copy() * 0.9
                
                # Heredamos un radio conservador
                new_neuron.r = neighbor.r * 0.9
            
            self.neurons.append(new_neuron)
            best_idx = len(self.neurons) - 1
        else:
            self.neurons[best_idx].last_used = current_step
            
        return best_idx
    
    def check_physics_feasibility(self, state, action_idx):
        """
        SIMULACIÓN DE ESFERAS (MOTOR DE INFERENCIA BASE).
        Antes de preguntar al cerebro si 'quiere' ir, preguntamos a la física si 'puede' ir.
        
        state: [dist_up, dist_right, dist_down, dist_left, goal_x, goal_y] (LIDAR normalizado)
        action_idx: 0=Up, 1=Right, 2=Down, 3=Left, 4=Stay
        
        Returns: True si la acción es físicamente viable, False si chocaría
        """
        # Quedarse quieto siempre es válido
        if action_idx == 4:
            return True
        
        # Mapeo de acciones a sensores LIDAR
        # LIDAR: [Up, Right, Down, Left, goal_x, goal_y]
        lidar_map = {0: 0, 1: 1, 2: 2, 3: 3}  # Up, Right, Down, Left
        
        if action_idx not in lidar_map:
            return True  # Acción desconocida, permitir por defecto
        
        sensor_idx = lidar_map[action_idx]
        distance_normalized = state[sensor_idx]  # 0.0 = pared pegada, 1.0 = libre
        
        # CONOCIMIENTO BASE (NO SE APRENDE, ES FÍSICA):
        # SAFE_MARGIN ajustado para mapas grandes
        # En 24x24: 1 celda = 1/24 ≈ 0.04
        # En 9x9: 1 celda = 1/9 ≈ 0.11
        # Usamos 0.03 para ser seguro incluso en el mapa más grande
        SAFE_MARGIN = 0.03  # Ajustado para mapas 24x24
        
        if distance_normalized < SAFE_MARGIN:
            # Simulación mental: "lanzo esfera virtual -> CHOCA"
            return False
        
        # Simulación mental: "lanzo esfera virtual -> PASA"
        return True

    def get_action(self, state, current_step=0):
        # 1. Obtener la opinión del cerebro (consultar neuronas)
        idx = self.get_active_neuron_idx(state, current_step)
        neuron = self.neurons[idx]
        q_values = neuron.q_values.copy()  # Copiar para no modificar memoria
        
        # 2. CAPA DE SIMULACIÓN FÍSICA (ACTION MASKING)
        # Esto es lo que convertía a la VNN en un MOTOR DE INFERENCIA
        # No "aprende" que las paredes son sólidas - lo SABE
        for action in range(self.action_dim):
            can_pass = self.check_physics_feasibility(state, action)
            
            if not can_pass:
                # La física prohíbe esta acción -> Q = -infinito
                q_values[action] = -float('inf')
        
        # 3. Decisión (solo entre acciones físicamente posibles)
        if random.random() < self.epsilon:
            # Exploración: solo acciones válidas
            valid_actions = [a for a in range(self.action_dim) if q_values[a] > -1e9]
            if not valid_actions:
                return 4  # Atrapado, quedarse quieto
            return random.choice(valid_actions)
        
        # Explotación: mejor acción de las posibles
        if np.all(q_values == 0):
            # Si no hay conocimiento, explorar válidos
            valid_actions = [a for a in range(self.action_dim) if q_values[a] > -1e9]
            if not valid_actions:
                return 4
            return random.choice(valid_actions)
        
        return np.argmax(q_values)

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
        
        if curr_neuron.should_mitose():
            daughters = curr_neuron.mitosis()
            self.neurons.pop(curr_idx)
            for d in daughters:
                d.id = len(self.neurons) 
                self.neurons.append(d)
        
        self.epsilon = max(PARAMS['epsilon_end'], self.epsilon * PARAMS['epsilon_decay'])

    def fuse_neurons(self):
        if len(self.neurons) < 2:
            return
            
        merged_count = 0
        to_remove = set()
        
        for i in range(len(self.neurons)):
            if i in to_remove:
                continue
                
            n1 = self.neurons[i]
            
            for j in range(i + 1, len(self.neurons)):
                if j in to_remove:
                    continue
                    
                n2 = self.neurons[j]
                
                # FUSIÓN CONCEPTUAL: No solo distancia física, sino equivalencia de política
                # Si dos neuronas dictan la MISMA política, son el MISMO CONCEPTO
                # aunque estén lejos en el espacio de sensores
                dist = np.linalg.norm(n1.c - n2.c)
                if dist < (n1.r + n2.r) * 3.0:  # Relajado: permite fusión conceptual remota
                    
                    # Check riguroso: ¿Misma política?
                    best_action_1 = np.argmax(n1.q_values)
                    best_action_2 = np.argmax(n2.q_values)
                    
                    # Si no dictan la misma acción, son conceptos distintos
                    if best_action_1 != best_action_2:
                        continue
                    
                    # Similitud de valores Q (qué tan seguros están)
                    q_diff = np.linalg.norm(n1.q_values - n2.q_values)
                    
                    # Si la acción es la misma y los valores parecidos -> MISMO CONCEPTO
                    if q_diff < 0.3:  # Relajado desde 0.2
                        # Fusión conceptual: crear hiper-esfera que cubre el espacio entre conceptos
                        total_weight = n1.r + n2.r
                        n1.c = (n1.c * n1.r + n2.c * n2.r) / total_weight
                        # Radio ampliado: cubre el hueco entre las dos neuronas (GENERALIZACIÓN)
                        n1.r = max(n1.r, n2.r) + dist * 0.5  # Ampliado para cubrir espacio intermedio
                        n1.q_values = (n1.q_values + n2.q_values) / 2
                        n1.pain_accum = 0
                        
                        to_remove.add(j)
                        merged_count += 1
                        break
        
        for idx in sorted(to_remove, reverse=True):
            self.neurons.pop(idx)
    
    def prune_dead_neurons(self, current_step, max_age=2000):
        initial_count = len(self.neurons)
        self.neurons = [n for n in self.neurons if (current_step - n.last_used) < max_age]

    def get_parameter_count(self):
        return len(self.neurons) * (6 + 1 + 5 + 2)

# ==========================================
# TRAINING
# ==========================================
def train(agent_name, env, map_manager):
    input_dim = 6
    action_dim = 5
    lidar_env = LidarWrapper(env)
    
    if agent_name == "MLP (Adam)":
        agent = DQN_Agent(input_dim, action_dim)
    else:
        agent = VolumetricCognitiveBrain(input_dim, action_dim)
        
    metrics = {'rewards': [], 'success': [], 'params': [], 'collisions': []}
    start_time = time.time()
    
    global_step_counter = 0
    
    for episode in range(PARAMS['n_episodes']):
        # CURRICULUM LEARNING: Aprender conceptos en mapas pequeños, transferir a grandes
        old_size = 0 if episode == 0 else size
        
        if episode < 200: size, obs = 6, 3
        elif episode < 500: size, obs = 9, 10
        else: size, obs = 12, 25
        
        # Reset exploración al cambiar de mapa (shock de adrenalina)
        if agent_name != "MLP (Adam)" and size != old_size and episode > 0:
            agent.epsilon = 0.8  # Re-explorar en mapa nuevo
        
        seed = map_manager.get_seed(episode)
        state_raw = env.reset(seed, size, obs) 
        state = lidar_env.get_lidar_obs()      
        
        total_reward = 0
        done = False

        collision_count = 0  # Track collisions per episode
        
        # FIX: Pasos proporcionales al tamaño del mapa
        # Mapa 6x6 -> 48 pasos, 12x12 -> 96 pasos, 24x24 -> 192 pasos
        current_max_steps = size * 8
        
        for step in range(current_max_steps):
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

        if agent_name != "MLP (Adam)":
            agent.prune_dead_neurons(global_step_counter, max_age=2000)
            if episode % 10 == 0:
                agent.fuse_neurons()
            
        metrics['rewards'].append(total_reward)
        metrics['success'].append(1 if info['goal'] else 0)

        metrics['collisions'].append(collision_count)
        
        if agent_name != "MLP (Adam)":
            metrics['params'].append(agent.get_parameter_count())
        else:
            metrics['params'].append(64*6 + 64 + 64*64 + 64 + 64*5 + 5)

        if episode % 50 == 0:
            avg_win = np.mean(metrics['success'][-50:])
            avg_collisions = np.mean(metrics['collisions'][-50:]) if len(metrics['collisions']) >= 50 else np.mean(metrics['collisions']) if metrics['collisions'] else 0

            print(f"[{agent_name}] Ep {episode} (Map {size}x{size}): WinRate {avg_win:.2f} | Avg Collisions: {avg_collisions:.1f} | Params: {metrics['params'][-1]}")

    dt = time.time() - start_time
    return metrics, dt, agent  # RETORNAMOS EL AGENT ENTRENADO

# ==========================================
# DEPLOYMENT TEST (LA JUNGLA)
# ==========================================
def run_deployment_test(agent_name, agent, env, n_test_episodes=100):
    print(f"\n🌍 FASE 2: DESPLIEGUE EN MUNDO NUEVO (24x24) - {agent_name}")
    
    test_size = 24
    test_obs = 60
    
    metrics = {'rewards': [], 'success': [], 'collisions': [], 'steps_to_goal': []}
    
    if agent_name == "MLP (Adam)":
        agent.model.eval() 
        agent.epsilon = 0.0
    else:
        agent.epsilon = 0.05  # Mínima exploración, máxima explotación de conceptos
    
    global_step = 0
    
    for episode in range(n_test_episodes):
        seed = random.randint(1000000, 9999999)
        lidar_env = LidarWrapper(env)
        
        state_raw = env.reset(seed, test_size, test_obs)
        state = lidar_env.get_lidar_obs()
        
        total_reward = 0
        done = False

        collision_count = 0  # Track collisions per episode
        collision_count = 0  # Contador de choques
        steps_taken = 0  # Contador de pasos
        
        # FIX: Pasos proporcionales al tamaño del mapa
        # 24x24 -> 240 pasos (margen generoso para exploración)
        limit_steps = test_size * 10
        
        for step in range(limit_steps):
            global_step += 1
            
            if agent_name == "MLP (Adam)":
                with torch.no_grad():
                    action = agent.get_action(state)
            else:
                action = agent.get_action(state, global_step)
            
            next_state_raw, reward, done, info = env.step(action)
            next_state = lidar_env.get_lidar_obs()
            
            # Tracking de colisiones
            if info.get('collision', False):
                collision_count += 1
            
            steps_taken += 1
            
            if agent_name != "MLP (Adam)":
                agent.update(state, action, reward, next_state, done, global_step)
            
            state = next_state
            total_reward += reward
            if done: break
        
        if agent_name != "MLP (Adam)":
            agent.prune_dead_neurons(global_step, max_age=5000)  # Conservador: guarda recuerdos
            if episode % 5 == 0: agent.fuse_neurons()
            
        metrics['success'].append(1 if info['goal'] else 0)

        metrics['collisions'].append(collision_count)
        metrics['collisions'].append(collision_count)
        metrics['steps_to_goal'].append(steps_taken if info['goal'] else limit_steps)
        
        if episode % 10 == 0:
            avg_success = np.mean(metrics['success'][-20:]) if len(metrics['success']) >= 20 else np.mean(metrics['success'])
            avg_collisions = np.mean(metrics['collisions'][-20:]) if len(metrics['collisions']) >= 20 else np.mean(metrics['collisions'])
            print(f"   [TEST] Ep {episode}: WinRate {avg_success:.2f} | Avg Collisions: {avg_collisions:.1f}")

    return metrics

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    env = GridWorld()
    map_manager = MapManager(PARAMS['n_episodes'])
    
    print("=== 🧠 FASE 1: ENTRENAMIENTO (ESCUELA) ===\n")
    
    print(">>> Entrenando Motor Cognitivo (VLNN - Organic)...")
    m_vlnn, t_vlnn, agent_vlnn = train("VLNN (Organic)", env, map_manager)
    
    print("\n>>> Entrenando Red Neuronal (MLP - Baseline)...")
    m_mlp, t_mlp, agent_mlp = train("MLP (Adam)", env, map_manager)
    
    print("\n\n=== 🔥 FASE 2: LA JUNGLA (GENERALIZACIÓN) ===")
    
    test_metrics_vlnn = run_deployment_test("VLNN (Organic)", agent_vlnn, env, n_test_episodes=100)
    test_metrics_mlp = run_deployment_test("MLP (Adam)", agent_mlp, env, n_test_episodes=100)
    
    # GRAFICAR TODO
    def smooth(data, w=20):
        if len(data) < w:
            return data
        return np.convolve(data, np.ones(w)/w, mode='valid')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Fase 1: Training
    ax1 = axes[0]
    if len(m_vlnn['success']) > 20: ax1.plot(smooth(m_vlnn['success']), label='VLNN', color='blue', linewidth=2)
    if len(m_mlp['success']) > 20: ax1.plot(smooth(m_mlp['success']), label='MLP', color='magenta', linestyle='--')
    ax1.axvline(x=200, color='gray', alpha=0.3, linestyle=':')
    ax1.axvline(x=500, color='gray', alpha=0.3, linestyle=':')
    ax1.set_title("FASE 1: Training (Escuela)")
    ax1.set_xlabel("Episodios")
    ax1.set_ylabel("Win Rate")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Fase 2: Deployment
    ax2 = axes[1]
    if len(test_metrics_vlnn['success']) > 20: 
        ax2.plot(smooth(test_metrics_vlnn['success']), label='VLNN (Adaptación Viva)', color='blue', linewidth=2)
    if len(test_metrics_mlp['success']) > 20: 
        ax2.plot(smooth(test_metrics_mlp['success']), label='MLP (Congelada)', color='magenta', linestyle='--')
    ax2.set_title("FASE 2: Deployment (Jungla 24x24)")
    ax2.set_xlabel("Episodios en Mundo Nuevo")
    ax2.set_ylabel("Win Rate")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('vnn_complete_experiment.png')
    print("\nResultados guardados en 'vnn_complete_experiment.png'")
    

        # Calcular colisiones por mapa
    vlnn_collisions_6x6 = np.mean(m_vlnn['collisions'][0:200]) if len(m_vlnn['collisions']) > 0 else 0
    vlnn_collisions_9x9 = np.mean(m_vlnn['collisions'][200:500]) if len(m_vlnn['collisions']) > 200 else 0
    vlnn_collisions_12x12 = np.mean(m_vlnn['collisions'][500:800]) if len(m_vlnn['collisions']) > 500 else 0
        
    mlp_collisions_6x6 = np.mean(m_mlp['collisions'][0:200]) if len(m_mlp['collisions']) > 0 else 0
    mlp_collisions_9x9 = np.mean(m_mlp['collisions'][200:500]) if len(m_mlp['collisions']) > 200 else 0
    mlp_collisions_12x12 = np.mean(m_mlp['collisions'][500:800]) if len(m_mlp['collisions']) > 500 else 0

    results = {
        "phase1_training": {
            "VLNN": {
                "time_sec": float(t_vlnn),
                "final_win_rate": float(np.mean(m_vlnn['success'][-50:])),
                "final_params": int(m_vlnn['params'][-1]),
                "collisions_6x6": float(vlnn_collisions_6x6),
                "collisions_9x9": float(vlnn_collisions_9x9),
                "collisions_12x12": float(vlnn_collisions_12x12)
            },
            "MLP": {
                "time_sec": float(t_mlp),
                "final_win_rate": float(np.mean(m_mlp['success'][-50:])),
                "final_params": int(m_mlp['params'][-1]),
                "collisions_6x6": float(mlp_collisions_6x6),
                "collisions_9x9": float(mlp_collisions_9x9),
                "collisions_12x12": float(mlp_collisions_12x12)
            }
        },
        "phase2_deployment": {
            "VLNN": {
            "final_win_rate": float(np.mean(test_metrics_vlnn['success'][-20:])),
            "avg_collisions": float(np.mean(test_metrics_vlnn['collisions'])),
            "avg_steps_to_goal": float(np.mean([s for i, s in enumerate(test_metrics_vlnn['steps_to_goal']) if test_metrics_vlnn['success'][i] == 1] or [0]))
        },
            "MLP": {
            "final_win_rate": float(np.mean(test_metrics_mlp['success'][-20:])),
            "avg_collisions": float(np.mean(test_metrics_mlp['collisions'])),
            "avg_steps_to_goal": float(np.mean([s for i, s in enumerate(test_metrics_mlp['steps_to_goal']) if test_metrics_mlp['success'][i] == 1] or [0]))
        }
        }
    }
    
    print("\n=== RESULTADOS FINALES (JSON) ===")
    print(json.dumps(results, indent=4))
