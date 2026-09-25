import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import time
import json

# ==========================================
# 0. CONFIGURACIÓN ROBUSTA
# ==========================================
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

PARAMS = {
    'grid_size': 12,      # Tamaño final (Curriculum lo modificará)
    'n_episodes': 1000,   # Total episodios
    'max_steps': 50,
    'lr': 0.005,          
    'gamma': 0.99,
    'hidden_dim': 64,
    'vnn_concepts': 32,   
    'n_runs': 1           
}

# ==========================================
# 1. GESTOR DE MAPAS (FAIRNESS)
# ==========================================
class MapManager:
    """Garantiza que todos los agentes vean EXACTAMENTE la misma secuencia de mapas."""
    def __init__(self, n_episodes):
        self.map_seeds = [random.randint(0, 1000000) for _ in range(n_episodes)]
    
    def get_seed(self, episode_idx):
        return self.map_seeds[episode_idx]

# ==========================================
# 2. EL ENTORNO (GRIDWORLD) V2
# ==========================================
class GridWorld:
    def __init__(self, size=12, n_obstacles=25):
        self.size = size
        self.n_obstacles = n_obstacles
        self.map = np.zeros((size, size))
        self.start = (0, 0)
        self.agent_pos = (0, 0)
        self.goal_pos = (size-1, size-1)
        self.obstacles = []
    
    def seed_map(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        
        self.map = np.zeros((self.size, self.size)) # Resize map if size changed
        self.obstacles = []
        
        candidates = []
        for i in range(self.size):
            for j in range(self.size):
                if (i,j) != (0,0) and (i,j) != (self.size-1, self.size-1):
                    candidates.append((i,j))
        
        # Ensure we don't sample more than available candidates
        n_obs = min(self.n_obstacles, len(candidates))
        self.obstacles = random.sample(candidates, n_obs)
        for obs in self.obstacles:
            self.map[obs[0], obs[1]] = 1
            
        self.goal_pos = (self.size-1, self.size-1)
        self.agent_pos = (0, 0)
        
    def reset(self, episode_seed):
        self.seed_map(episode_seed)
        return self.get_obs()

    def get_obs(self):
        patch = np.ones((5, 5)) 
        ax, ay = self.agent_pos
        
        for i in range(-2, 3):
            for j in range(-2, 3):
                nx, ny = ax + i, ay + j
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    if (nx, ny) in self.obstacles:
                        patch[i+2, j+2] = 1 
                    else:
                        patch[i+2, j+2] = 0 
                else:
                    patch[i+2, j+2] = 1 
        
        gx, gy = self.goal_pos
        dx, dy = gx - ax, gy - ay
        dist = np.sqrt(dx**2 + dy**2) + 1e-5
        goal_vec = np.array([dx/dist, dy/dist])
        
        return np.concatenate([patch.flatten(), goal_vec])

    def step(self, action):
        moves = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1), 4: (0, 0)}
        dx, dy = moves[action]
        nx, ny = self.agent_pos[0] + dx, self.agent_pos[1] + dy
        
        # Calcular distancia ANTERIOR al objetivo
        gx, gy = self.goal_pos
        ax, ay = self.agent_pos
        dist_old = np.sqrt((gx - ax)**2 + (gy - ay)**2)
        
        reward = -0.01 
        done = False
        collision = False
        reached_goal = False
        
        if not (0 <= nx < self.size and 0 <= ny < self.size) or ((nx, ny) in self.obstacles):
            reward = -1.0 # Castigo fuerte por chocar
            collision = True
        else:
            self.agent_pos = (nx, ny)
            
            # SHAPED REWARD
            dist_new = np.sqrt((gx - nx)**2 + (gy - ny)**2)
            reward += (dist_old - dist_new) * 2.0 
            
            if self.agent_pos == self.goal_pos:
                reward = 10.0 
                done = True
                reached_goal = True
                
        return self.get_obs(), reward, done, {'collision': collision, 'goal': reached_goal}

# ==========================================
# 3. LOS MODELOS
# ==========================================

class MLPPolicy(nn.Module):
    def __init__(self, input_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, PARAMS['hidden_dim']),
            nn.ReLU(),
            nn.Linear(PARAMS['hidden_dim'], PARAMS['hidden_dim']),
            nn.ReLU(),
            nn.Linear(PARAMS['hidden_dim'], action_dim),
            nn.Softmax(dim=-1)
        )
    def forward(self, x):
        return self.net(x)

class VNNLayer(nn.Module):
    def __init__(self, input_dim, n_concepts, mode='soft'):
        super().__init__()
        self.mode = mode
        self.centers = nn.Parameter(torch.randn(n_concepts, input_dim))
        self.radii = nn.Parameter(torch.ones(n_concepts) * 1.5) 

    def forward(self, x):
        if x.dim() == 1: x = x.unsqueeze(0)
        dists = torch.cdist(x, self.centers, p=2) 
        
        if self.mode == 'soft':
            return torch.exp(- (dists**2) / (2 * self.radii**2 + 1e-6))
        elif self.mode == 'hard':
            return torch.sigmoid(20 * (self.radii - dists))

class VNNPolicy(nn.Module):
    def __init__(self, input_dim, action_dim, mode='soft'):
        super().__init__()
        self.vnn = VNNLayer(input_dim, PARAMS['vnn_concepts'], mode=mode)
        self.head = nn.Sequential(
            nn.Linear(PARAMS['vnn_concepts'], action_dim),
            nn.Softmax(dim=-1)
        )
    def forward(self, x):
        features = self.vnn(x)
        return self.head(features)

# ==========================================
# 4. ENTRENAMIENTO CON CURRICULUM
# ==========================================
def train_agent(agent_type, env, map_manager):
    input_dim = 25 + 2
    action_dim = 5
    
    if agent_type == 'MLP': policy = MLPPolicy(input_dim, action_dim)
    elif agent_type == 'VNN-Soft': policy = VNNPolicy(input_dim, action_dim, mode='soft')
    elif agent_type == 'VNN-Hard': policy = VNNPolicy(input_dim, action_dim, mode='hard')
        
    optimizer = optim.Adam(policy.parameters(), lr=PARAMS['lr'])
    
    metrics = {
        'rewards': [],
        'success': [],
        'steps': [],
        'collisions': []
    }
    
    start_time = time.time()
    
    for episode in range(PARAMS['n_episodes']):
        # --- CURRICULUM LEARNING (TU IDEA) ---
        # Fase 1 (0-300): Mapa Chico (6x6), Fácil
        # Fase 2 (300-600): Mapa Mediano (9x9), Intermedio
        # Fase 3 (600+): Mapa Grande (12x12), Difícil (Test Real)
        
        if episode < 300:
            current_size = 6
            n_obs = 3 
        elif episode < 600:
            current_size = 9
            n_obs = 10
        else:
            current_size = 12
            n_obs = 25 
            
        # Reconfiguramos el entorno al vuelo
        env.size = current_size
        env.n_obstacles = n_obs
        
        # FAIRNESS: Usar semilla predeterminada para este episodio
        seed = map_manager.get_seed(episode)
        state = env.reset(seed)
        
        log_probs = []
        rewards = []
        
        ep_collision = 0
        ep_steps = 0
        ep_success = 0
        
        for step in range(PARAMS['max_steps']):
            state_t = torch.FloatTensor(state)
            probs = policy(state_t)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            
            next_state, reward, done, info = env.step(action.item())
            
            log_probs.append(dist.log_prob(action))
            rewards.append(reward)
            state = next_state
            
            ep_steps += 1
            if info['collision']: ep_collision = 1
            if info['goal']: ep_success = 1
            
            if done: break
            
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + PARAMS['gamma'] * G
            returns.insert(0, G)
        returns = torch.tensor(returns)
        
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        else:
            returns = returns - returns.mean()

        policy_loss = []
        for log_prob, G in zip(log_probs, returns):
            policy_loss.append(-log_prob * G)
        
        optimizer.zero_grad()
        if policy_loss:
            loss = torch.stack(policy_loss).sum()
            loss.backward()
            optimizer.step()
        
        metrics['rewards'].append(sum(rewards))
        metrics['success'].append(ep_success)
        metrics['steps'].append(ep_steps)
        metrics['collisions'].append(ep_collision)
        
        if episode % 100 == 0:
            print(f"[{agent_type}] Ep {episode} (Sz:{current_size}): WinRate {np.mean(metrics['success'][-50:]):.2f}")

    train_time = time.time() - start_time
    return metrics, train_time

# ==========================================
# 5. EJECUCIÓN Y COMPARACIÓN
# ==========================================
if __name__ == "__main__":
    env = GridWorld(size=6) # Init con tamaño pequeño
    map_manager = MapManager(PARAMS['n_episodes']) 

    print("--- INICIANDO EXPERIMENTO V2: CURRICULUM LEARNING ---")

    print("\n1. Entrenando MLP...")
    m_mlp, t_mlp = train_agent('MLP', env, map_manager)

    print("\n2. Entrenando VNN-Soft...")
    m_soft, t_soft = train_agent('VNN-Soft', env, map_manager)

    print("\n3. Entrenando VNN-Hard (Bolas Duras)...")
    m_hard, t_hard = train_agent('VNN-Hard', env, map_manager)

    # ==========================================
    # 6. VISUALIZACIÓN
    # ==========================================
    def smooth(data, window=50):
        return np.convolve(data, np.ones(window)/window, mode='valid')

    plt.figure(figsize=(15, 12))

    # 1. Tasa de Éxito
    plt.subplot(3, 1, 1)
    if len(m_mlp['success']) > 50: plt.plot(smooth(m_mlp['success']), label='MLP', color='magenta', alpha=0.6)
    if len(m_soft['success']) > 50: plt.plot(smooth(m_soft['success']), label='VNN-Soft', color='orange', alpha=0.8)
    if len(m_hard['success']) > 50: plt.plot(smooth(m_hard['success']), label='VNN-Hard', color='blue', linewidth=2)
    # Dibujar lineas de fases
    plt.axvline(x=300, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=600, color='k', linestyle='--', alpha=0.3)
    plt.text(150, 0.9, 'Fase 1 (6x6)', ha='center')
    plt.text(450, 0.9, 'Fase 2 (9x9)', ha='center')
    plt.text(800, 0.9, 'Fase 3 (12x12)', ha='center')
    
    plt.title('Tasa de Éxito con Curriculum Learning')
    plt.ylabel('Success Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. Eficiencia (Pasos)
    plt.subplot(3, 1, 2)
    if len(m_mlp['steps']) > 50: plt.plot(smooth(m_mlp['steps']), label='MLP', color='magenta', alpha=0.6)
    if len(m_soft['steps']) > 50: plt.plot(smooth(m_soft['steps']), label='VNN-Soft', color='orange', alpha=0.8)
    if len(m_hard['steps']) > 50: plt.plot(smooth(m_hard['steps']), label='VNN-Hard', color='blue', linewidth=2)
    plt.axvline(x=300, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=600, color='k', linestyle='--', alpha=0.3)
    plt.title('Pasos por Episodio (Eficiencia)')
    plt.ylabel('Pasos')
    plt.grid(True, alpha=0.3)

    # 3. Barras Finales
    plt.subplot(3, 1, 3)
    labels = ['MLP', 'VNN-Soft', 'VNN-Hard']
    # Promedio de los últimos 200 episodios (Fase Difícil)
    final_acc = [
        np.mean(m_mlp['success'][-200:]), 
        np.mean(m_soft['success'][-200:]), 
        np.mean(m_hard['success'][-200:])
    ]
    colors = ['magenta', 'orange', 'blue']
    plt.bar(labels, final_acc, color=colors, alpha=0.7)
    plt.title('Performance en Fase Difícil (Test Final)')
    plt.ylabel('Success Rate (12x12)')
    plt.ylim(0, 1.0)
    for i, v in enumerate(final_acc):
        plt.text(i, v + 0.01, f"{v:.2f}", ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig('vnn_curriculum_comparison.png')
    print("\n¡Resultados guardados en 'vnn_curriculum_comparison.png'!")
    # plt.show()

    results = {
        "experiment": "GridWorld V2: Curriculum Learning + Shaped Rewards",
        "phases": ["small_6x6", "medium_9x9", "large_12x12"],
        "metrics": {
            "MLP": {
                "final_success_rate": float(final_acc[0]),
                "training_time_sec": float(t_mlp)
            },
            "VNN-Soft": {
                "final_success_rate": float(final_acc[1]),
                "training_time_sec": float(t_soft)
            },
            "VNN-Hard": {
                "final_success_rate": float(final_acc[2]),
                "training_time_sec": float(t_hard)
            }
        },
        "config": PARAMS
    }

    print("\n--- EXPERIMENT RESULTS (JSON) ---")
    print(json.dumps(results, indent=4))
