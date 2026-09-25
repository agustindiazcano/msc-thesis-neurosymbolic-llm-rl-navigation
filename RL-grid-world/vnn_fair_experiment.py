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
    'grid_size': 12,
    'n_episodes': 1000,   # Aumentado para ver convergencia real
    'max_steps': 50,
    'lr': 0.005,          # Un poco más bajo para estabilidad
    'gamma': 0.99,
    'hidden_dim': 64,
    'vnn_concepts': 32,   # Cantidad de esferas/prototipos
    'n_runs': 1           # Cantidad de runs independientes (seeds globales)
}

# ==========================================
# 1. GESTOR DE MAPAS (FAIRNESS)
# ==========================================
class MapManager:
    """Garantiza que todos los agentes vean EXACTAMENTE la misma secuencia de mapas."""
    def __init__(self, n_episodes):
        # Pre-generamos las seeds para cada episodio
        self.map_seeds = [random.randint(0, 1000000) for _ in range(n_episodes)]
    
    def get_seed(self, episode_idx):
        return self.map_seeds[episode_idx]

# ==========================================
# 2. EL ENTORNO (GRIDWORLD)
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
        """Reinicia el mapa usando una semilla específica."""
        random.seed(seed)
        np.random.seed(seed)
        
        self.map.fill(0)
        self.obstacles = []
        
        # Generar obstáculos aleatorios pero deterministas por seed
        candidates = []
        for i in range(self.size):
            for j in range(self.size):
                if (i,j) != (0,0) and (i,j) != (self.size-1, self.size-1):
                    candidates.append((i,j))
        
        self.obstacles = random.sample(candidates, self.n_obstacles)
        for obs in self.obstacles:
            self.map[obs] = 1
            
        self.goal_pos = (self.size-1, self.size-1)
        self.agent_pos = (0, 0)
        
    def reset(self, episode_seed):
        self.seed_map(episode_seed)
        return self.get_obs()

    def get_obs(self):
        # Observación: Patch 5x5 + Vector Goal
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
            # No movemos al agente si choca
        else:
            self.agent_pos = (nx, ny)
            
            # --- NUEVO: RECOMPENSA POR ACERCARSE (SHAPED REWARD) ---
            # Calcular distancia NUEVA
            dist_new = np.sqrt((gx - nx)**2 + (gy - ny)**2)
            
            # Si me acerqué, premio. Si me alejé, castigo leve.
            # Multiplicamos por 2.0 para que sea una señal fuerte
            reward += (dist_old - dist_new) * 2.0 
            
            if self.agent_pos == self.goal_pos:
                reward = 10.0 # ¡PREMIO MAYOR! (Antes era 1.0, ahora 10 para motivar)
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
            nn.Linear(PARAMS['hidden_dim'], PARAMS['hidden_dim']), # Capa extra para darle chance
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
            # RBF Gaussiana
            return torch.exp(- (dists**2) / (2 * self.radii**2 + 1e-6))
        elif self.mode == 'hard':
            # Simulación Diferenciable de Hard Sphere (Sigmoide muy empinada)
            # Esto emula tu lógica booleana: Si dist < radio -> 1, sino -> 0
            # Beta alto (20) hace la transición casi vertical
            return torch.sigmoid(20 * (self.radii - dists))

class VNNPolicy(nn.Module):
    def __init__(self, input_dim, action_dim, mode='soft'):
        super().__init__()
        self.vnn = VNNLayer(input_dim, PARAMS['vnn_concepts'], mode=mode)
        # La "Policy" es lineal sobre los conceptos geométricos activados
        self.head = nn.Sequential(
            nn.Linear(PARAMS['vnn_concepts'], action_dim),
            nn.Softmax(dim=-1)
        )
    def forward(self, x):
        features = self.vnn(x)
        return self.head(features)

# ==========================================
# 4. ENTRENAMIENTO (REINFORCE + BASELINE)
# ==========================================
def train_agent(agent_type, env, map_manager):
    input_dim = 25 + 2
    action_dim = 5
    
    if agent_type == 'MLP': policy = MLPPolicy(input_dim, action_dim)
    elif agent_type == 'VNN-Soft': policy = VNNPolicy(input_dim, action_dim, mode='soft')
    elif agent_type == 'VNN-Hard': policy = VNNPolicy(input_dim, action_dim, mode='hard')
        
    optimizer = optim.Adam(policy.parameters(), lr=PARAMS['lr'])
    
    # Métricas
    metrics = {
        'rewards': [],
        'success': [],
        'steps': [],
        'collisions': []
    }
    
    start_time = time.time()
    
    for episode in range(PARAMS['n_episodes']):
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
            
        # --- REINFORCE con Baseline ---
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + PARAMS['gamma'] * G
            returns.insert(0, G)
        returns = torch.tensor(returns)
        
        # Baseline simple: Normalizar retornos (reduce varianza)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        else:
            returns = returns - returns.mean() # Si solo hubo un paso

        policy_loss = []
        for log_prob, G in zip(log_probs, returns):
            policy_loss.append(-log_prob * G)
        
        optimizer.zero_grad()
        if policy_loss:
            loss = torch.stack(policy_loss).sum()
            loss.backward()
            optimizer.step()
        
        # Guardar métricas
        metrics['rewards'].append(sum(rewards))
        metrics['success'].append(ep_success)
        metrics['steps'].append(ep_steps)
        metrics['collisions'].append(ep_collision)
        
        if episode % 100 == 0:
            print(f"[{agent_type}] Ep {episode}: WinRate {np.mean(metrics['success'][-50:]):.2f}")

    train_time = time.time() - start_time
    return metrics, train_time

# ==========================================
# 5. EJECUCIÓN Y COMPARACIÓN
# ==========================================
if __name__ == "__main__":
    env = GridWorld(size=PARAMS['grid_size'])
    map_manager = MapManager(PARAMS['n_episodes']) # ¡La clave de la justicia!

    print("--- INICIANDO DUELO JUSTO (VNN vs MLP) ---")

    print("\n1. Entrenando MLP...")
    m_mlp, t_mlp = train_agent('MLP', env, map_manager)

    print("\n2. Entrenando VNN-Soft...")
    m_soft, t_soft = train_agent('VNN-Soft', env, map_manager)

    print("\n3. Entrenando VNN-Hard (Bolas Duras)...")
    m_hard, t_hard = train_agent('VNN-Hard', env, map_manager)

    # ==========================================
    # 6. VISUALIZACIÓN PROFESIONAL
    # ==========================================
    def smooth(data, window=50):
        return np.convolve(data, np.ones(window)/window, mode='valid')

    plt.figure(figsize=(15, 10))

    # 1. Tasa de Éxito (Lo más importante)
    plt.subplot(2, 2, 1)
    if len(m_mlp['success']) > 50: plt.plot(smooth(m_mlp['success']), label='MLP', color='magenta', alpha=0.6)
    if len(m_soft['success']) > 50: plt.plot(smooth(m_soft['success']), label='VNN-Soft', color='orange', alpha=0.8)
    if len(m_hard['success']) > 50: plt.plot(smooth(m_hard['success']), label='VNN-Hard', color='blue', linewidth=2)
    plt.title('Tasa de Éxito (Moving Avg)')
    plt.ylabel('Probabilidad de llegar')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. Eficiencia (Pasos) - Solo cuentan los éxitos para no ensuciar
    # (Aquí graficamos crudo suavizado de todos, penalizando failures con max_steps implícito)
    plt.subplot(2, 2, 2)
    if len(m_mlp['steps']) > 50: plt.plot(smooth(m_mlp['steps']), label='MLP', color='magenta', alpha=0.6)
    if len(m_soft['steps']) > 50: plt.plot(smooth(m_soft['steps']), label='VNN-Soft', color='orange', alpha=0.8)
    if len(m_hard['steps']) > 50: plt.plot(smooth(m_hard['steps']), label='VNN-Hard', color='blue', linewidth=2)
    plt.title('Eficiencia (Pasos por Episodio)')
    plt.ylabel('Pasos (Menor es mejor)')
    plt.grid(True, alpha=0.3)

    # 3. Seguridad (Colisiones)
    plt.subplot(2, 2, 3)
    if len(m_mlp['collisions']) > 50: plt.plot(smooth(m_mlp['collisions']), label='MLP', color='magenta', alpha=0.6)
    if len(m_soft['collisions']) > 50: plt.plot(smooth(m_soft['collisions']), label='VNN-Soft', color='orange', alpha=0.8)
    if len(m_hard['collisions']) > 50: plt.plot(smooth(m_hard['collisions']), label='VNN-Hard', color='blue', linewidth=2)
    plt.title('Tasa de Colisiones')
    plt.ylabel('Colisiones')
    plt.grid(True, alpha=0.3)

    # 4. Barras Resumen
    plt.subplot(2, 2, 4)
    labels = ['MLP', 'VNN-Soft', 'VNN-Hard']
    final_acc = [np.mean(m_mlp['success'][-100:]), np.mean(m_soft['success'][-100:]), np.mean(m_hard['success'][-100:])]
    colors = ['magenta', 'orange', 'blue']
    plt.bar(labels, final_acc, color=colors, alpha=0.7)
    plt.title('Precisión Final (Últimos 100 eps)')
    plt.ylabel('Success Rate')
    plt.ylim(0, 1.0)

    for i, v in enumerate(final_acc):
        plt.text(i, v + 0.01, f"{v:.2f}", ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig('vnn_fair_comparison.png')
    print("\n¡Resultados guardados! Revisá 'vnn_fair_comparison.png'")
    # plt.show() # Commented out for headless execution

    # JSON Results (Final Artifact)
    results = {
        "experiment": "GridWorld Fair Comparison (Robust REINFORCE)",
        "metrics": {
            "MLP": {
                "final_success_rate": float(np.mean(m_mlp['success'][-100:])),
                "training_time_sec": float(t_mlp)
            },
            "VNN-Soft": {
                "final_success_rate": float(np.mean(m_soft['success'][-100:])),
                "training_time_sec": float(t_soft)
            },
            "VNN-Hard": {
                "final_success_rate": float(np.mean(m_hard['success'][-100:])),
                "training_time_sec": float(t_hard)
            }
        },
        "config": PARAMS
    }

    print("\n--- EXPERIMENT RESULTS (JSON) ---")
    print(json.dumps(results, indent=4))
