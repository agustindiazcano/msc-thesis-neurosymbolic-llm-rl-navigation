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
# 0. CONFIGURACIÓN Y SEMILLAS
# ==========================================
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

PARAMS = {
    'grid_size': 12,
    'n_episodes': 500,
    'max_steps': 50,
    'lr': 0.01,
    'gamma': 0.99,
    'hidden_dim': 64,  # Para MLP
    'vnn_concepts': 32 # Cantidad de "Esferas" (Prototipos)
}

# ==========================================
# 1. EL ENTORNO (GRIDWORLD)
# ==========================================
class GridWorld:
    def __init__(self, size=12, n_obstacles=20):
        self.size = size
        self.n_obstacles = n_obstacles
        self.map = np.zeros((size, size))
        self.start = (0, 0)
        self.agent_pos = (0, 0)
        self.goal_pos = (size-1, size-1)
        self.obstacles = []
        self.reset_map()

    def reset_map(self):
        """Genera un mapa nuevo con obstáculos aleatorios"""
        self.map.fill(0)
        self.obstacles = []
        # Bordes
        # (Opcional: el código maneja límites, pero los obstaculos añaden dificultad)
        for _ in range(self.n_obstacles):
            ox, oy = np.random.randint(0, self.size, 2)
            if (ox, oy) != self.start and (ox, oy) != self.goal_pos:
                self.obstacles.append((ox, oy))
                self.map[ox, oy] = 1 # 1 = Pared
        
        self.goal_pos = (self.size-1, self.size-1)
        # Asegurar que start y goal no son obstáculos
        if self.goal_pos in self.obstacles: self.obstacles.remove(self.goal_pos)
        
    def reset(self):
        self.agent_pos = (0, 0)
        return self.get_obs()

    def get_obs(self):
        """
        Retorna:
        1. Local Patch (5x5) centrado en agente. 1=Pared, 0=Libre/Goal.
        2. Vector relativo al goal (normalizado).
        """
        patch = np.ones((5, 5)) # Relleno con paredes por defecto (fuera de mapa)
        ax, ay = self.agent_pos
        
        for i in range(-2, 3):
            for j in range(-2, 3):
                nx, ny = ax + i, ay + j
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    if (nx, ny) in self.obstacles:
                        patch[i+2, j+2] = 1 # Pared
                    else:
                        patch[i+2, j+2] = 0 # Libre
                else:
                    patch[i+2, j+2] = 1 # Fuera de mapa es pared
        
        # Goal Vector
        gx, gy = self.goal_pos
        dx, dy = gx - ax, gy - ay
        dist = np.sqrt(dx**2 + dy**2) + 1e-5
        goal_vec = np.array([dx/dist, dy/dist])
        
        return np.concatenate([patch.flatten(), goal_vec])

    def step(self, action):
        # 0: Up, 1: Right, 2: Down, 3: Left, 4: Stay
        moves = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1), 4: (0, 0)}
        dx, dy = moves[action]
        nx, ny = self.agent_pos[0] + dx, self.agent_pos[1] + dy
        
        reward = -0.01 # Step penalty
        done = False
        
        # Check collision or out of bounds
        if not (0 <= nx < self.size and 0 <= ny < self.size) or ((nx, ny) in self.obstacles):
            reward = -1.0 # Collision
            # No move
        else:
            self.agent_pos = (nx, ny)
            if self.agent_pos == self.goal_pos:
                reward = 1.0 # Goal!
                done = True
                
        return self.get_obs(), reward, done

# ==========================================
# 2. LOS MODELOS (MLP vs VNN)
# ==========================================

# A. Standard MLP (Blackbox)
class MLPPolicy(nn.Module):
    def __init__(self, input_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, PARAMS['hidden_dim']),
            nn.ReLU(),
            nn.Linear(PARAMS['hidden_dim'], action_dim),
            nn.Softmax(dim=-1)
        )
    def forward(self, x):
        return self.net(x)

# B. VNN Layer (Neurosymbolic Core)
class VNNLayer(nn.Module):
    """
    Esta capa simula tus esferas.
    En lugar de aprender features oscuros, aprende PROTOTIPOS (Centros).
    """
    def __init__(self, input_dim, n_concepts, mode='soft'):
        super().__init__()
        self.mode = mode
        self.n_concepts = n_concepts
        # Centros inicializados aleatoriamente (conceptos latentes)
        # En una VNN real, estos se aprenderían con Bayes/Heurística
        self.centers = nn.Parameter(torch.randn(n_concepts, input_dim))
        self.radii = nn.Parameter(torch.ones(n_concepts) * 2.0) # Radios aprendibles

    def forward(self, x):
        # x: (batch, input_dim)
        # Calcular distancias a todos los centros: ||x - c||^2
        # Expansion para broadcast: (batch, 1, dim) - (1, concepts, dim)
        if x.dim() == 1: x = x.unsqueeze(0)
        
        dists = torch.cdist(x, self.centers, p=2) # (batch, concepts)
        
        if self.mode == 'soft':
            # Soft: RBF Activation (Gaussian) -> Simula Fuzzy Logic
            # phi(x) = exp(-dist^2 / r^2)
            return torch.exp(- (dists**2) / (self.radii**2 + 1e-6))
        
        elif self.mode == 'hard':
            # Hard: Boolean Logic (Dentro o Fuera)
            # phi(x) = 1 si dist < r, else 0
            # Usamos sigmoide steep para simular hard en entrenamiento (diferenciable)
            # En inferencia real sería step function.
            return torch.sigmoid(10 * (self.radii - dists))

class VNNPolicy(nn.Module):
    def __init__(self, input_dim, action_dim, mode='soft'):
        super().__init__()
        # Capa 1: VNN (Geometric Feature Extractor)
        self.vnn = VNNLayer(input_dim, PARAMS['vnn_concepts'], mode=mode)
        # Capa 2: Linear Policy (Logic Combination)
        # La politica es solo una combinación lineal de los conceptos activados
        self.head = nn.Sequential(
            nn.Linear(PARAMS['vnn_concepts'], action_dim),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x):
        features = self.vnn(x)
        return self.head(features)

# ==========================================
# 3. ENTRENAMIENTO (REINFORCE)
# ==========================================
def train_agent(agent_type, env):
    input_dim = 25 + 2 # 5x5 patch + 2 goal vec
    action_dim = 5
    
    if agent_type == 'MLP':
        policy = MLPPolicy(input_dim, action_dim)
    elif agent_type == 'VNN-Soft':
        policy = VNNPolicy(input_dim, action_dim, mode='soft')
    elif agent_type == 'VNN-Hard':
        policy = VNNPolicy(input_dim, action_dim, mode='hard')
        
    optimizer = optim.Adam(policy.parameters(), lr=PARAMS['lr'])
    
    rewards_history = []
    success_rate = deque(maxlen=50)
    
    start_time = time.time()
    
    for episode in range(PARAMS['n_episodes']):
        state = env.reset()
        log_probs = []
        rewards = []
        
        # Generar nuevo mapa cada 10 episodios para forzar generalización
        if episode % 10 == 0: env.reset_map()
        
        for step in range(PARAMS['max_steps']):
            state_t = torch.FloatTensor(state)
            probs = policy(state_t)
            
            # Samplear acción
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            
            next_state, reward, done = env.step(action.item())
            
            log_probs.append(dist.log_prob(action))
            rewards.append(reward)
            state = next_state
            
            if done: break
            
        # Calcular Retornos (Monte Carlo)
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + PARAMS['gamma'] * G
            returns.insert(0, G)
        returns = torch.tensor(returns)
        # Normalizar retornos (estabilidad)
        if returns.std() > 0:
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)
            
        # Policy Gradient Update
        policy_loss = []
        for log_prob, G in zip(log_probs, returns):
            policy_loss.append(-log_prob * G)
        
        optimizer.zero_grad()
        if policy_loss:
            policy_loss = torch.stack(policy_loss).sum()
            policy_loss.backward()
            optimizer.step()
        
        total_reward = sum(rewards)
        rewards_history.append(total_reward)
        success_rate.append(1 if total_reward > 0 else 0)
        
        if episode % 50 == 0:
            print(f"[{agent_type}] Ep {episode}: Reward {total_reward:.2f} | WinRate {np.mean(success_rate):.2f}")

    train_time = time.time() - start_time
    return rewards_history, np.mean(success_rate), train_time

# ==========================================
# 4. EJECUCIÓN COMPARATIVA
# ==========================================
if __name__ == "__main__":
    env = GridWorld(size=PARAMS['grid_size'])

    print("--- INICIANDO EXPERIMENTO GRIDWORLD VNN vs MLP ---")

    print("\n1. Entrenando MLP (Baseline)...")
    r_mlp, win_mlp, t_mlp = train_agent('MLP', env)

    print("\n2. Entrenando VNN-Soft (Fuzzy Logic)...")
    r_soft, win_soft, t_soft = train_agent('VNN-Soft', env)

    print("\n3. Entrenando VNN-Hard (Boolean Spheres)...")
    r_hard, win_hard, t_hard = train_agent('VNN-Hard', env)

    # ==========================================
    # 5. VISUALIZACIÓN
    # ==========================================
    def moving_average(a, n=20):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    plt.figure(figsize=(14, 6))

    # Curvas de Aprendizaje
    plt.subplot(1, 2, 1)
    if len(r_mlp) > 20: plt.plot(moving_average(r_mlp), label=f'MLP (Win: {win_mlp:.2f})', color='magenta', linestyle='--')
    if len(r_soft) > 20: plt.plot(moving_average(r_soft), label=f'VNN-Soft (Win: {win_soft:.2f})', color='orange')
    if len(r_hard) > 20: plt.plot(moving_average(r_hard), label=f'VNN-Hard (Win: {win_hard:.2f})', color='blue', linewidth=2)
    plt.title('Curvas de Aprendizaje (Reward Promedio)')
    plt.xlabel('Episodios')
    plt.ylabel('Retorno Acumulado')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Barras de Eficiencia
    plt.subplot(1, 2, 2)
    models = ['MLP', 'VNN-Soft', 'VNN-Hard']
    times = [t_mlp, t_soft, t_hard]
    colors = ['magenta', 'orange', 'blue']
    plt.bar(models, times, color=colors, alpha=0.7)
    plt.title('Tiempo de Entrenamiento (s)')
    plt.ylabel('Segundos')
    for i, v in enumerate(times):
        plt.text(i, v + 0.1, f"{v:.1f}s", ha='center')

    plt.tight_layout()
    plt.savefig('vnn_gridworld_comparison.png')
    print("\nResultados guardados en 'vnn_gridworld_comparison.png'")
    # plt.show()
    
    # JSON RESULTS
    results_json = {
        "experiment": "GridWorld Survival (REINFORCE)",
        "metrics": {
            "MLP": {
                "final_win_rate": float(win_mlp),
                "training_time_sec": float(t_mlp)
            },
            "VNN-Soft": {
                "final_win_rate": float(win_soft),
                "training_time_sec": float(t_soft)
            },
            "VNN-Hard": {
                "final_win_rate": float(win_hard),
                "training_time_sec": float(t_hard)
            }
        },
        "config": PARAMS
    }

    print("\n--- EXPERIMENT RESULTS (JSON) ---")
    print(json.dumps(results_json, indent=4))
