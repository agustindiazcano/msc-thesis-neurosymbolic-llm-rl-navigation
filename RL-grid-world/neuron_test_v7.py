import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- 1. CONFIGURACIÓN DEL LABERINTO VIVO ---
GRID_SIZE = 20
TARGET = np.array([18.0, 18.0])
START = np.array([2.0, 2.0])

# Definimos obstáculos iniciales (Muros estáticos)
static_walls = [
    {'x': [10, 10], 'y': [0, 15]}, # Muro central largo
    {'x': [5, 15], 'y': [10, 10]}, # Muro horizontal
]

# Definimos obstáculos móviles (Puertas que se cierran)
class DynamicDoor:
    def __init__(self, pos_open, pos_closed, speed=0.1):
        self.pos = np.array(pos_open, dtype=float)
        self.target = np.array(pos_closed, dtype=float)
        self.origin = np.array(pos_open, dtype=float)
        self.speed = speed
        self.closing = True
        self.radius = 1.5 # Tamaño del obstáculo

    def update(self):
        # Moverse entre abierto y cerrado
        dest = self.target if self.closing else self.origin
        direction = dest - self.pos
        dist = np.linalg.norm(direction)
        
        if dist < self.speed:
            self.closing = not self.closing # Cambiar dirección
        else:
            self.pos += (direction / dist) * self.speed

# --- 2. LOS COMPETIDORES ---

# A. SLAM-LIKE AGENT (Memoria Rígida)
# Simula un robot que "recuerda" dónde estaban los muros y no actualiza rápido
class MemoryAgent:
    def __init__(self):
        self.pos = START.copy()
        self.path = [self.pos.copy()]
        self.memory_map = [] # Guarda posiciones de obstaculos que vio
        self.crashed = False

    def step(self, doors):
        if self.crashed: return

        # Planificación simple hacia el target
        direction = TARGET - self.pos
        dir_norm = direction / np.linalg.norm(direction)
        
        # Este agente confía en su memoria. Si vio una puerta abierta hace 10 frames,
        # cree que sigue abierta. (Simulamos fallo de actualización)
        # Solo reacciona si está MUY cerca (0.5)
        
        next_pos = self.pos + dir_norm * 0.3
        
        # Chequeo de colisión REAL
        collision = False
        for door in doors:
            if np.linalg.norm(next_pos - door.pos) < door.radius:
                collision = True
        
        if collision:
            self.crashed = True # Game Over para el clásico
        else:
            self.pos = next_pos
            self.path.append(self.pos.copy())

# B. VNN SMART AGENT (Reactivo Volumétrico)
class VNNAgent:
    def __init__(self):
        self.pos = START.copy()
        self.path = [self.pos.copy()]
        self.frustration_spheres = [] # Memoria temporal de fallos
        self.stuck_timer = 0

    def step(self, doors):
        # 1. Vector Objetivo
        vec_target = TARGET - self.pos
        dist_target = np.linalg.norm(vec_target)
        if dist_target > 0: vec_target /= dist_target
        
        # 2. Vector Repulsión (Sensores en vivo)
        # La VNN siente el volumen de la puerta AHORA, no donde estaba antes
        vec_repulse = np.zeros(2)
        
        # Detectar puertas dinámicas
        for door in doors:
            diff = self.pos - door.pos
            dist = np.linalg.norm(diff)
            # Campo de fuerza repulsivo (Lagrange Soft Constraint)
            if dist < door.radius + 1.0: # Margen de seguridad
                force = (1.0 / (dist + 0.01)) * 2.0
                vec_repulse += (diff / dist) * force
        
        # Detectar esferas de frustración propias
        for sphere in self.frustration_spheres:
            diff = self.pos - sphere['c']
            dist = np.linalg.norm(diff)
            if dist < sphere['r']:
                vec_repulse += (diff/dist) * sphere['p']

        # 3. Fusión
        move_vec = vec_target + vec_repulse
        
        # Normalizar velocidad
        if np.linalg.norm(move_vec) > 0:
            move_vec = (move_vec / np.linalg.norm(move_vec)) * 0.3
            
        # Predicción de movimiento
        next_pos = self.pos + move_vec
        
        # Validación (Evitar traspasar muros por velocidad)
        collision = False
        for door in doors:
            if np.linalg.norm(next_pos - door.pos) < door.radius:
                collision = True
        
        if not collision:
            self.pos = next_pos
            self.stuck_timer = max(0, self.stuck_timer - 0.1)
        else:
            # Inteligencia de atasco
            self.stuck_timer += 1
            if self.stuck_timer > 5:
                # Crear esfera de "Aqui no"
                self.frustration_spheres.append({'c': self.pos.copy(), 'r': 2.0, 'p': 3.0})
                self.stuck_timer = 0
        
        self.path.append(self.pos.copy())

# --- 3. SIMULACIÓN ---
door1 = DynamicDoor([10, 5], [10, 10], speed=0.2) # Puerta cerrándose en el paso
door2 = DynamicDoor([15, 15], [5, 15], speed=0.3) # Muro moviéndose lateralmente

mem_agent = MemoryAgent()
vnn_agent = VNNAgent()

# Corremos 300 frames
frames = 300
history_door1 = []
history_door2 = []

print("Simulando Laberinto Dinámico...")
for i in range(frames):
    door1.update()
    door2.update()
    
    # Guardar historia para plotear estelas
    history_door1.append(door1.pos.copy())
    history_door2.append(door2.pos.copy())
    
    mem_agent.step([door1, door2])
    vnn_agent.step([door1, door2])

# --- 4. VISUALIZACIÓN ---
plt.figure(figsize=(10, 10))

# Dibujar Target
plt.plot(TARGET[0], TARGET[1], 'g*', markersize=20, label='Meta')
plt.plot(START[0], START[1], 'ko', label='Inicio')

# Dibujar Obstáculos Dinámicos (Estelas)
hd1 = np.array(history_door1)
plt.plot(hd1[:,0], hd1[:,1], 'k:', alpha=0.3, linewidth=10, label='Trayectoria Muro Móvil')

# Dibujar Posición Final Obstáculos
circle1 = plt.Circle(door1.pos, door1.radius, color='black', alpha=0.7)
plt.gca().add_patch(circle1)

# Trayectorias Agentes
path_mem = np.array(mem_agent.path)
path_vnn = np.array(vnn_agent.path)

# Agente Clásico (Memoria)
plt.plot(path_mem[:,0], path_mem[:,1], 'r--', linewidth=2, label='Memoria Rígida (Choca)')
if mem_agent.crashed:
    plt.plot(path_mem[-1,0], path_mem[-1,1], 'rx', markersize=15, markeredgewidth=3)

# Agente VNN
plt.plot(path_vnn[:,0], path_vnn[:,1], 'b-', linewidth=3, label='VNN Reactiva (Esquiva)')

# Dibujar "Pensamientos" VNN (Esferas Frustración)
for s in vnn_agent.frustration_spheres:
    c = plt.Circle(s['c'], s['r']/2, color='blue', alpha=0.1)
    plt.gca().add_patch(c)

plt.title("Inteligencia en Entornos Dinámicos: El Laberinto Vivo")
plt.legend()
plt.xlim(0, 20)
plt.ylim(0, 20)
plt.grid(True, alpha=0.3)
plt.show()