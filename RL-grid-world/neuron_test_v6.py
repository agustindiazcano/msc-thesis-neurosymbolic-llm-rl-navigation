import numpy as np
import matplotlib.pyplot as plt
import time

# --- 1. EL ENTORNO (LA TRAMPA EN U) ---
# Una pared en forma de U que bloquea el paso directo al objetivo
def is_collision(pos):
    x, y = pos
    # Pared Fondo
    if 4.0 <= x <= 6.0 and -2.0 <= y <= 2.0: return True
    # Pared Lateral Arriba
    if 4.0 <= x <= 6.0 and 2.0 <= y <= 2.5: return True
    # Pared Lateral Abajo
    if 4.0 <= x <= 6.0 and -2.5 <= y <= -2.0: return True
    return False

TARGET = np.array([8.0, 0.0])
START = np.array([0.0, 0.0])

# --- 2. LOS COMPETIDORES ---

# A. STANDARD POTENTIAL FIELD (El Tonto / Greedy)
# Simula un MLP o Algoritmo clásico que solo sigue el gradiente del objetivo
class GreedyAgent:
    def __init__(self):
        self.pos = START.copy()
        self.path = [self.pos.copy()]
        
    def step(self):
        # Vector hacia el objetivo
        direction = TARGET - self.pos
        dist = np.linalg.norm(direction)
        if dist > 0: direction /= dist # Normalizar
        
        # Intentar mover
        new_pos = self.pos + direction * 0.2
        
        # Si choca, se desliza (física básica) o se queda quieto
        if not is_collision(new_pos):
            self.pos = new_pos
        else:
            # Choque! (Se queda trabado tratando de atravesar la pared)
            pass 
            
        self.path.append(self.pos.copy())

# B. VOLUMETRIC LOGIC AGENT (El Inteligente)
class VNN_SmartAgent:
    def __init__(self):
        self.pos = START.copy()
        self.path = [self.pos.copy()]
        self.memory_spheres = [] # Aquí guardamos "experiencia"
        self.stuck_counter = 0
        
    def step(self):
        # 1. Atracción al Objetivo (Igual que el tonto)
        attraction = TARGET - self.pos
        dist = np.linalg.norm(attraction)
        if dist > 0: attraction /= dist
        
        # 2. REPULSIÓN VOLUMÉTRICA (La Magia)
        # Las esferas de memoria empujan al robot lejos de zonas malas
        repulsion = np.zeros(2)
        for s in self.memory_spheres:
            d_vec = self.pos - s['c']
            d = np.linalg.norm(d_vec)
            if d < s['r']:
                # Fuerza repulsiva inversamente proporcional a la distancia
                repulsion += (d_vec / (d + 0.1)) * s['power']
        
        # Sumar fuerzas
        move_vec = attraction + repulsion
        
        # Normalizar velocidad máxima
        if np.linalg.norm(move_vec) > 0:
            move_vec = (move_vec / np.linalg.norm(move_vec)) * 0.2
        
        # 3. INTELIGENCIA DE "FRUSTRACIÓN"
        prev_pos = self.pos.copy()
        test_pos = self.pos + move_vec
        
        if not is_collision(test_pos):
            self.pos = test_pos
            self.stuck_counter = max(0, self.stuck_counter - 1)
        else:
            # ESTOY CHOCANDO O NO AVANZO
            self.stuck_counter += 1
            
            # Si llevo mucho tiempo atascado en el mismo sitio...
            if self.stuck_counter > 5:
                # CREAR UNA ESFERA DE "AQUÍ NO ES" (Frustración)
                # Esto "rellena" volumétricamente la trampa
                self.memory_spheres.append({
                    'c': self.pos.copy(),
                    'r': 1.5,     # Radio de influencia
                    'power': 2.0  # Fuerza de repulsión
                })
                self.stuck_counter = 0 # Reiniciar contador
                # Pequeño salto random para desencajar
                self.pos += np.random.randn(2) * 0.1
        
        self.path.append(self.pos.copy())

# --- 3. EJECUCIÓN ---

greedy = GreedyAgent()
smart = VNN_SmartAgent()

print("Simulando escape de trampa...")
for _ in range(200): # 200 pasos
    greedy.step()
    smart.step()

# --- 4. VISUALIZACIÓN ---
plt.figure(figsize=(10, 6))

# Dibujar Paredes (Trampa en U)
plt.plot([4, 4], [-2.5, 2.5], 'k-', linewidth=5) # Fondo
plt.plot([4, 6], [2.5, 2.5], 'k-', linewidth=5)  # Brazo arriba
plt.plot([4, 6], [-2.5, -2.5], 'k-', linewidth=5) # Brazo abajo

# Dibujar Target
plt.plot(TARGET[0], TARGET[1], 'g*', markersize=20, label='Objetivo')

# Trayectoria Greedy
path_g = np.array(greedy.path)
plt.plot(path_g[:, 0], path_g[:, 1], 'r--', linewidth=2, label='Greedy/MLP (Atascado)')

# Trayectoria Smart
path_s = np.array(smart.path)
plt.plot(path_s[:, 0], path_s[:, 1], 'b-', linewidth=3, label='VNN Smart (Escapa)')

# Dibujar las esferas de memoria del Smart
for s in smart.memory_spheres:
    circle = plt.Circle(s['c'], s['r']/3, color='blue', alpha=0.2)
    plt.gca().add_patch(circle)

plt.title("Prueba de Inteligencia Espacial: Problema del Mínimo Local")
plt.legend()
plt.xlim(-1, 9)
plt.ylim(-4, 4)
plt.grid(True, alpha=0.3)
plt.show()