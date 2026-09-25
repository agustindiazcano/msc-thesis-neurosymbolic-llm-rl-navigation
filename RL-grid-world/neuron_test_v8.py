import numpy as np
import matplotlib.pyplot as plt
import copy
import time

# --- 1. ENTORNO FÍSICO (CARTPOLE) ---
class CartPole:
    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5 
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = 0.02 

    def step(self, state, force):
        x, x_dot, theta, theta_dot = state
        costheta = np.cos(theta); sintheta = np.sin(theta)
        temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0/3.0 - self.masspole * costheta**2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        
        # Euler integration
        x += self.tau * x_dot
        x_dot += self.tau * xacc
        theta += self.tau * theta_dot
        theta_dot += self.tau * thetaacc
        
        return np.array([x, x_dot, theta, theta_dot])

# --- 2. EL INDIVIDUO (VNN GENÉTICA) ---
class GeneticVNN:
    def __init__(self, n_neurons=15): # ¡Solo 15 neuronas para ser eficientes!
        self.n_neurons = n_neurons
        # Genoma: Matriz (N, 6) -> [Centro(4), Radio(1), Valor_Fuerza(1)]
        # Inicializamos aleatoriamente en el rango útil del problema
        # Estados típicos: x~[-2,2], v~[-2,2], th~[-0.2, 0.2], w~[-2,2]
        self.genes = np.random.uniform(-1, 1, (n_neurons, 6))
        # Ajustar rangos específicos
        self.genes[:, 4] = np.random.uniform(0.1, 1.0, n_neurons) # Radios positivos
        self.genes[:, 5] = np.random.uniform(-20, 20, n_neurons)  # Fuerzas fuertes
        
        self.fitness = 0.0

    def predict(self, state):
        # Decodificar genoma
        centers = self.genes[:, :4]
        radii = self.genes[:, 4]
        values = self.genes[:, 5]
        
        # Inferencia Vectorizada
        # Distancia (1, 4) - (N, 4) -> (N,)
        dists = np.linalg.norm(state - centers, axis=1)
        
        # Gaussian Kernel
        weights = np.exp(-dists**2 / (2 * (radii/2)**2))
        
        # Mask (Radio estricto para localidad)
        mask = dists < radii
        weights *= mask
        
        denom = np.sum(weights) + 1e-8
        force = np.sum(weights * values) / denom
        
        # Si nadie se activa, fuerza 0 (o pánico aleatorio)
        if denom < 1e-5: return 0.0
        return np.clip(force, -20, 20) # Limitar fuerza motores

# --- 3. MOTOR EVOLUTIVO (ALGORITMO GENÉTICO) ---
class Evolution:
    def __init__(self, pop_size=50, mutation_rate=0.1):
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.population = [GeneticVNN() for _ in range(pop_size)]
        self.generation = 0
        self.best_history = []
        
    def evaluate_fitness(self):
        env = CartPole()
        for ind in self.population:
            state = np.array([0, 0, np.random.uniform(-0.05, 0.05), 0]) # Inicio casi recto
            steps = 0
            # Simular hasta que se caiga o llegue a 500 pasos (10 segundos)
            for _ in range(500):
                force = ind.predict(state)
                state = env.step(state, force)
                
                # Condiciones de fallo (Ángulo > 12 grados o Salirse de pista)
                if abs(state[2]) > 0.20 or abs(state[0]) > 2.4:
                    break
                steps += 1
            
            # Penalizar el uso de fuerza excesiva (Eficiencia Energética) opcional
            ind.fitness = steps

    def select_survivors(self):
        # Ordenar por fitness (Mayor es mejor)
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        # Quedarse con el Top 20% (Elites)
        n_elites = int(self.pop_size * 0.2)
        survivors = self.population[:n_elites]
        return survivors

    def crossover_and_mutate(self, survivors):
        new_pop = []
        # Elites pasan directo (Inmortalidad del campeón)
        new_pop.extend([copy.deepcopy(s) for s in survivors])
        
        # Rellenar el resto con hijos
        while len(new_pop) < self.pop_size:
            # Torneo simple: Elegir 2 padres al azar de los supervivientes
            p1, p2 = np.random.choice(survivors, 2, replace=False)
            
            # Crear hijo vacío
            child = GeneticVNN(n_neurons=p1.n_neurons)
            
            # Crossover (Mezcla de genes)
            # Para cada neurona, lanzamos una moneda de quién la heredamos
            mask = np.random.rand(child.n_neurons) > 0.5
            child.genes[mask] = p1.genes[mask]
            child.genes[~mask] = p2.genes[~mask]
            
            # Mutación (Radiación cósmica)
            # Modificar ligeramente algunos genes
            mutation_mask = np.random.rand(*child.genes.shape) < self.mutation_rate
            noise = np.random.normal(0, 0.2, size=child.genes.shape) # Pequeños ajustes
            child.genes[mutation_mask] += noise[mutation_mask]
            
            # Asegurar radios positivos
            child.genes[:, 4] = np.abs(child.genes[:, 4])
            
            new_pop.append(child)
        
        self.population = new_pop

    def run_generation(self):
        self.evaluate_fitness()
        best_fit = self.population[0].fitness # Ya que no hemos ordenado aún en eval, pero ordenamos luego
        # (Mejor ordenar primero para stats)
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        best_fit = self.population[0].fitness
        mean_fit = np.mean([x.fitness for x in self.population])
        
        self.best_history.append(best_fit)
        
        survivors = self.select_survivors()
        self.crossover_and_mutate(survivors)
        self.generation += 1
        
        return best_fit, mean_fit

# --- 4. EJECUCIÓN ---

print("Iniciando Evolución Volumétrica (VNN-GA)...")
ga = Evolution(pop_size=50, mutation_rate=0.15) # 50 individuos

generations = 30
stats = []

t0 = time.time()
for gen in range(generations):
    best, mean = ga.run_generation()
    stats.append((best, mean))
    # Barra de progreso
    bar = "#" * int(best / 20)
    print(f"Gen {gen+1:02d} | Best Fit: {best:5.1f} steps | Mean: {mean:5.1f} | {bar}")
    
    if best >= 499: # Éxito total
        print(" -> ¡Evolución Convergida! Equilibrio Perfecto Alcanzado.")
        break

total_time = time.time() - t0

# --- 5. VISUALIZACIÓN ---
stats = np.array(stats)
plt.figure(figsize=(10, 6))
plt.plot(stats[:, 0], 'g-o', label='Mejor Individuo (Campeón)')
plt.plot(stats[:, 1], 'b--', label='Promedio Población')
plt.axhline(y=500, color='r', linestyle=':', label='Objetivo (10 segs)')
plt.xlabel("Generación")
plt.ylabel("Tiempo de Supervivencia (Frames)")
plt.title(f"Evolución de Calibración VNN ({total_time:.1f}s)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# --- 6. VALIDACIÓN DEL CAMPEÓN ---
champion = ga.population[0] # El mejor de la última gen (ya ordenado)
print(f"\nValidando al Campeón (Genoma de 15 Neuronas)...")
print("Genes (Muestra de 3 neuronas):")
print(champion.genes[:3]) # Ver qué aprendió

# Prueba de robustez rápida con el campeón
wins = 0
for _ in range(10):
    env = CartPole()
    state = np.array([0, 0, np.random.uniform(-0.1, 0.1), 0]) # Perturbación inicial
    steps = 0
    for _ in range(500):
        action = champion.predict(state)
        state = env.step(state, action)
        if abs(state[2]) > 0.20 or abs(state[0]) > 2.4: break
        steps += 1
    if steps == 500: wins += 1

print(f"Tasa de Éxito del Campeón: {wins}/10 pruebas perfectas.")