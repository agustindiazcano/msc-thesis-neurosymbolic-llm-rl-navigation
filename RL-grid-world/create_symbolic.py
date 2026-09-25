# Script para agregar capa simbólica al experimento
import re

with open('c:/projects/VNN_smart/vlnn_symbolic_experiment.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Agregar la clase SymbolicCortex después de los imports
symbolic_cortex_class = '''

# ==========================================
# SYMBOLIC CORTEX - Representación Simbólica
# ==========================================
class SymbolicCortex:
    """
    Implementación del 'Semantic Hash Map' y 'Representación Distribuida'.
    Convierte grupos de neuronas (esferas) en Símbolos Lógicos reutilizables.
    """
    def __init__(self, resolution=0.5):
        self.resolution = resolution  # Tamaño de la celda de cuantización
        self.symbol_registry = {}     # { Hash_Geométrico : ID_Símbolo }
        self.vocab_size = 0           # Cantidad de "conceptos" aprendidos
        self.known_patterns = {}      # { ID : Estructura_Relativa }

    def quantize(self, vector):
        """Convierte coordenadas continuas (float) a discretas (int) para el Hash."""
        return tuple(np.round(vector / self.resolution).astype(int))

    def encode_cluster(self, neurons_cluster):
        """
        Toma un grupo de neuronas, normaliza su posición relativa y busca si
        ya existe un símbolo para esta forma geométrica.
        """
        if not neurons_cluster:
            return None, None, False

        # 1. Encontrar el centroide del cluster (para relatividad)
        centers = np.array([n.c for n in neurons_cluster])
        centroid = np.mean(centers, axis=0)
        
        # 2. Calcular offsets relativos (Topología Local)
        relative_offsets = []
        for c in centers:
            rel_pos = self.quantize(c - centroid)
            relative_offsets.append(rel_pos)
        
        # 3. Crear el Hash Geométrico (invariable a traslación)
        geometric_hash = tuple(sorted(relative_offsets))

        # 4. Buscamos en el Registro de Símbolos
        if geometric_hash in self.symbol_registry:
            symbol_id = self.symbol_registry[geometric_hash]
            is_new = False
        else:
            # Forma nueva, aprendemos el símbolo
            self.vocab_size += 1
            symbol_id = f"SYM_{self.vocab_size}"
            self.symbol_registry[geometric_hash] = symbol_id
            self.known_patterns[symbol_id] = geometric_hash
            is_new = True

        return symbol_id, centroid, is_new
'''

# Insertar después de "import json"
content = content.replace(
    'import json',
    'import json' + symbolic_cortex_class
)

# 2. Modificar VolumetricCognitiveBrain para agregar cortex
# Buscar __init__ de VolumetricCognitiveBrain y agregar cortex
content = re.sub(
    r'(class VolumetricCognitiveBrain:.*?def __init__\(self, input_dim, action_dim\):.*?self\.neurons = \[\])',
    r'''\1
        self.cortex = SymbolicCortex(resolution=0.5)
        self.compressed_memory = []  # Memoria simbólica comprimida''',
    content,
    flags=re.DOTALL
)

# 3. Agregar método consolidate_memory antes de get_active_neuron_idx
consolidate_method = '''
    
    def consolidate_memory(self, verbose=False):
        """
        Consolidación de Memoria: Comprime neuronas crudas en símbolos.
        Simula el proceso de memoria a largo plazo.
        """
        if not self.neurons:
            return

        # 1. Agrupamiento espacial simple (grid-based clustering)
        clusters = {}
        for neuron in self.neurons:
            grid_key = self.cortex.quantize(neuron.c)
            if grid_key not in clusters:
                clusters[grid_key] = []
            clusters[grid_key].append(neuron)

        # 2. Simbolización
        new_compressed_memory = []
        neurons_to_keep = []

        for grid_pos, cluster in clusters.items():
            if len(cluster) < 3:
                # Muy pocas neuronas, no vale la pena simbolizar
                neurons_to_keep.extend(cluster)
                continue

            # Convertir cluster en símbolo
            sym_id, centroid, is_new = self.cortex.encode_cluster(cluster)
            
            if sym_id is None:
                neurons_to_keep.extend(cluster)
                continue
            
            # Guardar referencia simbólica (mucho más liviano que N neuronas)
            new_compressed_memory.append({
                'symbol_id': sym_id,
                'position': centroid,
                'q_values': np.mean([n.q_values for n in cluster], axis=0),
                'radius': np.mean([n.r for n in cluster])
            })
            
            if verbose and is_new:
                print(f"🧠 [CORTEX] Nuevo Concepto: {sym_id} (comprime {len(cluster)} neuronas)")

        # 3. Actualizar memoria
        self.compressed_memory = new_compressed_memory
        # Mantener solo neuronas recientes no consolidadas
        self.neurons = neurons_to_keep
        
        if verbose:
            print(f"🧹 Consolidación: {len(self.compressed_memory)} símbolos | {len(self.neurons)} neuronas activas")
'''

# Buscar el método get_active_neuron_idx y agregar consolidate_memory antes
content = re.sub(
    r'(\n    def get_active_neuron_idx)',
    consolidate_method + r'\1',
    content
)

# 4. Modificar get_action para consultar símbolos también
# Buscar donde se obtienen q_values y agregar consulta a símbolos
old_get_action = r'''(    def get_action\(self, state, current_step=0\):
        # 1\. Obtener la opinión del cerebro \(consultar neuronas\)
        idx = self\.get_active_neuron_idx\(state, current_step\)
        neuron = self\.neurons\[idx\]
        q_values = neuron\.q_values\.copy\(\)  # Copiar para no modificar memoria)'''

new_get_action = r'''\1
        
        # 1.5. CONSULTAR MEMORIA SIMBÓLICA (Símbolos aprendidos)
        for memory_item in self.compressed_memory:
            sym_pos = memory_item['position']
            dist = np.linalg.norm(state[:len(sym_pos)] - sym_pos)
            symbol_radius = memory_item['radius'] * 1.5  # Símbolos tienen más influencia
            
            if dist < symbol_radius:
                weight = 1.0 / (dist + 1e-6)
                q_values += memory_item['q_values'] * weight * 0.5  # Influencia moderada'''

content = re.sub(old_get_action, new_get_action, content, flags=re.DOTALL)

# 5. Agregar métricas de símbolos
content = content.replace(
    "metrics = {'rewards': [], 'success': [], 'params': [], 'collisions': []}",
    "metrics = {'rewards': [], 'success': [], 'params': [], 'collisions': [], 'symbols': [], 'raw_neurons': [], 'compression_ratio': []}"
)

# 6. Agregar tracking de símbolos después de cada episodio en training
content = re.sub(
    r"(        if agent_name != \"MLP \(Adam\)\":[\r\n]+            metrics\['params'\]\.append\(agent\.get_parameter_count\(\)\))",
    r'''\1
            metrics['symbols'].append(len(agent.compressed_memory))
            metrics['raw_neurons'].append(len(agent.neurons))
            total_params = len(agent.neurons) + len(agent.compressed_memory)
            original_params = metrics['params'][-1] if metrics['params'] else 1
            metrics['compression_ratio'].append(total_params / max(original_params, 1))''',
    content
)

# 7. Agregar consolidación periódica en el loop de training
content = re.sub(
    r"(            if episode % 5 == 0: agent\.fuse_neurons\(\))",
    r'''\1
            # Consolidación simbólica cada 50 episodios
            if episode % 50 == 0:
                agent.consolidate_memory(verbose=(episode % 100 == 0))''',
    content
)

# 8. Actualizar el print para mostrar símbolos
content = re.sub(
    r'print\(f"\[{agent_name}\] Ep {episode} \(Map {size}x{size}\): WinRate {avg_win:.2f} \| Avg Collisions: {avg_collisions:.1f} \| Params: {metrics\[\'params\'\]\[-1\]}"\)',
    r'''if agent_name != "MLP (Adam)" and 'symbols' in metrics and metrics['symbols']:
                syms = metrics['symbols'][-1]
                raw = metrics['raw_neurons'][-1]
                print(f"[{agent_name}] Ep {episode} (Map {size}x{size}): WinRate {avg_win:.2f} | Collisions: {avg_collisions:.1f} | Symbols: {syms} | Raw: {raw}")
            else:
                print(f"[{agent_name}] Ep {episode} (Map {size}x{size}): WinRate {avg_win:.2f} | Collisions: {avg_collisions:.1f} | Params: {metrics['params'][-1]}")''',
    content
)

# 9. Agregar consolidación en deployment también
content = re.sub(
    r"(        if agent_name != \"MLP \(Adam\)\":[\r\n]+            agent\.prune_dead_neurons\(global_step, max_age=5000\)[\r\n]+            if episode % 5 == 0: agent\.fuse_neurons\(\))",
    r'''\1
            # Consolidación cada 10 episodios en deployment
            if episode % 10 == 0:
                agent.consolidate_memory(verbose=False)''',
    content
)

with open('c:/projects/VNN_smart/vlnn_symbolic_experiment.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ Experimento simbólico creado: vlnn_symbolic_experiment.py")
print("   - SymbolicCortex agregado")
print("   - consolidate_memory() implementado")
print("   - Inferencia con símbolos activada")
print("   - Métricas de compresión agregadas")
