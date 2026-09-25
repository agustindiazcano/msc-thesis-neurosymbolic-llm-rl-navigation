with open('c:/projects/VNN_smart/vlnn_symbolic_experiment.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Buscar la línea donde se define __init__ de VolumetricCognitiveBrain
# Y agregar cortex y compressed_memory después de epsilon
new_lines = []
for i, line in enumerate(lines):
    new_lines.append(line)
    # Buscar la línea que tiene "self.epsilon = PARAMS['epsilon_start']" en VolumetricCognitiveBrain
    if "self.epsilon = PARAMS['epsilon_start']" in line and i > 290 and i < 300:
        # Agregar las dos líneas faltantes
        new_lines.append("        self.cortex = SymbolicCortex(resolution=0.5)\r\n")
        new_lines.append("        self.compressed_memory = []  # Memoria simbólica comprimida\r\n")

with open('c:/projects/VNN_smart/vlnn_symbolic_experiment.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print("✅ Fixed: cortex y compressed_memory agregados a VolumetricCognitiveBrain.__init__")
