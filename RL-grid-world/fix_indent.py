with open('c:/projects/VNN_smart/vlnn_complete_experiment.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Encontrar y fix las líneas con indentación incorrecta (deben empezar en columna 4, no 8)
fixed_lines = []
for i, line in enumerate(lines):
    # Las líneas de cálculo de colisiones (alrededor de 665-672) tienen indentación extra
    if i >= 664 and i <= 674:  # Ajustar el rango según sea necesario
        if line.startswith('        vlnn_collisions') or line.startswith('        mlp_collisions'):
            line = line[4:]  # Remover 4 espacios extra
    fixed_lines.append(line)

with open('c:/projects/VNN_smart/vlnn_complete_experiment.py', 'w', encoding='utf-8') as f:
    f.writelines(fixed_lines)

print("✅ Fixed indentation")
