with open('c:/projects/VNN_smart/vlnn_symbolic_scaling_experiment.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Arreglar los strings mal formateados
fixes = [
    ('print("=== 🧠 FASE 1: ENTRENAMIENTO (ESCUELA) ===\r\n")', 
     'print("=== 🧠 FASE 1: ENTRENAMIENTO (ESCUELA) ===\\n")'),
    
    ('print("\r\n\u003e\u003e\u003e Entrenando VLNN Simbólica...")',
     'print("\\n\u003e\u003e\u003e Entrenando VLNN Simbólica...")'),
    
    ('print("\r\n\u003e\u003e\u003e Entrenando MLP Baseline...")',
     'print("\\n\u003e\u003e\u003e Entrenando MLP Baseline...")'),
    
    ('print("\r\n\r\n=== 🔥 FASE 2: PRUEBAS DE ESCALADO (MÚLTIPLES ENTORNOS) ===")',
     'print("\\n\\n=== 🔥 FASE 2: PRUEBAS DE ESCALADO (MÚLTIPLES ENTORNOS) ===")'),
    
    ('print(f"\r\n📍 ENTORNO: {env_name} ({config[\'size\']}x{config[\'size\']})\r\n")',
     'print(f"\\n📍 ENTORNO: {env_name} ({config[\'size\']}x{config[\'size\']})\\n")'),
    
    ('print("\r\n📊 Gráficas guardadas en \'vlnn_symbolic_scaling_comparison.png\'")',
     'print("\\n📊 Gráficas guardadas en \'vlnn_symbolic_scaling_comparison.png\'")'),
    
    ('print("\r\n=== RESULTADOS FINALES (JSON) ===")',
     'print("\\n=== RESULTADOS FINALES (JSON) ===")'),
]

for old, new in fixes:
    content = content.replace(old, new)

with open('c:/projects/VNN_smart/vlnn_symbolic_scaling_experiment.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ Strings multilinea arreglados")
