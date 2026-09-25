# Script para agregar tracking de colisiones en training
import re

with open('c:/projects/VNN_smart/vlnn_complete_experiment.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Agregar collisions a las métricas de training
content = content.replace(
    "metrics = {'rewards': [], 'success': [], 'params': []}",
    "metrics = {'rewards': [], 'success': [], 'params': [], 'collisions': []}"
)

# 2. Agregar tracking de colisiones en el paso de entrenamiento (después de env.step)
# Buscar el bloque de training step y agregar tracking
content = re.sub(
    r"(            next_state_raw, reward, done, info = env\.step\(action\)\r?\n            next_state = lidar_env\.get_lidar_obs\(\))\r?\n(\r?\n            if agent_name == \"MLP \(Adam\)\":)",
    r"\1\r\n            \r\n            # Track collisions\r\n            if 'collision_count' not in locals():\r\n                collision_count = 0\r\n            if info.get('collision', False):\r\n                collision_count += 1\r\n\2",
    content
)

# 3. Agregar reset de collision_count al inicio de cada episodio en training
content = re.sub(
    r"(        total_reward = 0\r?\n        done = False)",
    r"\1\r\n        collision_count = 0  # Track collisions per episode",
    content
)

# 4. Agregar registro de colisiones al final de cada episodio
content = re.sub(
    r"(        metrics\['success'\]\.append\(1 if info\['goal'\] else 0\))",
    r"\1\r\n        metrics['collisions'].append(collision_count)",
    content
)

# 5. Agregar reporte de colisiones en el print cada 50 episodios
content = re.sub(
    r'(        if episode % 50 == 0:\r?\n            avg_win = np\.mean\(metrics\[\'success\'\]\[-50:\]\)\r?\n)(            print\(f"\[{agent_name}\] Ep {episode} \(Map {size}x{size}\): WinRate {avg_win:.2f} \| Params: {metrics\[\'params\'\]\[-1\]}"\))',
    r'\1            avg_collisions = np.mean(metrics[\'collisions\'][-50:]) if len(metrics[\'collisions\']) >= 50 else np.mean(metrics[\'collisions\']) if metrics[\'collisions\'] else 0\r\n\2'.replace(
        'WinRate {avg_win:.2f}',
        'WinRate {avg_win:.2f} | Avg Collisions: {avg_collisions:.1f}'
    ),
    content
)

# 6. Agregar métricas de colisiones por mapa al JSON final
# Calcular colisiones por mapa (6x6: eps 0-199, 9x9: eps 200-499, 12x12: eps 500-799)
json_addition = '''
        # Calcular colisiones por mapa
        vlnn_collisions_6x6 = np.mean(m_vlnn['collisions'][0:200]) if len(m_vlnn['collisions']) > 0 else 0
        vlnn_collisions_9x9 = np.mean(m_vlnn['collisions'][200:500]) if len(m_vlnn['collisions']) > 200 else 0
        vlnn_collisions_12x12 = np.mean(m_vlnn['collisions'][500:800]) if len(m_vlnn['collisions']) > 500 else 0
        
        mlp_collisions_6x6 = np.mean(m_mlp['collisions'][0:200]) if len(m_mlp['collisions']) > 0 else 0
        mlp_collisions_9x9 = np.mean(m_mlp['collisions'][200:500]) if len(m_mlp['collisions']) > 200 else 0
        mlp_collisions_12x12 = np.mean(m_mlp['collisions'][500:800]) if len(m_mlp['collisions']) > 500 else 0
'''

content = re.sub(
    r'(    results = \{)',
    json_addition + r'\n    results = {',
    content
)

# Agregar al JSON de resultados
content = re.sub(
    r'("VLNN": \{\s+"time_sec": float\(t_vlnn\),\s+"final_win_rate": float\(np\.mean\(m_vlnn\[\'success\'\]\[-50:\]\)\),\s+"final_params": int\(m_vlnn\[\'params\'\]\[-1\]\)\s+\})',
    r'''"VLNN": {
                "time_sec": float(t_vlnn),
                "final_win_rate": float(np.mean(m_vlnn['success'][-50:])),
                "final_params": int(m_vlnn['params'][-1]),
                "collisions_6x6": float(vlnn_collisions_6x6),
                "collisions_9x9": float(vlnn_collisions_9x9),
                "collisions_12x12": float(vlnn_collisions_12x12)
            }''',
    content
)

content = re.sub(
    r'("MLP": \{\s+"time_sec": float\(t_mlp\),\s+"final_win_rate": float\(np\.mean\(m_mlp\[\'success\'\]\[-50:\]\)\),\s+"final_params": int\(m_mlp\[\'params\'\]\[-1\]\)\s+\})',
    r'''"MLP": {
                "time_sec": float(t_mlp),
                "final_win_rate": float(np.mean(m_mlp['success'][-50:])),
                "final_params": int(m_mlp['params'][-1]),
                "collisions_6x6": float(mlp_collisions_6x6),
                "collisions_9x9": float(mlp_collisions_9x9),
                "collisions_12x12": float(mlp_collisions_12x12)
            }''',
    content
)

with open('c:/projects/VNN_smart/vlnn_complete_experiment.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ Tracking de colisiones en curriculum agregado")
