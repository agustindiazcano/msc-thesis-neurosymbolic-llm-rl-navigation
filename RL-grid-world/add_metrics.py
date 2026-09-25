# Script para agregar tracking de métricas
import re

with open('c:/projects/VNN_smart/vlnn_complete_experiment.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Agregar tracking después de env.step
content = content.replace(
    '''            next_state_raw, reward, done, info = env.step(action)
            next_state = lidar_env.get_lidar_obs()
            
            if agent_name != "MLP (Adam)":''',
    '''            next_state_raw, reward, done, info = env.step(action)
            next_state = lidar_env.get_lidar_obs()
            
            # Tracking de colisiones
            if info.get('collision', False):
                collision_count += 1
            
            steps_taken += 1
            
            if agent_name != "MLP (Adam)":'''
)

# Agregar métricas después de success
content = content.replace(
    '''        metrics['success'].append(1 if info['goal'] else 0)
        
        if episode % 10 == 0:''',
    '''        metrics['success'].append(1 if info['goal'] else 0)
        metrics['collisions'].append(collision_count)
        metrics['steps_to_goal'].append(steps_taken if info['goal'] else limit_steps)
        
        if episode % 10 == 0:'''
)

# Agregar métricas al JSON final
content = re.sub(
    r'"VLNN": \{\s+"final_win_rate": float\(np\.mean\(test_metrics_vlnn\[\'success\'\]\[-20:\]\)\)\s+\}',
    '''"VLNN": {
            "final_win_rate": float(np.mean(test_metrics_vlnn['success'][-20:])),
            "avg_collisions": float(np.mean(test_metrics_vlnn['collisions'])),
            "avg_steps_to_goal": float(np.mean([s for i, s in enumerate(test_metrics_vlnn['steps_to_goal']) if test_metrics_vlnn['success'][i] == 1] or [0]))
        }''',
    content
)

content = re.sub(
    r'"MLP": \{\s+"final_win_rate": float\(np\.mean\(test_metrics_mlp\[\'success\'\]\[-20:\]\)\)\s+\}',
    '''"MLP": {
            "final_win_rate": float(np.mean(test_metrics_mlp['success'][-20:])),
            "avg_collisions": float(np.mean(test_metrics_mlp['collisions'])),
            "avg_steps_to_goal": float(np.mean([s for i, s in enumerate(test_metrics_mlp['steps_to_goal']) if test_metrics_mlp['success'][i] == 1] or [0]))
        }''',
    content
)

with open('c:/projects/VNN_smart/vlnn_complete_experiment.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ Métricas agregadas correctamente")
