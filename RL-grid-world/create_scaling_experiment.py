# Script para crear experimento de escalado con 3 agentes y 4 mapas
import shutil

# Copiar el archivo simbólico como base
shutil.copy('c:/projects/VNN_smart/vlnn_symbolic_experiment.py', 
            'c:/projects/VNN_smart/vlnn_symbolic_scaling_experiment.py')

with open('c:/projects/VNN_smart/vlnn_symbolic_scaling_experiment.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Modificar para entrenar 3 agentes
training_section = '''if __name__ == "__main__":
    env = GridWorld()
    map_manager = MapManager(PARAMS['n_episodes'])
    
    print("=== 🧠 FASE 1: ENTRENAMIENTO (ESCUELA) ===\\n")
    
    print(">>> Entrenando VLNN Estándar (sin símbolos)...")
    m_vlnn_std, t_vlnn_std, agent_vlnn_std = train("VLNN (Standard)", env, map_manager)
    
    print("\\n>>> Entrenando VLNN Simbólica...")
    m_vlnn_sym, t_vlnn_sym, agent_vlnn_sym = train("VLNN (Symbolic)", env, map_manager)
    
    print("\\n>>> Entrenando MLP Baseline...")
    m_mlp, t_mlp, agent_mlp = train("MLP (Adam)", env, map_manager)
    
    print("\\n\\n=== 🔥 FASE 2: PRUEBAS DE ESCALADO (MÚLTIPLES ENTORNOS) ===")
    
    # 4 Mapas de despliegue con dificultad incremental
    test_configs = [
        {"name": "Jungla", "size": 24, "obs": 60},
        {"name": "Desierto", "size": 30, "obs": 95},
        {"name": "Antártida", "size": 40, "obs": 170},
        {"name": "Patagonia", "size": 50, "obs": 265}
    ]
    
    results_by_env = {}
    
    for config in test_configs:
        env_name = config["name"]
        print(f"\\n📍 ENTORNO: {env_name} ({config['size']}x{config['size']})\\n")
        
        # Test VLNN Estándar
        test_vlnn_std = run_deployment_test(
            "VLNN (Standard)", agent_vlnn_std, env, 
            test_size=config["size"], test_obs=config["obs"], n_test_episodes=100
        )
        
        # Test VLNN Simbólica
        test_vlnn_sym = run_deployment_test(
            "VLNN (Symbolic)", agent_vlnn_sym, env,
            test_size=config["size"], test_obs=config["obs"], n_test_episodes=100
        )
        
        # Test MLP
        test_mlp = run_deployment_test(
            "MLP (Adam)", agent_mlp, env,
            test_size=config["size"], test_obs=config["obs"], n_test_episodes=100
        )
        
        results_by_env[env_name] = {
            "VLNN_Standard": test_vlnn_std,
            "VLNN_Symbolic": test_vlnn_sym,
            "MLP": test_mlp
        }'''

# Reemplazar el main actual
content = content.replace(
    'if __name__ == "__main__":',
    '# MAIN MODIFICADO PARA 3 AGENTES\nif __name__ == "__main__":'
)

# Buscar el main y reemplazarlo completamente
import re
content = re.sub(
    r'if __name__ == "__main__":.*?plt\.savefig.*?json\.dumps\(results, indent=4\)\)',
    training_section + '''
    
    # GRAFICAR COMPARACIÓN DE ESCALADO
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Gráfica 1: Win Rate por entorno
    ax1 = axes[0, 0]
    env_names = list(results_by_env.keys())
    vlnn_std_wr = [np.mean(results_by_env[e]["VLNN_Standard"]['success'][-20:]) for e in env_names]
    vlnn_sym_wr = [np.mean(results_by_env[e]["VLNN_Symbolic"]['success'][-20:]) for e in env_names]
    mlp_wr = [np.mean(results_by_env[e]["MLP"]['success'][-20:]) for e in env_names]
    
    x = np.arange(len(env_names))
    width = 0.25
    ax1.bar(x - width, vlnn_std_wr, width, label='VLNN Standard', color='cyan')
    ax1.bar(x, vlnn_sym_wr, width, label='VLNN Symbolic', color='blue')
    ax1.bar(x + width, mlp_wr, width, label='MLP', color='magenta')
    ax1.set_xlabel('Entorno')
    ax1.set_ylabel('Win Rate')
    ax1.set_title('Win Rate por Entorno de Deployment')
    ax1.set_xticks(x)
    ax1.set_xticklabels(env_names)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gráfica 2: Colisiones por entorno
    ax2 = axes[0, 1]
    vlnn_std_col = [np.mean(results_by_env[e]["VLNN_Standard"]['collisions']) for e in env_names]
    vlnn_sym_col = [np.mean(results_by_env[e]["VLNN_Symbolic"]['collisions']) for e in env_names]
    mlp_col = [np.mean(results_by_env[e]["MLP"]['collisions']) for e in env_names]
    
    ax2.bar(x - width, vlnn_std_col, width, label='VLNN Standard', color='cyan')
    ax2.bar(x, vlnn_sym_col, width, label='VLNN Symbolic', color='blue')
    ax2.bar(x + width, mlp_col, width, label='MLP', color='magenta')
    ax2.set_xlabel('Entorno')
    ax2.set_ylabel('Avg Collisions')
    ax2.set_title('Colisiones Promedio por Entorno')
    ax2.set_xticks(x)
    ax2.set_xticklabels(env_names)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Gráfica 3: Parámetros/Símbolos (solo VLNN)
    ax3 = axes[1, 0]
    # Mostrar cómo los parámetros escalan
    map_sizes = [24, 30, 40, 50]
    vlnn_std_params = [len(agent_vlnn_std.neurons) for _ in map_sizes]  # Asumimos que crece
    vlnn_sym_params = [len(agent_vlnn_sym.neurons) + len(agent_vlnn_sym.compressed_memory) for _ in map_sizes]
    
    ax3.plot(map_sizes, vlnn_std_params, marker='o', label='VLNN Standard (neuronas)', color='cyan', linewidth=2)
    ax3.plot(map_sizes, vlnn_sym_params, marker='s', label='VLNN Symbolic (neuronas+símbolos)', color='blue', linewidth=2)
    ax3.set_xlabel('Tamaño de Mapa')
    ax3.set_ylabel('Parámetros')
    ax3.set_title('Escalado de Parámetros vs Tamaño de Mapa')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Gráfica 4: Training curves
    ax4 = axes[1, 1]
    def smooth(data, w=20):
        if len(data) < w: return data
        return np.convolve(data, np.ones(w)/w, mode='valid')
    
    if len(m_vlnn_std['success']) > 20: ax4.plot(smooth(m_vlnn_std['success']), label='VLNN Standard', color='cyan', linewidth=2)
    if len(m_vlnn_sym['success']) > 20: ax4.plot(smooth(m_vlnn_sym['success']), label='VLNN Symbolic', color='blue', linewidth=2)
    if len(m_mlp['success']) > 20: ax4.plot(smooth(m_mlp['success']), label='MLP', color='magenta', linestyle='--')
    ax4.set_xlabel('Episodios')
    ax4.set_ylabel('Win Rate')
    ax4.set_title('FASE 1: Training Curves')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('vlnn_symbolic_scaling_comparison.png')
    print("\\n📊 Gráficas guardadas en 'vlnn_symbolic_scaling_comparison.png'")
    
    # JSON de resultados
    results = {
        "phase1_training": {
            "VLNN_Standard": {
                "final_win_rate": float(np.mean(m_vlnn_std['success'][-50:])),
                "final_params": len(agent_vlnn_std.neurons),
                "time_sec": float(t_vlnn_std)
            },
            "VLNN_Symbolic": {
                "final_win_rate": float(np.mean(m_vlnn_sym['success'][-50:])),
                "final_params": len(agent_vlnn_sym.neurons) + len(agent_vlnn_sym.compressed_memory),
                "symbols": len(agent_vlnn_sym.compressed_memory),
                "time_sec": float(t_vlnn_sym)
            },
            "MLP": {
                "final_win_rate": float(np.mean(m_mlp['success'][-50:])),
                "final_params": 4933,
                "time_sec": float(t_mlp)
            }
        },
        "phase2_deployment": {}
    }
    
    for env_name in env_names:
        results["phase2_deployment"][env_name] = {
            "VLNN_Standard": {
                "win_rate": float(np.mean(results_by_env[env_name]["VLNN_Standard"]['success'][-20:])),
                "avg_collisions": float(np.mean(results_by_env[env_name]["VLNN_Standard"]['collisions']))
            },
            "VLNN_Symbolic": {
                "win_rate": float(np.mean(results_by_env[env_name]["VLNN_Symbolic"]['success'][-20:])),
                "avg_collisions": float(np.mean(results_by_env[env_name]["VLNN_Symbolic"]['collisions']))
            },
            "MLP": {
                "win_rate": float(np.mean(results_by_env[env_name]["MLP"]['success'][-20:])),
                "avg_collisions": float(np.mean(results_by_env[env_name]["MLP"]['collisions']))
            }
        }
    
    print("\\n=== RESULTADOS FINALES (JSON) ===")
    print(json.dumps(results, indent=4))''',
    content,
    flags=re.DOTALL
)

# 2. Modificar run_deployment_test para aceptar parámetros de tamaño
old_deploy_sig = 'def run_deployment_test(agent_name, agent, env, n_test_episodes=100):'
new_deploy_sig = 'def run_deployment_test(agent_name, agent, env, test_size=24, test_obs=60, n_test_episodes=100):'
content = content.replace(old_deploy_sig, new_deploy_sig)

# Reemplazar las líneas hardcodeadas de test_size y test_obs
content = content.replace(
    'test_size = 24\n    test_obs = 60',
    '# test_size y test_obs vienen como parámetros'
)

# 3. Modificar train para diferenciar entre Standard y Symbolic
# Buscar donde se decide si consolidar y añadir lógica condicional
content = content.replace(
    '    if agent_name == "MLP (Adam)":\n        agent = DQN_Agent(input_dim, action_dim)\n    else:\n        agent = VolumetricCognitiveBrain(input_dim, action_dim)',
    '''    if agent_name == "MLP (Adam)":
        agent = DQN_Agent(input_dim, action_dim)
    elif agent_name == "VLNN (Symbolic)":
        agent = VolumetricCognitiveBrain(input_dim, action_dim)
        agent.is_symbolic = True  # Flag para saber si consolida
    else:  # VLNN (Standard)
        agent = VolumetricCognitiveBrain(input_dim, action_dim)
        agent.is_symbolic = False  # No consolidará'''
)

# Modificar las llamadas a consolidate_memory para que solo se ejecuten si is_symbolic
content = content.replace(
    '''if episode % 50 == 0:
                agent.consolidate_memory(verbose=(episode % 100 == 0))''',
    '''if episode % 50 == 0 and hasattr(agent, 'is_symbolic') and agent.is_symbolic:
                agent.consolidate_memory(verbose=(episode % 100 == 0))'''
)

content = content.replace(
    '''if episode % 10 == 0:
                agent.consolidate_memory(verbose=False)''',
    '''if episode % 10 == 0 and hasattr(agent, 'is_symbolic') and agent.is_symbolic:
                agent.consolidate_memory(verbose=False)'''
)

# Actualizar prints para distinguir entre los 3 modelos
content = content.replace(
    'if agent_name != "MLP (Adam)" and \'symbols\' in metrics and metrics[\'symbols\']:',
    'if agent_name == "VLNN (Symbolic)" and \'symbols\' in metrics and metrics[\'symbols\']:'
)

with open('c:/projects/VNN_smart/vlnn_symbolic_scaling_experiment.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ Experimento de escalado creado: vlnn_symbolic_scaling_experiment.py")
print("   - 3 agentes: VLNN Standard, VLNN Symbolic, MLP")
print("   - 4 entornos: Jungla (24x24), Desierto (30x30), Antártida (40x40), Patagonia (50x50)")
print("   - Gráficas comparativas de escalado de parámetros")
