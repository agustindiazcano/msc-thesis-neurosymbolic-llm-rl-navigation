with open('c:/projects/VNN_smart/vlnn_symbolic_experiment.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Buscar el bloque de training donde se llama a fuse_neurons y agregar consolidate_memory
# El patrón es: después de prune_dead_neurons y fuse_neurons, agregar consolidate_memory

# En training (alrededor de línea 622-626)
old_training_block = '''        if agent_name != "MLP (Adam)":
            agent.prune_dead_neurons(global_step_counter, max_age=2000)
            if episode % 10 == 0:
                agent.fuse_neurons()'''

new_training_block = '''        if agent_name != "MLP (Adam)":
            agent.prune_dead_neurons(global_step_counter, max_age=2000)
            if episode % 10 == 0:
                agent.fuse_neurons()
            # CONSOLIDACIÓN SIMBÓLICA cada 50 episodios
            if episode % 50 == 0:
                agent.consolidate_memory(verbose=(episode % 100 == 0))'''

content = content.replace(old_training_block, new_training_block)

with open('c:/projects/VNN_smart/vlnn_symbolic_experiment.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ Llamada a consolidate_memory agregada en training loop")
print("   Se ejecutará cada 50 episodios")
print("   Verbose output cada 100 episodios")
