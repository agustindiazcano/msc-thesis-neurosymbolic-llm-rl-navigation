with open('c:/projects/VNN_smart/vlnn_complete_experiment.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix escaped quotes
content = content.replace("\\'", "'")

# Fix the print line to include collisions
content = content.replace(
    'print(f"[{agent_name}] Ep {episode} (Map {size}x{size}): WinRate {avg_win:.2f} | Params: {metrics[\'params\'][-1]}")',
    'print(f"[{agent_name}] Ep {episode} (Map {size}x{size}): WinRate {avg_win:.2f} | Avg Collisions: {avg_collisions:.1f} | Params: {metrics[\'params\'][-1]}")'
)

with open('c:/projects/VNN_smart/vlnn_complete_experiment.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ Fixed")
