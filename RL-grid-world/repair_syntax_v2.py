
import os

target_file = 'c:/projects/VNN_smart/vlnn_symbolic_scaling_experiment.py'

with open(target_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
i = 0
while i < len(lines):
    line = lines[i]
    stripped = line.strip()
    
    # Check for the specific problematic pattern: a line that is just '")'
    if i + 1 < len(lines):
        next_line = lines[i+1]
        next_stripped = next_line.strip()
        
        if next_stripped == '")':
            # This is extremely likely a broken print statement split
            merged_line = line.rstrip('\n').rstrip('\r') + next_stripped
            new_lines.append(merged_line + '\n')
            i += 2 # Skip next line
            continue
            
    new_lines.append(line)
    i += 1

with open(target_file, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print("Fixed broken print statements v2.")
