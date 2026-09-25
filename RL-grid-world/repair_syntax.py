
import os

target_file = 'c:/projects/VNN_smart/vlnn_symbolic_scaling_experiment.py'

with open(target_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
i = 0
while i < len(lines):
    line = lines[i]
    stripped = line.strip()
    
    # Check for the specific broken print statements we saw
    if stripped == 'print("' and i + 1 < len(lines):
        next_line = lines[i+1]
        # Merge this line and the next one
        # Removing the newline char from the first line
        merged_line = line.rstrip('\n').rstrip('\r') + next_line.lstrip()
        new_lines.append(merged_line)
        i += 2 # Skip next line as we merged it
    elif stripped == 'print(f"' and i + 1 < len(lines):
         # Also check for f-strings if broken similarly
        next_line = lines[i+1]
        merged_line = line.rstrip('\n').rstrip('\r') + next_line.lstrip()
        new_lines.append(merged_line)
        i += 2
    else:
        new_lines.append(line)
        i += 1

with open(target_file, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print("Fixed broken print statements.")
