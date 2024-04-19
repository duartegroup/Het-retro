lines = []
with open ('reactant-train-split-1.txt', 'r') as f:
    for line in f.readlines():
        lines.append(line)
with open ('reactant-train-split-2.txt', 'r') as f:
    for line in f.readlines():
        lines.append(line)
with open('reactant-train.txt', 'w') as f:
    f.write(''.join(lines))
