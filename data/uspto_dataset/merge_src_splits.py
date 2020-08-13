lines = []
with open ('src-train-split-1.txt', 'r') as f:
    for line in f.readlines():
        lines.append(line)
with open ('src-train-split-2.txt', 'r') as f:
    for line in f.readlines():
        lines.append(line)
with open('src-train.txt', 'w') as f:
    f.write(''.join(lines))
