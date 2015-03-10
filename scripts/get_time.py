import sys

t = 0
for line in open(sys.argv[1]):
    toks = line.split('\t')
    if toks[0] == "user" or toks[0] == "sys":
        m,s = [float(i) for i in toks[1][:-2].split('m')]
        t += m*60 + s
print t
