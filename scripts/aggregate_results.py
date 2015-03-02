import fnmatch
import os, sys
from os.path import isdir, join

m = sys.argv[1]
outfile = sys.argv[2]
k = int(sys.argv[3])
fout = open(outfile, 'w+')
fout.write("model,k,metric,value\n")

for model in os.listdir(m):
    if not isdir(join(m, model)):
        continue
    for file in os.listdir(join(m, model)):
        if isdir(join(join(m, model), file)):
            for f in os.listdir(join(join(m, model), file)):
                if f == 'eval_summary_final.dat':
                    fname = join(join(join(m, model), file), f)
                    for line in open(fname).readlines()[1:]:
                        tokens = line.split('\t')
                        fout.write("%s,%d,%s,%s\n" % (file, \
                            k if model==0 else int(model), \
                            tokens[0], tokens[1]))

            continue
        if file == 'eval_summary_final.dat':
            fname = join(join(m, model), file)
            for line in open(fname).readlines()[1:]:
                tokens = line.split('\t')
                fout.write("%s,%d,%s,%s\n" % (model, k, tokens[0], tokens[1]))
fout.close()
