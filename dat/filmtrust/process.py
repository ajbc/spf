import sys
import scipy.io
from collections import defaultdict
import os
from os.path import join, exists
import random

### command line args

raw_dir = sys.argv[1]
output_dir = sys.argv[2]


### split math

train = 89
test = 10
valid = 1

total = float(train + test + valid)
train /= total
test /= total
valid /= total

print (train, test, valid)

random.seed(11)


### read in everything

ratings = open(join(raw_dir, "ratings.txt"), 'r')
user_ratings = defaultdict(dict)
ur = defaultdict(set)
for line in ratings:
    user, item, rating = [int(float(x)) for x in line.strip().split()]
    user_ratings[user][item] = rating
    ur[user].add(item)
ratings.close()

trustnetwork = open(join(raw_dir, "trust.txt"), 'r')
network = set()
for line in trustnetwork:
    user, friend, value = [int(x) for x in line.strip().split()]
    network.add((user, friend))
trustnetwork.close()


### write out everything

if not exists(output_dir):
    os.mkdir(output_dir)

train_file = open(join(output_dir, "train.tsv"), 'w+')
valid_file = open(join(output_dir, "validation.tsv"), 'w+')
test_file = open(join(output_dir, "test.tsv"), 'w+')
network_file = open(join(output_dir, "network.tsv"), 'w+')

a = 0
b = 0
c = 0
for user in user_ratings:
    ratings = user_ratings[user].items()
    random.shuffle(ratings)
    R = len(ratings)
    for i in range(R):
        item, rating = ratings[i]
        r = i
        if (r < test * R and not (r+1) < test * R) or \
           (r < (test + valid) * R and not (r+1) < (test + valid) * R):
            r += random.random()

        if r < test * R:
            test_file.write("%d\t%d\t%d\n" % (user, item, rating))
            b += 1
        elif r < (test + valid) * R:
            valid_file.write("%d\t%d\t%d\n" % (user, item, rating))
            c += 1
        else:
            train_file.write("%d\t%d\t%d\n" % (user, item, rating))
            a += 1

for user, friend in network:
    if len(ur[user] & ur[friend]) != 0:
        network_file.write("%d\t%d\n" % (user, friend))


train_file.close()
valid_file.close()
test_file.close()
network_file.close()

total = float(a + b + c)
print (a/total, b/total, c/total)
