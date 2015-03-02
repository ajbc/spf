import sys
import scipy.io
from collections import defaultdict
import os
from os.path import join, exists
import random

### command line args

ratings_file = sys.argv[1]
network_file = sys.argv[2]
output_dir = sys.argv[3]

splitchar = '\t'


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

ratings = open(ratings_file, 'r')
user_ratings = defaultdict(list)
ur = defaultdict(set)
pop = defaultdict(int)
times = defaultdict(int)
total = 0
for line in ratings:
    user, item, time = [int(x) for x in line.strip().split(splitchar)]
    user_ratings[user].append((item, time))
    times[time] += 1
    ur[user].add(item)
    pop[item] += 1
    total += 1
ratings.close()

trustnetwork = open(network_file, 'r')
network = set()
for line in trustnetwork:
    user, friend = [int(x) for x in line.strip().split(splitchar)]
    if (friend, user) not in network:
        network.add((user, friend))
trustnetwork.close()


### write out everything

if not exists(output_dir):
    os.mkdir(output_dir)

validation_start = 0
test_start = 0
cur = 0.0
for time in sorted(times.keys(),):
    cur += times[time]
    if cur / total >= train and validation_start == 0:
        validation_start = time
        continue
    elif cur / total >= train + valid and test_start == 0 and time != validation_start:
        test_start = time
        break
print validation_start, test_start
train_file = open(join(output_dir, "train.tsv"), 'w+')
valid_file = open(join(output_dir, "validation.tsv"), 'w+')
test_file = open(join(output_dir, "test.tsv"), 'w+')
network_file = open(join(output_dir, "network.tsv"), 'w+')

a = 0
b = 0
c = 0
uinc = set()
for user in user_ratings:
    ratings = user_ratings[user]
    R = len(ratings)
    for i in range(R):
        item, time = ratings[i]
        tr = 0
        va = 0
        te = 0
        if time < validation_start:
            tr += 1
        elif time >= validation_start and time < test_start:
            va += 1
        else:
            te += 1

    if tr < 1: # user must have at least one training item
        print "user %d has no training items" % user
        continue
    uinc.add(user)

    for i in range(R):
        item, time = ratings[i]
        if pop[item] < 2:
            continue
        if time < validation_start:
            train_file.write("%d\t%d\t1\n" % (user, item))
            a += 1
        elif time >= validation_start and time < test_start:
            valid_file.write("%d\t%d\t1\n" % (user, item))
            c += 1
        else:
            test_file.write("%d\t%d\t1\n" % (user, item))
            b += 1

for user, friend in network:
    if user not in uinc or friend not in uinc:
        continue
    if len(ur[user] & ur[friend]) != 0:
        network_file.write("%d\t%d\n" % (user, friend))


train_file.close()
valid_file.close()
test_file.close()
network_file.close()

total = float(a + b + c)
print a, b, c, total
print (a/total, b/total, c/total)
