import sys
from scipy.special import digamma
from scipy.sparse import lil_matrix, dok_matrix
import numpy as np
from collections import defaultdict

import os
from os.path import isfile, join

data_stem = sys.argv[1]
fit_stem = sys.argv[2]
K = int(sys.argv[3]) #+ 1 # for intercept 1 is added INTERTAG
#TODO: add non-binary option

users = {}
items = {}
for line in open(join(data_stem, "user_map_sorec.dat"), 'r'):
    user, id = [int(i.strip()) for i in line.split(',')]
    users[user] = id
for line in open(join(data_stem, "item_map_sorec.dat"), 'r'):
    item, id = [int(i.strip()) for i in line.split(',')]
    items[item] = id

user_data_train = defaultdict(set)
for line in open(join(data_stem, "train.tsv"), 'r'):
    triplet = tuple([int(x.strip()) for x in line.split('\t')])
    user, item, rating = triplet

    if rating == 0:
        continue
    #if model.binary:
    #    self.train_triplets.append((user, item))
    #else:
    #    self.train_triplets.append(triplet)

    if user not in users:
        continue

    if item not in items:
        continue

    user_data_train[user].add(item)

for line in open(join(data_stem, "validation.tsv"), 'r'):
    triplet = tuple([int(x.strip()) for x in line.split('\t')])
    user, item, rating = triplet

    if rating == 0:
        continue
    if user not in users:
        continue
    if item not in items:
        continue

    user_data_train[user].add(item)


#if model.trust or model.iat:
#    data.read_network(model, data_stem + "/network.tsv")
#print model.user_count, model.item_count

# read in learned CTR MF params

theta = np.zeros((len(users), K))
beta = np.zeros((len(items), K))
fU = open(join(fit_stem, 'final-U.dat'))
fV = open(join(fit_stem, 'final-V.dat'))
i = 0
for line in fU:
    theta[i,:] = [float(x.strip()) for x in line.strip().split(' ')]
    i += 1
    if i >= len(users):
        print "more lines than users"
        break
i = 0
for line in fV:
    if i >= len(items):
        break
    beta[i,:] = [float(x.strip()) for x in line.strip().split(' ')]
    if i >= len(items)+ len(users):
        print "more lines than items + users"
        break
    i += 1
fU.close()
fV.close()
print "done reading in theta and beta"


np.random.seed(42)
print len(users)
#print sorted(users, key=lambda x: users[x])[:10]
#users_set = set(sorted(users, key=lambda x: np.random.rand())[:1000])
#items_set = set(sorted(items, key=lambda x: -data.item_counts[model.items[x]])[:10000])
#users = set([users[user] for user in users_set])
#items = set([items[item] for item in items_set])
#print "subsetted item an users done."

# read in relevant test data
user_data = defaultdict(dict)
for line in open(join(data_stem, "test.tsv"), 'r'):
    user, item, rating = \
        tuple([int(x.strip()) for x in line.split('\t')])
    if user not in users or item not in items:
        continue
    #if (users_set and user not in users_set) or (items_set and item not in items_set):
    #    continue
    if rating != 0:
        user_data[user][item] = rating

print "evaluating predictions for each user-item pair"

print "creating rankings for each user..."
f = open(join(fit_stem, "rankings.out"), 'w+')
for user in users:
    if user not in user_data: # ignore users with no heldout
        continue
    #print user
    preds = {}
    for item in items:
        preds[item] = sum(theta[users[user]] * beta[items[item]])

    rank = 1
    for item in sorted(preds, key=lambda i:-preds[i]):
        if item in user_data_train[user]:
            continue
        pred = preds[item]
        rating = user_data[user][item] if item in \
            user_data[user] else 0
        f.write("%d, %d, %d, %f, %d\n" % \
            (user, item, rating, pred, rank))
        rank += 1
f.close()
