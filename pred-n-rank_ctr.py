import sys
from scipy.special import digamma
from scipy.sparse import lil_matrix, dok_matrix
import numpy as np
from collections import defaultdict

import os
from os.path import isfile, join

data_stem = sys.argv[1]
fit_stem = sys.argv[2]
K = int(sys.argv[3])

users = {}
items = {}
for line in open(join(data_stem, "user_map.dat"), 'r'):
    user, id = [int(i.strip()) for i in line.split(',')]
    users[user] = id
for line in open(join(data_stem, "item_map.dat"), 'r'):
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
    #print i
    i += 1
    if i > len(users):
        print "more lines than users"
        break
i = 0
for line in fV:
    beta[i,:] = [float(x.strip()) for x in line.strip().split(' ')]
    if i > len(items):
        print "more lines than items"
        break
    i += 1
fU.close()
fV.close()
print "done reading in theta and beta"

#print len(users), len(items)
#print sorted(users, key=lambda x: users[x])[:10]
#users_set = set(sorted(users, key=lambda x: np.random.rand())[:1000])
#items_set = set(sorted(items, key=lambda x: -data.item_counts[model.items[x]])[:10000])
#users = set([users[user] for user in users_set])
#items = set([items[item] for item in items_set])
#print "subsetted item an users done."
np.random.seed(42)
user_set = users
item_set = set()
if len(users) > 10000:
    user_set = set(sorted(sorted(users.keys()), key=lambda x: np.random.rand())[:10000])

# read in relevant test data
test_ratings = [] #dict_matrix(int, len(users_set), len(items_set))
user_data = defaultdict(dict)
final_users = set()
for line in open(join(data_stem, "test.tsv"), 'r'):
    user, item, rating = \
        tuple([int(x.strip()) for x in line.split('\t')])
    if user not in user_set or item not in items:
        continue
    if rating != 0:
        test_ratings.append((user, item, rating))
        user_data[user][item] = rating
        final_users.add(user)
        item_set.add(item)
user_set = final_users

print "subsetted users (%d) and items (%d)." % (len(user_set), len(item_set))
print len(user_data), "users to evaluate"

print "evaluating predictions for each user-item pair"

print "creating rankings for each user..."
f = open(join(fit_stem, "rankings.out"), 'w+')
for user in users:
    #print user
    if user not in user_data: # ignore users with no heldout
        #print "user has no heldout"
        continue
    #else:
    #    print "good to go!"
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
