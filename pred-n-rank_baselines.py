import sys
from scipy.special import digamma
from scipy.sparse import lil_matrix, dok_matrix
import numpy as np
from collections import defaultdict

import os
from os.path import isfile, join

data_stem = sys.argv[1]
output_stem = sys.argv[2]
#TODO: add non-binary option

users = {}
items = {}
user_data_train = defaultdict(dict)
item_pop = defaultdict(int)
for line in open(data_stem + "/train.tsv", 'r'):
    triplet = tuple([int(x.strip()) for x in line.split('\t')])
    user, item, rating = triplet

    if rating == 0:
        continue
    #if model.binary:
    #    self.train_triplets.append((user, item))
    #else:
    #    self.train_triplets.append(triplet)

    if user not in users:
        users[user] = len(users)
    
    if item not in items:
        items[item] = len(items)

    user_data_train[users[user]][items[item]] = rating
    item_pop[items[item]] += 1

#if model.trust or model.iat: 
#    data.read_network(model, data_stem + "/network.tsv")
#print model.user_count, model.item_count


np.random.seed(42)
print len(users)
#print sorted(users, key=lambda x: users[x])[:10]
#users_set = set(sorted(users, key=lambda x: np.random.rand())[:1000])
#items_set = set(sorted(items, key=lambda x: -data.item_counts[model.items[x]])[:10000])
#users = set([users[user] for user in users_set])
#items = set([items[item] for item in items_set])
#print "subsetted item an users done."

# read in relevant test data
test_ratings = [] #dict_matrix(int, len(users_set), len(items_set))
user_data = defaultdict(dict)
for line in open(data_stem + "/test.tsv", 'r'):
    user, item, rating = \
        tuple([int(x.strip()) for x in line.split('\t')])
    if user not in users or item not in items:
        print "skipping", user, item
        continue
    #if (users_set and user not in users_set) or (items_set and item not in items_set):
    #    continue
    if rating != 0:
        test_ratings.append((users[user], items[item], rating))
        user_data[users[user]][items[item]] = rating

print "evaluating predictions for each user-item pair"

print "creating rankings for each user..."
f1 = open(output_stem + "/rankings-rand.out", 'w+')
f2 = open(output_stem + "/rankings-pop.out", 'w+')
#f = open("preds_n_rankings.csv", 'w+')
for userO in users:
    userpreds_random = {}
    userpreds_popularity = {}
    user = users[userO]
    if len(user_data[user]) == 0:
        continue
    for itemO in items:
        item = items[itemO]
        if user in user_data_train and item in user_data_train[user]:
            print "skipping", user, item, "because in training data"
            continue
        rating = user_data[user][item] if \
            user in user_data and item in user_data[user] \
            else 0
        userpreds_random[itemO] = (rating, float(np.random.rand()))
        userpreds_popularity[itemO] = \
            (rating, float(item_pop[item]))
    
    print "creating a ranking for user %d" % user
    rank_r = 1
    rank_p = 1
    for item in sorted(userpreds_random, key=lambda i:-userpreds_random[i][1]):
        rating, pred = userpreds_random[item]
        print "%d, %d, %d, %f, %d, %d\n" % \
            (userO, item, rating, pred, rank_r, len(user_data[user]))
        f1.write("%d, %d, %d, %f, %d, %d\n" % \
            (userO, item, rating, pred, rank_r, len(user_data[user])))
        rank_r += 1
    for item in sorted(userpreds_popularity, key=lambda i:-userpreds_popularity[i][1]):
        rating, pred = userpreds_popularity[item]
        f2.write("%d, %d, %d, %f, %d, %d\n" % \
            (userO, item, rating, pred, rank_p, len(user_data[user])))
        rank_p += 1
f1.close()
f2.close()
