import sys
from scipy.special import digamma
from scipy.sparse import lil_matrix, dok_matrix
import numpy as np
from collections import defaultdict

import os
from os.path import isfile, join

sys.path.insert(0, '/home/statler/achaney/spf/src/spf')
from ptf import *
from ptfstore import *

data_stem = sys.argv[1]
fit_stem = sys.argv[2]

# load model
model, directed, binary, user_count, item_count, user_mapping, item_mapping = \
    load_model(fit_stem+"/model_settings.dat")
model.nofdiv=False

# load data
data = dataset(user_mapping, item_mapping, binary, directed)

data.read_ratings(data_stem + "/train.tsv")
if model.trust:
    data.read_network(data_stem + "/network.tsv")
print data.user_count, data.item_count

# load parameters
last_params = ''
for f in [ f for f in os.listdir(fit_stem) if isfile(join(fit_stem,f)) ]:
    if f.startswith('params-'):
        last_params = join(fit_stem, f)
        break

params = ptfstore.load(last_params, model, data)


print "done reading in model"


np.random.seed(42)
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
    if user not in data.users or item not in data.items:
        continue
    #if (users_set and user not in users_set) or (items_set and item not in items_set):
    #    continue
    if rating != 0:
        test_ratings.append((data.users[user], data.items[item], rating))
        user_data[data.users[user]][data.items[item]] = rating

userpreds_random = defaultdict(dict)
userpreds_popularity = defaultdict(dict)
print "evaluating predictions for each user-item pair"

print "creating rankings for each user..."
f = open(fit_stem + "/rankings.out", 'w+')
for user in data.users:
    print user
    preds = {}
    for item in data.items:
        pred = 0

        if model.intercept:
            pred += params.inter[data.items[item]]

        if model.MF:
            M = sum(params.theta[data.users[user]] * \
                params.beta[data.items[item]])

        if model.MF:
            pred += M

        if model.trust:
            T = 0
            for vser in data.friends[data.users[user]]:
                if data.binary:
                    rating_v = 1 if item in data.user_data[vser] else 0
                else:
                    rating_v = data.sparse_ratings.get(data.items[item], vser)
                if rating_v != 0:
                    T += params.tau.get(data.users[user], vser) * rating_v

            if not model.nofdiv and data.friend_counts[data.users[user]][data.items[item]] != 0:
                T /= data.friend_counts[data.users[user]][data.items[item]]

        if model.trust:
            pred += T

        preds[item] = pred

    rank = 1
    exclude = set()
    if data.binary:
        exlude = set(data.user_data[data.users[user]])
    else:
        for item,rating in data.user_data[data.users[user]]:
            exclude.add(item)
    for item in sorted(preds, key=lambda i:-preds[i]):
        if data.items[item] in exclude:
            continue
        pred = preds[item]
        rating = user_data[data.users[user]][data.items[item]] if \
            data.items[item] in user_data[data.users[user]] else 0
        f.write("%d, %d, %d, %f, %d, %d\n" % \
            (user, item, rating, pred, rank, len(user_data[data.users[user]])))
        rank += 1
f.close()
