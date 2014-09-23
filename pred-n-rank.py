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
model = load_model(fit_stem+"/model_settings.dat")
model.nofdiv=False
model.binary=True

# load data
data = dataset(model)

data.read_ratings(model, data_stem + "/train.tsv") 
if model.trust: 
    data.read_network(model, data_stem + "/network.tsv")
print model.user_count, model.item_count

# load parameters
last_params = ''
for f in [ f for f in os.listdir(fit_stem) if isfile(join(fit_stem,f)) ]:
    if f.startswith('params-'):
        last_params = join(fit_stem, f)
        break

params = ptfstore.load(last_params, model)


print "done reading in model"


np.random.seed(42)
print len(model.users)
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
    if user not in model.users or item not in model.items:
        continue
    #if (users_set and user not in users_set) or (items_set and item not in items_set):
    #    continue
    if rating != 0:
        test_ratings.append((model.users[user], model.items[item], rating))
        user_data[model.users[user]][model.items[item]] = rating

userpreds_random = defaultdict(dict)
userpreds_popularity = defaultdict(dict)
print "evaluating predictions for each user-item pair"

print "creating rankings for each user..."
f = open(fit_stem + "/rankings.out", 'w+')
for user in model.users:
    print user
    preds = {}
    for item in model.items:
        pred = 0

        if model.intercept:
            pred += params.inter[model.items[item]]

        if model.MF:
            M = sum(params.theta[model.users[user]] * \
                params.beta[model.items[item]])
        
        if model.MF:
            pred += M
        
        if model.trust:
            T = 0
            for vser in data.friends[model.users[user]]:
                if model.binary:
                    rating_v = 1 if item in data.user_data[vser] else 0
                else:
                    rating_v = data.sparse_ratings.get(model.items[item], vser) 
                if rating_v != 0:
                    T += params.tau.get(model.users[user], vser) * rating_v
        
            if not model.nofdiv and data.friend_counts[model.users[user]][model.items[item]] != 0:
                T /= data.friend_counts[model.users[user]][model.items[item]]

        if model.trust:
            pred += T

        preds[item] = pred

    rank = 1
    exclude = set()
    if model.binary:
        exlude = set(data.user_data[model.users[user]])
    else:
        for item,rating in data.user_data[model.users[user]]:
            exclude.add(item)
    for item in sorted(preds, key=lambda i:-preds[i]):
        if model.items[item] in exclude:
            continue
        pred = preds[item]
        rating = user_data[model.users[user]][model.items[item]] if \
            model.items[item] in user_data[model.users[user]] else 0
        f.write("%d, %d, %d, %f, %d, %d\n" % \
            (user, item, rating, pred, rank, len(user_data[model.users[user]])))
        rank += 1
f.close()
