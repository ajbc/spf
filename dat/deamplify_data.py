import sys
from collections import defaultdict
from random import randint, shuffle, sample

dir = sys.argv[1]
out = sys.argv[2]
per = float(sys.argv[3]) / 100

# read in network
#print "* reading network data"

network = defaultdict(set)
for line in open(dir +'/network.tsv'):
    if ',' in line:
        a, b = [int(x) for x in line.strip().split(',')]
    else:
        a, b = [int(x) for x in line.strip().split('\t')]
    network[a].add(b)
    network[b].add(a)

#print "* reading train, test, and validation data"

user_items = defaultdict(set)
train = defaultdict(set)
test = defaultdict(set)
valid = defaultdict(set)
all_items = set()
for line in open(dir +'/train.tsv'):
    u,i,r = [int(x) for x in line.strip().split('\t')]
    if len(network[u]) == 0:
        continue
    user_items[u].add(i)
    train[u].add(i)
    all_items.add(i)
for line in open(dir +'/test.tsv'):
    u,i,r = [int(x) for x in line.strip().split('\t')]
    if len(network[u]) == 0:
        continue
    user_items[u].add(i)
    test[u].add(i)
    all_items.add(i)
for line in open(dir +'/validation.tsv'):
    u,i,r = [int(x) for x in line.strip().split('\t')]
    if len(network[u]) == 0:
        continue
    user_items[u].add(i)
    valid[u].add(i)
    all_items.add(i)

#print "  %d unique items in original data" % len(all_items)

#print "* exchanging items for each user"
changed = 0
# start with those with the largest number of items
for user in sorted(user_items.keys(), key=lambda x: -len(user_items[x])):
    shared = set()
    omit = set()
    for friend in network[user]:
        for item in user_items[friend]:
            if item in user_items[user]:
                shared.add(item)
            else:
                omit.add(item)

    candidates = all_items - omit - shared
    items = list(user_items[user] - set(candidates))
    shuffle(items)
    #if len(shared) != 0:
    #    print "user", user, "has", (len(shared) * 100.0 / len(user_items[user])), "%% items shared"
    for item in items:
        if len(shared) * 1.0 / len(user_items[user]) <= per or len(candidates) == 0:
            break


        pick = sample(candidates, 1)[0]
        changed += 1

        if item in train[user]:
            train[user].remove(item)
            train[user].add(pick)
        if item in test[user]:
            test[user].remove(item)
            test[user].add(pick)
        if item in valid[user]:
            valid[user].remove(item)
            valid[user].add(pick)

        candidates.remove(pick)
        user_items[user].remove(item)
        user_items[user].add(pick)


    #print "user", user, "has", (len(shared) * 100.0 / len(user_items[user])), "%% items shared"
all_items = set()
for user in user_items.keys():
    all_items = all_items | user_items[user]
#print "  %d unique items in amplified data" % len(all_items)
print changed, "changed items"

f = open(out +'/train.tsv', 'w+')
for user in train:
    for item in train[user]:
        f.write('%d\t%d\t1\n' % (user,item))
f.close()
#print "* done writing out training data"

f = open(out +'/test.tsv', 'w+')
for user in test:
    for item in test[user]:
        f.write('%d\t%d\t1\n' % (user,item))
f.close()
#print "* done writing out testing data"

f = open(out +'/validation.tsv', 'w+')
for user in valid:
    for item in valid[user]:
        f.write('%d\t%d\t1\n' % (user,item))
f.close()
#print "* done writing out validation data"
