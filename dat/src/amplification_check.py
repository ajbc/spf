import sys
from collections import defaultdict
from random import randint, shuffle

dir = sys.argv[1]

# read in network
print "* reading network data"

network = defaultdict(set)
for line in open(dir +'/network.tsv'):
    if ',' in line:
        a, b = [int(x) for x in line.strip().split(',')]
    else:
        a, b = [int(x) for x in line.strip().split('\t')]
    network[a].add(b)
    network[b].add(a)

print "* reading train, test, and validation data"

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

print "  %d unique items in original data" % len(all_items)

print "* finding shared"

# start with those with the fewest number of items
shared_sum = 0
U = 0
for user in sorted(user_items.keys(), key=lambda x: len(user_items[x])):
    shared = set()
    for friend in network[user]:
        for item in user_items[friend]:
            if item in user_items[user]:
                shared.add(item)

    shared_sum += len(shared)*100.0/ len(user_items[user])
    U += 1
print (shared_sum / U), "%  shared"
