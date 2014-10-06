import sys
from collections import defaultdict

activity_filename = sys.argv[1]
network_filename = sys.argv[2]

# read in network
print "* reading network data"
network = defaultdict(set)
for line in open(network_filename):
    if ',' in line:
        a, b = [int(x) for x in line.strip().split(',')]
    else:
        a, b = [int(x) for x in line.strip().split('\t')]
    network[a].add(b)
    network[b].add(a)

print "* reading activity data"

user_items = defaultdict(set)
train = defaultdict(set)
test = defaultdict(set)
valid = defaultdict(set)
all_items = set()
for line in open(activity_filename):
    u,i,r = [int(x) for x in line.strip().split('\t')]
    if len(network[u]) == 0:
        continue
    user_items[u].add(i)
    train[u].add(i)
    all_items.add(i)

print "* finding shared"

# start with those with the fewest number of items
shared_sum = 0
U = 0
for user in user_items.keys():
    shared = set()
    for friend in network[user]:
        for item in user_items[friend]:
            if item in user_items[user]:
                shared.add(item)

    shared_sum += len(shared)*100.0/ len(user_items[user])
    U += 1
print (shared_sum / U), "%  shared"
