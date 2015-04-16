import sys
from collections import defaultdict
import random

random.seed(11)


path = sys.argv[1]
undir = (len(sys.argv) == 3)

users = set()
items = set()
fout = open(path +'/ratings.dat', 'w+')
user_items = defaultdict(set)
user_counts = defaultdict(int)
for line in open(path + '/train.tsv'):
    user, item, rating = [int(x.strip()) for x in line.split('\t')]
    fout.write("%d\t%d\t%d\n" % (user, item, rating))
    items.add(item)
    users.add(user)
    user_items[user].add(item)
    user_counts[user] += 1
for line in open(path + '/validation.tsv'):
    user, item, rating = [int(x.strip()) for x in line.split('\t')]
    if user in users and item in items:
        fout.write("%d\t%d\t%d\n" % (user, item, rating))
        user_items[user].add(item)
        user_counts[user] += 1
test_users = set()
test_items = set()
ratings = dict()
for line in open(path + '/test.tsv'):
    user, item, rating = [int(x.strip()) for x in line.split('\t')]
    if user in users and item in items:
        test_users.add(user)
        test_items.add(item)
        ratings[(user,item)] = rating
        user_items[user].add(item)

test_max = 60000000
test_index = 1
test_count = 0
fout_test = open(path +'/test-%02d.dat' % test_index, 'w+')

print "starting test batch 1"
for user in test_users:
    all_items = list(items)
    random.shuffle(all_items)
    while user_counts[user] > 0 and len(all_items) != 0:
        item = all_items.pop()
        if item not in user_items:
            fout.write("%d\t%d\t0\n" % (user, item))
            user_counts[user] -= 1

    for item in test_items:
        if (user,item) in ratings:
            fout_test.write("%d\t%d\t%d\n" % (user, item, rating))
        else:
            fout_test.write("%d\t%d\t0\n" % (user, item))
        test_count += 1

    if test_count > test_max:
        fout_test.close()
        test_index += 1
        fout_test = open(path +'/test-%02d.dat' % test_index, 'w+')
        print "starting test batch %d" % test_index
        test_count = 0

fout.close()
fout_test.close()

fout = open(path +'/network.dat', 'w+')
for line in open(path + '/network.tsv'):
    user, friend = [int(x.strip()) for x in line.split('\t')]
    if user in users and friend in users:
        fout.write("%d\t%d\t1\n" % (user, friend))
        if undir:
            fout.write("%d\t%d\t1\n" % (friend, user))
fout.close()
