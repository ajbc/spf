import sys
from collections import defaultdict

path = sys.argv[1]
undir = (len(sys.argv) == 3)

users = set()
items = set()
fout = open(path +'/ratings.dat', 'w+')
for line in open(path + '/train.tsv'):
    user, item, rating = [int(x.strip()) for x in line.split('\t')]
    fout.write("%d %d %d\n" % (user, item, rating))
    items.add(item)
    users.add(user)
for line in open(path + '/validation.tsv'):
    user, item, rating = [int(x.strip()) for x in line.split('\t')]
    if user in users and item in items:
        fout.write("%d %d %d\n" % (user, item, rating))
fout.close()
test_users = set()
test_items = set()
ratings = dict()
fout = open(path +'/test.dat', 'w+')
for line in open(path + '/test.tsv'):
    user, item, rating = [int(x.strip()) for x in line.split('\t')]
    if user in users and item in items:
        test_users.add(user)
        test_items.add(item)
        ratings[(user,item)] = rating
for user in test_users:
    for item in test_items:
        if (user,item) in ratings:
            fout.write("%d %d %d\n" % (user, item, rating))
        else:
            fout.write("%d %d 0\n" % (user, item))
fout.close()

fout = open(path +'/network.dat', 'w+')
for line in open(path + '/network.tsv'):
    user, friend = [int(x.strip()) for x in line.split('\t')]
    if user in users and friend in users:
        fout.write("%d %d 1\n" % (user, friend))
        if undir:
            fout.write("%d %d 1\n" % (friend, user))
fout.close()
