import sys
from collections import defaultdict

path = sys.argv[1]

fin = open(path +'/train.tsv')
fout_items = open(path +'/items.dat', 'w+')
fout_users = open(path +'/users.dat', 'w+')

users = defaultdict(list)
items = defaultdict(list)
for line in fin:
    user, item, rating = [int(x.strip()) for x in line.split('\t')]
    users[user].append(item)
    items[item].append(user)

umap = {}
imap = {}
fmap_items = open(path +'/item_map.dat', 'w+')
fmap_users = open(path +'/user_map.dat', 'w+')
for user in users:
    umap[user] = len(umap)
    fmap_users.write("%d,%d\n" % (user, umap[user]))
for item in items:
    imap[item] = len(imap)
    fmap_items.write("%d,%d\n" % (item, imap[item]))
fmap_users.close()
fmap_items.close()

for user in sorted(users, key=lambda x: umap[x]):
    line = str(len(users[user]))
    for item in users[user]:
        line += ' ' + str(imap[item])
    fout_users.write(line + '\n')

for item in sorted(items, key=lambda x: imap[x]):
    line = str(len(items[item]))
    for user in items[item]:
        line += ' ' + str(umap[user])
    fout_items.write(line + '\n')

fout_items.close()
fout_users.close()

