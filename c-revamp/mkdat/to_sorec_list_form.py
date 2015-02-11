import sys
from collections import defaultdict

path = sys.argv[1]

fin = open(path +'/train.tsv')
finn = open(path +'/network.tsv')
fout_items = open(path +'/items_sorec.dat', 'w+')
fout_users = open(path +'/users_sorec.dat', 'w+')

users = defaultdict(list)
items = set()
for line in fin:
    user, item, rating = [int(x.strip()) for x in line.split('\t')]
    users[user].append(item)
    items.add(item)


umap = {}
imap = {}
fmap_items = open(path +'/item_map_sorec.dat', 'w+')
fmap_users = open(path +'/user_map_sorec.dat', 'w+') # this should be the same
for user in users:
    umap[user] = len(umap)
    fmap_users.write("%d,%d\n" % (user, umap[user]))
for item in items:
    imap[item] = len(imap)
    fmap_items.write("%d,%d\n" % (item, imap[item]))
fmap_users.close()
fmap_items.close()


user_data = defaultdict(list)
item_data = defaultdict(list)
for user in users:
    for item in users[user]:
        user_data[umap[user]].append(imap[item])
        item_data[imap[item]].append(umap[user])

for line in finn:
    user, friend = [int(x.strip()) for x in line.split('\t')]
    if user not in umap or friend not in umap:
        continue

    user_data[umap[user]].append(len(imap) + umap[friend])
    item_data[len(imap) + umap[friend]].append(umap[user])

    # undirected
    #user_data[umap[friend]].append(len(imap) + umap[user])
    #item_data[len(imap) + umap[user]].append(umap[friend])

for item in sorted(items, key=lambda x: imap[x]):
    line = str(len(item_data[imap[item]]))
    for user in item_data[imap[item]]:
        line += ' ' + str(user)
    fout_items.write(line + '\n')

for user in sorted(users, key=lambda x: umap[x]):
    line = str(len(user_data[umap[user]]))
    for item in user_data[umap[user]]:
        line += ' ' + str(item)
    fout_users.write(line + '\n')

    line = str(len(item_data[len(imap) + umap[user]]))
    for user in item_data[len(imap) + umap[user]]:
        line += ' ' + str(user)
    fout_items.write(line + '\n')

fout_items.close()
fout_users.close()

