import sys
from scipy.special import digamma
import numpy as np
import pickle
from collections import defaultdict
from os.path import join

# parse args
input_file = open(sys.argv[1], 'r')
output_dir = sys.argv[2]

# metrics
rmse = 0
mae = 0
N = 0.0
U = 0.0
ave_crr = 0
ave_rr = 0
ave_rank = 0.0
ave_p = defaultdict(float)

# create a per-user summary file
fout_user = open(join(output_dir, 'user_eval.tsv'), 'w+')
fout_user.write("user.id\trmse\tmae\tfirst\trr\tcrr\tp.1\tp.10\tp.100\n")

users = set()
# per user metrics
found = 0.0 # float so division plays nice
crr = 0
first = 0
rr = 0
user_rmse = 0
user_mae = 0
p = defaultdict(float) # precision

USER = -1
fout_userD = open(join(output_dir, 'user_10047490.tsv'), 'w+')
for line in input_file:
    user, item, rating, pred, rank = line.split(',')
    user = int(user.strip())
    if user not in users:
        #print user
        if USER != -1:
            # log old user
            rmse += user_rmse
            mae += user_mae
            N += found
            ave_crr += crr
            ave_rr += rr
            ave_rank += first

            # per user stats
            if found == 0:
                fout_user.write("%d\t-1\t-1\t-1\t0\t0\t0\t0\t0\n" % USER)
                #    continue
            else:#if found != 0:
                fout_user.write("%d\t%f\t%f\t%d\t%f\t%f\t%f\t%f\t%f\n" %
                    (USER, (user_rmse/found), (user_mae/found), first, rr, \
                    crr, p[1], p[10], p[100]))
            U += 1

        # prep for new user
        users.add(user)
        USER = user

        found = 0.0 # float so division plays nice
        crr = 0
        first = 0
        rr = 0
        user_rmse = 0
        user_mae = 0
        p = defaultdict(float) # precision

    rating = int(rating.strip())

    rank = int(rank.strip())
    prediction = float(pred.strip())
    if rating != 0:
        found += 1
        #prediction += 1e-10 # for social only
        #likelihood +=


        user_rmse += (rating - prediction)**2
        user_mae += abs(rating - prediction)
        crr += (1.0 / rank)
        if found == 1:
            first = rank
            rr = (1.0 / rank)

    if rank == 1 or rank == 10 or rank == 100:
        p[rank] = found / rank
        ave_p[rank] += p[rank]
    item = int(item)
    if user == 10047490:
        fout_userD.write("%d\t%s %d\t%f\n" % (rank, "H" if rating != 0 else " ", item, prediction))
        '''fout.write('\t%f\n' % self.params.inter[self.data.items[item]])
        fout.write('\t%f:' % (sum(self.params.theta[self.data.users[user]] * \
                    self.params.beta[self.data.items[item]])))
        for comp in self.params.theta[self.data.users[user]] * self.params.beta[self.data.items[item]]:
            fout.write(' %f' % comp)
        fout.write('\n')'''


# log lat user
rmse += user_rmse
mae += user_mae
N += found
ave_crr += crr
ave_rr += rr
ave_rank += first

# per user stats
if found == 0:
    fout_user.write("%d\t-1\t-1\t-1\t0\t0\t0\t0\t0\n" % USER)
else:
    fout_user.write("%d\t%f\t%f\t%d\t%f\t%f\t%f\t%f\t%f\n" %
        (USER, (user_rmse/found), (user_mae/found), first, rr, crr, \
        p[1], p[10], p[100]))

fout_user.close()

# overall stats
fout_summary = open(join(output_dir, 'summary_eval.dat'), 'w+')
print rmse, N
fout_summary.write("rmse\t%f\n" % (rmse/N))
fout_summary.write("mae\t%f\n" % (mae/N))
fout_summary.write("rank\t%f\n" % (ave_rank/U))
fout_summary.write("RR\t%f\n" % (ave_rr/U))
fout_summary.write("CRR\t%f\n" % (ave_crr/U))
fout_summary.write("p.1\t%f\n" % (ave_p[1]/U))
fout_summary.write("p.10\t%f\n" % (ave_p[10]/U))
fout_summary.write("p.100\t%f\n" % (ave_p[100]/U))
fout_summary.close()
