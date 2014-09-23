import sys
from scipy.special import digamma
import numpy as np
import pickle
from collections import defaultdict
#from scipy.stats import kendalltau
from sklearn.metrics import auc

# parse args
input_file = open(sys.argv[1], 'r')
output = sys.argv[2]
output_file_prec = open(output + ".prec.csv", 'w+')
output_file_user = open(output + ".user.csv", 'w+')
output_file_tau = open(output + ".tau.csv", 'w+')
output_file_stats = open(output + ".stats.csv", 'w+')

# data = iterable of (user, item, rating, prediction, per user ranking)
data = []
ave_precision = defaultdict(float)
ave_precisionN = defaultdict(float)
recall = defaultdict(float)
ave_recall = defaultdict(float)
users = set()
found = 0.0
#rating_count = defaultdict(int)
#user_data = []
taus = []
ave_auc = 0
ave_auc_N = 0
output_file_user.write("user.id,num.heldout,auc,tau,precision.1,precision.5,precision.10,precision.25,precision.50,precision.100,precisionN.1,precisionN.5,precisionN.10,precisionN.25,precisionN.50,precisionN.100\n")

rmse = 0.0
mae = 0.0
rmse_N = 0
ave_crr = 0.0

rankr = 0
for line in input_file:
    tokens = line.split(',')
    #src = tokens.pop(-1).strip()
    #print tokens
    user, item, rating, pred, rank, num_heldout = \
        tuple([float(x.strip()) for x in tokens])
    if num_heldout == 0:
        #print "user %d has no heldout!"% user
        continue
    #if not (src == 'unobs' or src == 'test'):
    #    #print "skipping src of type", src
    #    continue
    #if src == 'test':
    #    print "yay!"
    rankr += 1
    rank = rankr

    if user not in users:
        print user
        if found != 0:
        # add in last user's recall values
            for r in range(len(recall)):
                recall[r] /= found
                ave_recall[r] += recall[r]
           
            if len(recall) > 1:
                user_auc = auc(recall, precision)
                ave_auc += user_auc
                ave_auc_N += auc(recall, precisionN)
            else:
                user_auc = 0 

            ## compute kendall's tau for last user
            # calculate rating ranks
            #rating_rank = {}
            #for r in rating_count.keys():
            #    rating_rank[r] = 1
            #    for i in range(5-int(r)): #TODO: sub max rating
            #        rating_rank[r] += rating_count[5-i]
            
            '''# find count of discorant and concordant pairs
            concordant = 0
            discordant = 0
            for i in range(len(user_data)):
                for j in range(i):
                    x = user_data[i]
                    y = user_data[j]

                    if rating_rank[x[0]] == rating_rank[y[0]] or \
                       x[1] == y[1]:
                        continue

                    if (rating_rank[x[0]] < rating_rank[y[0]] and x[1] < y[1]) or \
                       (rating_rank[x[0]] > rating_rank[y[0]] and x[1] > y[1]):
                        concordant += 1
                    else:
                        discordant += 1

            # calculating tau
            n = len(user_data)
            taus.append((concordant - discordant) / (0.5 * n * (n-1)))
            '''
            #x = []
            #y = []
            #for i in user_data:
            #    x.append(rating_rank[i[0]])
            #    y.append(i[1])
            #tau = kendalltau(x, y)

            #print((concordant - discordant) / (0.5 * n * (n-1)), tau)
            #if tau != 1 and not np.isnan(np.sum(tau)):
            #    taus.append((user, tau[0], tau[1]))
            
            #output_file_user.write("%d,%d,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n" % \
            #    (user,num_heldout,user_auc,tau[0],
            #    precision[0], precision[4], precision[9], precision[24], precision[49], precision[99],
            #    precisionN[0], precisionN[4], precisionN[9], precisionN[24], precisionN[49], precisionN[99]))
            output_file_user.write("%d,%d,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n" % \
                (user,num_heldout,user_auc,
                precision[0], precision[4], precision[9], precision[24], precision[49], precision[99],
                precisionN[0], precisionN[4], precisionN[9], precisionN[24], precisionN[49], precisionN[99]))

        #print "new user %f" % user
        # reset for new user
        users.add(user)
        recall = []
        precision = []
        precisionN = []
        found = 0.0
        #rating_count = defaultdict(int)
        #user_data = []
        rankr = 0


    # precision
    if rating != 0:
        found += 1
        rmse += (rating - pred)**2
        mae += abs(rating - pred)
        rmse_N += 1
        ave_crr += (1.0 / rank)
    #print "user", user, ":",found, "found at threshold",rank
    ave_precision[rank] += found
    ave_precisionN[rank] += found / min(rank, num_heldout)
    precision.append(found / rank)
    precisionN.append(found / min(rank, num_heldout))
    
    # recall
    recall.append(found)

    # kendall's tau
    #rating_count[rating] += 1
    #user_data.append((rating, rank))



for rank in ave_precision:
    output_file_prec.write("%d, %f, %f, %f\n" % (rank, \
           ave_precision[rank] / rank / len(users), \
           ave_precisionN[rank] / len(users), \
           ave_recall[rank] / len(users)))
    '''print (rank, \
           ave_precision[rank] / rank / len(users), \
           ave_recall[rank] / len(users))'''

#ave_tau = 0
#for tau in taus:
#    output_file_tau.write("%d, %f, %f\n" % tau)
#    ave_tau += tau[0]

rmse = np.sqrt(rmse / rmse_N)
mae /= rmse_N
U = len(users)
output_file_stats.write("%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n" % \
    (ave_auc/U, ave_auc_N/U, mae, rmse, ave_crr/U, \
     ave_precision[1]/U, ave_precision[10]/10/U, ave_precision[100]/100/U, \
     ave_precisionN[1]/U, ave_precisionN[10]/10/U, ave_precisionN[100]/100/U))

output_file_stats.write("ave auc: %f\n" % (ave_auc / len(users)))
output_file_stats.write("ave normalized auc: %f\n" % (ave_auc_N / len(users)))
#output_file_stats.write("ave tau: %f\n" % (ave_tau / len(users)))
output_file_stats.write("rmse: %f\n" % (rmse))
output_file_stats.write("ave crr: %f\n" % (ave_crr / len(users)))
output_file_stats.write("ave prec @ 1,10,100: %f %f %f\n" % \
    (ave_precision[1] / len(users), ave_precision[10] / 10 / len(users), ave_precision[100] / 100 /len(users)))
output_file_stats.write("ave normalized prec @ 1,10,100: %f %f %f\n" % \
    (ave_precisionN[1] / len(users), ave_precisionN[10] / 10 / len(users), ave_precisionN[100] / 100 /len(users)))
