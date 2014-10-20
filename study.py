import argparse
import multiprocessing
from os import mkdir, getcwd
from os.path import exists, join
import numpy as np
import random
from ptf import *
from infer_svi import *

#tmp
#import time

#VERBOSE = False



def get_eval_sets(data):
    # so random subsets are always the same
    lr = random.Random()
    lr.seed(42)

    # return unmapped (original) sets of users & items
    return data.users.keys(), data.items.keys()


### Model study classes ###

class BaselineStudy(multiprocessing.Process):
    def __init__(self, data, out_dir, seed, K=0, SVI=False):
        multiprocessing.Process.__init__(self)
        self.data = data
        self.out_dir = out_dir
        self.K = K
        self.SVI = SVI

        # set up local random generation
        self.lr = random.Random()
        self.lr.seed(seed)#1412616690)

        # create output dir if needed
        if not exists(self.out_dir):
            mkdir(self.out_dir)

        print "initializing study at", self.out_dir

    def fit(self):
        return

    def pred(self, user, item, details=False):
        return self.lr.random()

    def intercept(self, item):
        return 0

    def printWhy(self, user, item, fout):
        fout.write('\twhatevs.\n')

    def eval(self):

        # pick a set of items and users
        users, items = get_eval_sets(self.data)

        # metrics
        rmse = 0
        mae = 0
        N = 0.0
        U = 0.0
        ave_crr = 0
        ave_rr = 0
        ave_rank = 0.0
        ave_p = defaultdict(float)
        ave_pN = defaultdict(float)

        # create a per-user summary file
        fout_user = open(join(self.out_dir, 'user_eval.tsv'), 'w+')
        fout_user.write("user.id\trmse\tmae\tfirst\trr\tcrr\tp.1\tp.10\tp.100\tpN.1\tpN.10\tpN100\n")
        fout_items = open(join(self.out_dir, 'item_inter.tsv'), 'w+')
        fout_items.write("item.id\theldout.count\tintercept\tpopularity\tcrr\tave.rr\n")
        item_crr = defaultdict(float)
        item_count = defaultdict(float)

        fout_userD = open(join(self.out_dir, 'user_10047490.tsv'), 'w+')

        for user in users:
            num_heldout = self.data.heldout_count(user)
            if num_heldout == 0:
                continue
            U += 1

            predictions = {}
            for item in items:
                predictions[item] = self.pred(user, item)

            found = 0.0 # float so division plays nice
            rank = 0
            crr = 0
            first = 0
            rr = 0
            user_rmse = 0
            user_mae = 0

            recall = []
            p = defaultdict(float) # precision
            pN = defaultdict(float) # normalized precision
            for item in sorted(predictions, key=lambda x: -predictions[x]):
                rating = self.data.test_rating(user, item)
                rank += 1
                if user == 10047490:
                    fout_userD.write("%d\t%s %d\t%f\n" % (rank, "H" if rating != 0 else " ", item, predictions[item]))
                    self.printWhy(user,item, fout_userD)

                if rating != 0:
                    found += 1

                    prediction = predictions[item]

                    user_rmse += (rating - prediction)**2
                    user_mae += abs(rating - prediction)
                    crr += (1.0 / rank)
                    item_crr[item] += (1.0 / rank)
                    item_count[item] += 1
                    if found == 1:
                        first = rank
                        rr = (1.0 / rank)

                if rank == 1 or rank == 10 or rank == 100:
                    p[rank] = found / rank
                    pN[rank] = found / min(rank, num_heldout)
                    ave_p[rank] += p[rank]
                    ave_pN[rank] += pN[rank]

            rmse += user_rmse
            mae += user_mae
            N += found
            ave_crr += crr
            ave_rr += rr
            ave_rank += first

            # per user stats
            if found == 0:
                fout_user.write("%d\t-1\t-1\t-1\t0\t0\t0\t0\t0\t0\t0\t0\n" % user)
                continue
            fout_user.write("%d\t%f\t%f\t%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n" %
                (user, (user_rmse/found), (user_mae/found), first, rr, crr, \
                p[1], p[10], p[100], pN[1], pN[10], pN[100]))

        for item in item_crr:
            fout_items.write("%d\t%d\t%f\t%d\t%f\t%f\n" % \
            (item, item_count[item], self.intercept(item), data.item_counts[item], item_crr[item], item_crr[item] / item_count[item]))

        fout_items.close()
        fout_user.close()

        # overall stats
        fout_summary = open(join(self.out_dir, 'summary_eval.dat'), 'w+')
        fout_summary.write("rmse\t%f\n" % (rmse/N))
        fout_summary.write("mae\t%f\n" % (mae/N))
        fout_summary.write("rank\t%f\n" % (ave_rank/U))
        fout_summary.write("RR\t%f\n" % (ave_rr/U))
        fout_summary.write("CRR\t%f\n" % (ave_crr/U))
        fout_summary.write("p.1\t%f\n" % (ave_p[1]/U))
        fout_summary.write("p.10\t%f\n" % (ave_p[10]/U))
        fout_summary.write("p.100\t%f\n" % (ave_p[100]/U))
        fout_summary.write("pN.1\t%f\n" % (ave_pN[1]/U))
        fout_summary.write("pN.10\t%f\n" % (ave_pN[10]/U))
        fout_summary.write("pN.100\t%f\n" % (ave_pN[100]/U))


    def run(self):
        self.fit()
        self.eval()


class PopularityStudy(BaselineStudy):
    def pred(self, user, item, details=False):
        return self.data.item_counts[item]

    def intercept(self, item):
        return self.data.item_counts[item]

    def printWhy(self, user, item, fout):
        fout.write('\t%d\n' % self.data.item_counts[item])


class SimpleTrustStudy(BaselineStudy):
    def pred(self, user, item, details=False):
        score = 0
        for vser in data.friends[data.users[user]]:
            score += self.data.rating(vser, self.data.items[item])
        return score

    def intercept(self, item):
        return 0

    def printWhy(self, user, item, fout):
        fout.write('cause\n')


from scipy.stats import mode
def ave(L):
    return mode(L)[0][0]

class PFStudy(BaselineStudy):
    def fit(self):
        #True, \ INTERTAG
        model = model_settings(self.K, MF=True, trust=False, intercept=False, \
            SVI=self.SVI)
        priors = set_priors(model, self.data)
        params = init_params(model, priors, self.data)
        infer(model, priors, params, data, self.out_dir, self.lr)
        self.params = params

    def intercept(self, item):
        return self.params.inter[self.data.items[item]]

    def pred(self, user, item, details=False):
        prediction = self.params.inter[self.data.items[item]]
        prediction += sum(self.params.theta[self.data.users[user]] * \
            self.params.beta[self.data.items[item]])
        if details:
            print '\tintercept', self.params.inter[self.data.items[item]]
            mu = ave(self.params.theta[self.data.users[user]])
            k = 0
            for v in self.params.theta[self.data.users[user]]:
                if v > mu:
                    print '\ttheta',k,v
                k += 1

            mu = ave(self.params.beta[self.data.items[item]])
            k = 0
            for v in self.params.beta[self.data.items[item]]:
                if v > mu:
                    print '\tbeta',k,v
                k += 1

        return prediction

    def printWhy(self, user, item, fout):
        fout.write('\t%f\n' % self.params.inter[self.data.items[item]])
        fout.write('\t%f:' % (sum(self.params.theta[self.data.users[user]] * \
                    self.params.beta[self.data.items[item]])))
        for comp in self.params.theta[self.data.users[user]] * self.params.beta[self.data.items[item]]:
            fout.write(' %f' % comp)
        fout.write('\n\ttheta    ')
        for comp in self.params.theta[self.data.users[user]]:
            fout.write(' %f' % comp)
        fout.write('\n\tbeta     ')
        for comp in self.params.beta[self.data.items[item]]:
            fout.write(' %f' % comp)
        fout.write('\n')


class SPFStudy(BaselineStudy):
    def fit(self):
        #model = model_settings(self.K, MF=True, trust=True, intercept=True, \ INTERTAG
        model = model_settings(self.K, MF=True, trust=True, intercept=False, \
            SVI=self.SVI)
        priors = set_priors(model, self.data)
        params = init_params(model, priors, self.data)
        infer(model, priors, params, data, self.out_dir, self.lr)
        self.params = params
        self.model = model
        #TODO: duplicate with above, except trust arg!

    def intercept(self, item):
        return self.params.inter[self.data.items[item]]
    def pred(self, user, item, details=False):
        prediction = self.params.inter[self.data.items[item]]
        prediction += sum(self.params.theta[self.data.users[user]] * \
            self.params.beta[self.data.items[item]])
        if details:
            print '\tintercept', self.params.inter[self.data.items[item]]
            mu = ave(self.params.theta[self.data.users[user]])
            k = 0
            for v in self.params.theta[self.data.users[user]]:
                if v > mu:
                    print '\ttheta',k,v
                k += 1

            mu = ave(self.params.beta[self.data.items[item]])
            k = 0
            for v in self.params.beta[self.data.items[item]]:
                if v > mu:
                    print '\tbeta',k,v
                k += 1

        T = 0.0
        for vser in data.friends[data.users[user]]:
            rating = self.data.rating(vser, self.data.items[item])
            if rating != 0:
                T += self.params.tau.get(self.data.users[user], vser) * rating
                if details:
                    print '\ttau', vser, self.params.tau.get(self.data.users[user], \
                        vser), rating

        if not self.model.nofdiv and self.data.friend_counts[self.data.users[user]][self.data.items[item]] != 0:
            T /= self.data.friend_counts[self.data.users[user]][self.data.items[item]]

        prediction += T

        return prediction

    def printWhy(self, user, item, fout):
        fout.write('\t%f\n' % self.params.inter[self.data.items[item]])
        fout.write('\t%f:' % (sum(self.params.theta[self.data.users[user]] * \
                    self.params.beta[self.data.items[item]])))
        for comp in self.params.theta[self.data.users[user]] * self.params.beta[self.data.items[item]]:
            fout.write(' %f' % comp)
        fout.write('\n\ttheta    ')
        for comp in self.params.theta[self.data.users[user]]:
            fout.write(' %f' % comp)
        fout.write('\n\tbeta     ')
        for comp in self.params.beta[self.data.items[item]]:
            fout.write(' %f' % comp)
        fout.write('\n')

        T = 0.0
        t = []
        for vser in data.friends[data.users[user]]:
            rating = self.data.rating(vser, self.data.items[item])
            if rating != 0:
                T += self.params.tau.get(self.data.users[user], vser) * rating
                t.append((self.params.tau.get(self.data.users[user], vser), rating))
            else:
                t.append((0.0,0))

        if not self.model.nofdiv and self.data.friend_counts[self.data.users[user]][self.data.items[item]] != 0:
            d = self.data.friend_counts[self.data.users[user]][self.data.items[item]]

            fout.write('\t%f  (%f/%d):\t' % ((T/d), T, d))

        else:
            fout.write('\t%f:' % T)
        for trust,rating in t:
            fout.write(' %f|%d' % (trust,rating))

        fout.write('\n')


class TrustStudy(BaselineStudy):
    def fit(self):
        #model = model_settings(self.K, MF=False, trust=True, intercept=True, \ INTERTAG
        model = model_settings(self.K, MF=False, trust=True, intercept=False, \
            SVI=self.SVI)
        priors = set_priors(model, self.data)
        params = init_params(model, priors, self.data)
        infer(model, priors, params, data, self.out_dir, self.lr)
        self.params = params
        self.model = model
        #TODO: duplicate with above, except MF arg!

    def intercept(self, item):
        return self.params.inter[self.data.items[item]]

    def pred(self, user, item, details=False):
        prediction = self.params.inter[self.data.items[item]]

        T = 0.0
        for vser in data.friends[data.users[user]]:
            rating = self.data.rating(vser, self.data.items[item])
            if rating != 0:
                T += self.params.tau.get(self.data.users[user], vser) * rating
                if details:
                    print '\ttau', vser, self.params.tau.get(self.data.users[user], \
                        vser), rating

        if not self.model.nofdiv and self.data.friend_counts[self.data.users[user]][self.data.items[item]] != 0:
            T /= self.data.friend_counts[self.data.users[user]][self.data.items[item]]

        prediction += T

        return prediction

    def printWhy(self, user, item, fout):
        fout.write('\t%f\n' % self.params.inter[self.data.items[item]])

        T = 0.0
        t = []
        for vser in data.friends[data.users[user]]:
            rating = self.data.rating(vser, self.data.items[item])
            if rating != 0:
                T += self.params.tau.get(self.data.users[user], vser) * rating
                t.append(self.params.tau.get(self.data.users[user], vser) * rating)
            else:
                t.append(0.0)

        if not self.model.nofdiv and self.data.friend_counts[self.data.users[user]][self.data.items[item]] != 0:
            d = self.data.friend_counts[self.data.users[user]][self.data.items[item]]

            fout.write('\t%f  (%f/%f):\t' % ((T/d), T, d))
            for i in t:
                fout.write('  %f|%f' % (i,i/d))

        else:
            fout.write('\t%f:' % T)
            for i in t:
                fout.write(' %f' % i)

        fout.write('\n')



#class SoRecPFStudy(PFStudy):
#    def fit(self):
#        self.data = self






def parse_args():
    #TODO: organize this a little better
    parser = argparse.ArgumentParser(description='Infer parameters for PTF.')

    parser.add_argument('data', metavar='data', type=str, \
        help='Directory of data source.  See README for required format.')

    parser.add_argument('--out', dest='out_dir', type=str, default=getcwd(), \
        help='Study directory; all output goes here.')

    parser.add_argument('--K', dest='K', type=int, default=10, \
        help='Number of components for matrix factorization.')

    parser.add_argument('--seed', dest='seed', type=int, default=10, \
        help='random seed')

    parser.add_argument('--binary',dest='binary',action='store_true',default=False)

    parser.add_argument('--SVI', dest='svi', action='store_true', \
        default=False, help='Use stochatic VI instead of batch VI.')

    parser.add_argument('--directed', dest='directed', action='store_true',
        default=False, help='Network input is directed (default undirected)')

    parser.add_argument('--verbose', dest='verbose', action='store_true',
        default=False, help='Give more output!')

    return parser.parse_args()


if __name__ == '__main__':
    ### set up
    args = parse_args()

    #if args.verbose:
    #    global VERBOSE
    #    VERBOSE= args.verbose

    data = dataset({}, {}, args.binary, args.directed)
    data.read_ratings(args.data + "/train.tsv")
    data.read_network(args.data + "/network.tsv")
    data.read_validation(args.data + "/validation.tsv")
    data.read_test(args.data + "/test.tsv")

    print "Data Loaded"


    # create the fit dir if it doesn't exist
    args.out_dir = args.out_dir + ('/' if args.out_dir != '' else '')
    if not exists(args.out_dir):
        mkdir(args.out_dir)


    ### fit, predict, and evaluate for each model
    rand = BaselineStudy(data, join(args.out_dir, 'random'), args.seed)
    rand.start()

    pop = PopularityStudy(data, join(args.out_dir, 'popularity'), args.seed)
    pop.start()

    simple_trust = SimpleTrustStudy(data, join(args.out_dir, 'simple_trust'), args.seed)
    simple_trust.start()

    trust = TrustStudy(data, join(args.out_dir, 'trust'), args.seed, args.K, args.svi)
    trust.start()

    pf = PFStudy(data, join(args.out_dir, 'PF'), args.seed, args.K, args.svi)
    pf.start()

    spf = SPFStudy(data, join(args.out_dir, 'SPF'), args.seed, args.K, args.svi)
    spf.start()



    ### write out per-user network properties
    fout = open(join(args.out_dir, "user_stats.csv"), 'w+')
    fout.write("user.id,num.train,num.heldout,degree,interconnectivity\n")
    users, items = get_eval_sets(data)
    for user in users:
        num_heldout = data.heldout_count(user)
        if num_heldout != 0:
            num_train = len(data.user_data[user])
            degree = data.friend_count(user)
            interconnectivity = data.interconnectivity(user)
            fout.write("%d,%d,%d,%d,%d\n" % \
                (user, num_train, num_heldout, degree, interconnectivity))
    fout.close()

    ### write out general data stats
    fout = open(join(args.out_dir, "data_stats.dat"), 'w+')
    fout.write("num users:\t%d\n" % len(data.users))
    fout.write("num items:\t%d\n" % len(data.items))
    fout.write("num ratings:\t%d\t%d\t%d\n" % \
        (data.rating_count[0], data.rating_count[1], data.rating_count[2]))
    fout.write("network connections:\t%d\n" % data.connection_count)
    fout.close()


    ### wait for each one to finish
    rand.join()
    pop.join()
    trust.join()
    simple_trust.join()
    pf.join()
    spf.join()

    ### aggregate results
    print "aggregate results here"
