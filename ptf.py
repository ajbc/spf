# PTF inference
import argparse
import scipy
from scipy.misc import factorial
from scipy.special import digamma, gamma, gammaln, psi
import numpy as np
from numpy import log, exp
import random #TODO: just use one source for random, not both np and this
from collections import defaultdict
import time
import ptfstore
from time import clock


def save_state(dire, iteration, model, params):
    #print "Saving state!"
    fname = "%s-iter%d.out" % (dire + '/model', iteration)
    ptfstore.dump(fname, model, params)

def get_predictions(model, params, data, test_set):
    preds = np.zeros(len(test_set.users))

    if model.intercept:
        preds += params.inter[test_set.items]
    elif model.trust:
        preds = np.ones(len(test_set.users))*1e-10


    if model.MF:
        M = (params.theta[test_set.users] * params.beta[test_set.items]).sum(axis=1)

    if model.MF:
        if model.eta:
            preds += M * params.eta
        else:
            preds += M

    if model.trust:
        T = np.zeros(len(test_set.users))
        for i in xrange(len(test_set.users)):
            user = test_set.users[i]
            item = test_set.items[i]
            for vser in data.friends[user]:
                if data.binary:
                    if item in data.user_data[vser]:
                        T[i] += params.tau.get(user, vser)
                else:
                    rating_v = data.sparse_ratings.get(item, vser)
                    if rating_v != 0:
                        T[i] += params.tau.get(user, vser) * rating_v

            if data.friend_counts[user][item] != 0 and not model.nofdiv:
                T[i] /= data.friend_counts[user][item] #NOFDIV

    if model.trust:
        if model.eta:
            preds += T * (1-params.eta)
        else:
            preds += T

    return preds

def approx_log_likelihood(model, params, data, priors):
    return
    # mirroring prem's code...
    sf = 0
    for user, item, rating in data.train_triplets:
        phi_M = params.theta[data.users[user]] * params.beta[data.items[item]]
        phi = phi_M / sum(phi_M)
        phi *= rating

        #s = sum(phi * (np.log(phi_M) - np.log(phi)))
        s = np.log(sum(phi_M)) * rating

        #s -= np.log(factorial(rating))

        s -= sum(phi_M)

        #print user, item, s, sum(phi_M)
        sf += s

    '''# theta
    elbo_theta = (priors['b_theta'] - params.b_theta) * np.array(params.theta)
    elbo_theta += (- priors['a_theta'] + params.a_theta) * np.array(np.log(params.theta))
    elbo_theta += - priors['a_theta']*np.log(priors['b_theta']) + \
        params.a_theta * np.log(params.b_theta)
    elbo_theta += - gammaln(params.a_theta) + gammaln(priors['a_theta'])

    # beta
    elbo_beta = (priors['b_beta'] - params.b_beta) * np.array(params.beta)
    elbo_beta += (- priors['a_beta'] + params.a_beta) * np.array(np.log(params.beta))
    elbo_beta += - priors['a_beta']*np.log(priors['b_beta']) + \
        params.a_beta * np.log(params.b_beta)
    elbo_beta += - gammaln(params.a_beta) + gammaln(priors['a_beta']) '''
    # theta
    elbo_theta = - priors['b_theta'] * np.array(params.theta)
    #print elbo_theta
    elbo_theta += priors['a_theta']*np.log(priors['b_theta'])
    #print elbo_theta
    elbo_theta -= gammaln(priors['a_theta'])
    #print elbo_theta
    elbo_theta += (priors['a_theta'] - 1) * np.array(np.log(params.theta))
    #print elbo_theta
    #print (priors['a_theta'] - 1)
    #print params.theta[0,0]
    #print np.log(params.theta)[0,0]
    #print np.array(np.log(params.theta))[0,0]

    # beta
    elbo_beta = - priors['b_beta'] * np.array(params.beta)
    elbo_beta += priors['a_beta'] * np.log(priors['b_beta'])
    elbo_beta -= gammaln(priors['a_beta'])
    elbo_beta += (priors['a_beta'] - 1) * np.array(np.log(params.beta))

    print sf + elbo_theta.sum() + elbo_beta.sum()



def get_log_likelihoods(model, params, data, test_set):
    predictions = get_predictions(model, params, data, test_set)
    likelihoods = np.log(predictions) * test_set.ratings - \
        np.log(factorial(test_set.ratings)) - predictions
    #likelihoods[np.isinf(likelihoods)] = 0
    #print "LIKELIHOODS"
    #print predictions[:10]
    #print test_set.ratings[:10]
    #print "LIKIHOODS END"
    return likelihoods


def get_elbo(model, priors, params, data):
    #print "getting log likelihoods for ratings"
    rating_likelihoods = get_log_likelihoods(model, params, data, data.validation)

    elbo = rating_likelihoods.sum() / len(rating_likelihoods)
    return elbo
    # theta
    elbo_theta = (-1*priors['b_theta'] + params.b_theta) * np.array(params.theta)
    elbo_theta += (priors['a_theta'] - params.a_theta) * np.array(np.log(params.theta))
    elbo_theta += priors['a_theta']*np.log(priors['b_theta']) - \
        params.a_theta * np.log(params.b_theta)
    elbo_theta += gammaln(params.a_theta) - gammaln(priors['a_theta'])

    # beta
    elbo_beta = (-1*priors['b_beta'] + params.b_beta) * np.array(params.beta)
    elbo_beta += (priors['a_beta'] - params.a_beta) * np.array(np.log(params.beta))
    elbo_beta += priors['a_beta']*np.log(priors['b_beta']) - \
        params.a_beta * np.log(params.b_beta)
    elbo_beta += gammaln(params.a_beta) - gammaln(priors['a_beta'])

    # tau
    if model.trust:
        #print "elbo_tau progression..."
        elbo_tau = (params.tau.multiply((priors['b_tau'].const_multiply(-1) + params.b_tau))).sum()
        #print "   ",elbo_tau
        elbo_tau += (((priors['a_tau'] - params.a_tau)).multiply(params.tau.log())).sum()
        #print "   ",elbo_tau
        elbo_tau += (priors['a_tau'].multiply(priors['b_tau'].log())).sum()
        #print "   ",elbo_tau
        elbo_tau -= (params.a_tau.multiply(params.b_tau.log())).sum()
        #print "   ",elbo_tau
        elbo_tau += params.a_tau.log_gamma_sum()
        #print "   ",elbo_tau
        elbo_tau -= priors['a_tau'].log_gamma_sum()
        #print "   ",elbo_tau
    else:
        elbo_tau = 0

    # intercept; b never varies from prior
    if model.intercept:
        elbo_inter = (priors['a_inter'] - params.a_inter) * np.log(params.inter)
        elbo_inter += priors['a_inter']*np.log(params.b_inter) - params.a_inter * np.log(params.b_inter)
        elbo_inter += gammaln(params.a_inter) - gammaln(priors['a_inter'])
    else:
        elbo_inter = np.zeros(3)


    elbo_ratings = elbo
    elbo += elbo_theta.sum()
    elbo += elbo_beta.sum()
    elbo += elbo_tau
    elbo += elbo_inter.sum()
    #print "ELBO: ratings: %f\ttheta: %f\tbeta: %f\ttau: %f\teta: %f\tinter: %f\t=> total: %f" % \
    #    (elbo_ratings, elbo_theta.sum(), elbo_beta.sum(), elbo_tau, elbo_eta, elbo_inter.sum(), elbo)
    return elbo


def get_ave_likelihood(model, params, data, test_set):
    return get_log_likelihoods(model, params, data, test_set).sum() \
        / test_set.size

class model_settings:
    def __init__(self, K, MF, trust, intercept, sorec=False, SVI=False, eta=False):
        self.K = K
        self.MF = MF
        self.trust = trust
        self.intercept = intercept
        self.sorec = sorec
        #self.nofdiv = False
        self.nofdiv = True #new!
        self.SVI = SVI
        self.eta = eta

    #@classmethod
    #def new(self, user_count, item_count, K, MF, trust, iat, intercept, users, items, undirected):
    #    return model_settings(user_count, item_count, args.K, MF, \
    #        trust, iat, intercept, users, items, undirected)

    @classmethod
    def fromargs(self, args):
        MF = False if args.model == "trust" or args.model == "IATonly" \
            else True
        trust = False if args.model == 'PF' or args.model == "IATonly" \
            or args.model == "IAT+MF" else True
        #sorec = True if args.sorec else False

        return model_settings(args.K, MF, trust, args.intercept)

class parameters:
    def __init__(self, model, readonly, data, priors):
        self.readonly = readonly
        print "  in parameters init (readonly=%s)" % str(readonly)

        print "   initializing model parameters"
        self.tau = dict_matrix(float, data.user_count, data.user_count)
        self.logtau = dict_matrix(float, data.user_count, data.user_count)
        self.inter = np.zeros(data.item_count)
        self.theta = np.zeros((data.user_count, model.K))
        self.logtheta = np.zeros((data.user_count, model.K))
        self.beta = np.zeros((data.item_count, model.K))
        self.logbeta = np.zeros((data.item_count, model.K))
        self.eta = 2.0 / 7.0
        self.logeta = digamma(2.0) - digamma(2.0 + 7.0)
        self.logetai = digamma(7.0) - digamma(2.0 + 7.0)

        if not readonly:
            self.a_eta = 2.0
            self.b_eta = 5.0
            print "    initializing intermediate variables"
            self.a_theta = np.ones((data.user_count, model.K))
            self.a_beta = np.ones((data.item_count, model.K))
            self.tau = np.zeros(3) #TODO: fix; this is heer for the logfile
            if model.trust:
                self.b_theta = np.ones((data.user_count, model.K))
                self.b_beta = np.ones((data.item_count, model.K))
                #self.tau = data.friend_matrix.copy(mult=10.0)
                #M = 100.0
                self.tau = data.friend_matrix.copy()#mult=M)
                self.logtau = data.friend_matrix.copy()#mult=np.log(M))
            else:
                self.b_theta = np.ones((data.user_count, model.K)) #TODO: do this outside if
                self.b_beta = np.ones((data.item_count, model.K))

            # per item intercepts
            self.a_inter = np.ones(data.item_count)
            self.b_inter = np.ones(data.item_count)*data.user_count# * 1e-9

            if model.intercept:
                self.inter = np.ones(data.item_count) * 0.1
                #self.b_inter = np.zeros(data.item_count)
                #for item in data.items:
                #    self.b_inter[data.items[item]] = data.item_counts[item]
            else:
                self.inter = np.zeros(data.item_count)

    def set_to_priors(self, priors):
        self.a_theta.fill(priors['a_theta'])
        self.b_theta.fill(priors['b_theta'])
        self.a_beta_prev = self.a_beta.copy()
        self.a_beta.fill(priors['a_beta'])
        self.b_beta.fill(priors['b_beta'])
        self.a_tau = priors['a_tau'].copy()
        self.b_tau = priors['b_tau'].copy()
        self.a_inter.fill(priors['a_inter'])
        self.a_eta = 2.0
        self.b_eta = 5.0


    def update_eta(self):
        self.eta = self.a_eta / (self.a_eta + self.b_eta)
        self.logeta = digamma(self.a_eta) - digamma(self.a_eta + self.b_eta)
        self.logetai = digamma(self.b_eta) - digamma(self.a_eta + self.b_eta)

    def update_shape(self, user, item, rating, model, data, MF_converged=True, \
        user_scale=1.0):
        log_phi_M = self.logtheta[user] + self.logbeta[item]
        if model.eta:
            log_phi_M += self.logeta
        phi_M = exp(log_phi_M)

        start = clock()
        log_phi_T = dict_row(float, 0)
        if model.trust and MF_converged:
            if not data.binary:
                for friend in self.logtau.rows[user]:
                    log_phi_T.cols[friend] = self.logtau.rows[user][friend] + \
                    data.log_sparse_ratings.rows[item][friend]
                    if model.eta:
                        log_phi_T.cols[friend] += self.logetai
                    #NOFDIV
            else:
                for friend in self.logtau.rows[user]:
                    if item in data.user_data[friend]:
                        log_phi_T.cols[friend] = self.logtau.rows[user][friend]
                        if model.eta:
                            log_phi_T.cols[friend] += self.logetai
        div = 0 if model.nofdiv or data.friend_counts[user][item] == 0 else \
            max(0, log(data.friend_counts[user][item]))
        if div != 0:
            log_phi_T.sub_const(div)
        phi_T = exp(log_phi_T)

        # this section is the problematic one

        phi_sum = self.inter[item]
        if model.MF:
            phi_sum += phi_M.sum()
        if model.trust and MF_converged:
            phi_sum += phi_T.sum()
        if phi_sum == 0:
            return
        mult = rating / phi_sum
        logmult = log(mult)
        log_user_scale = log(user_scale)

        if model.intercept:
            self.a_inter[item] += self.inter[item] * mult * user_scale # for binary data, this fixes intercept max at 1

        if model.MF:
            self.a_theta[user] += exp(log_phi_M + logmult)
            if model.SVI:
                self.a_beta[item] += exp(log_phi_M + logmult + log_user_scale)
            else:
                self.a_beta[item] += exp(log_phi_M + logmult)

            self.a_eta += sum(exp(log_phi_M + logmult))
            #self.a_beta[item] += exp(log_phi_M + logmult + log_user_scale)
        if model.trust and MF_converged:
            if phi_T.sum() != 0:
                log_phi_T.add_const(logmult)
                self.a_tau.row_add(user, log_phi_T.exp())
                self.b_eta += log_phi_T.exp().sum()

    def update_MF(self, model, data, user_scale=1.0, \
                users_updated=False, \
                items_updated=False, items_seen_counts=False, tau0=1, kappa=1):
        if not model.MF:
            return

        #TODO: sets of users/items updated are always passed in
        # they don't need defaults
        if users_updated == False:
            users_updated = set(data.users.values())
            items_updated = set(data.items.values())

        usrs = list(users_updated)
        if model.eta:
            self.b_theta += self.beta.sum(axis=0) * self.eta
        else:
            self.b_theta += self.beta.sum(axis=0)

        for user in users_updated:
            self.theta[user] = self.a_theta[user] / self.b_theta
            self.logtheta[user] = psi(self.a_theta[user]) - \
                log(self.b_theta)

        if model.eta:
            self.b_beta += self.theta.sum(axis=0) * self.eta
        else:
            self.b_beta += self.theta.sum(axis=0)

        for item in items_updated:
            if model.SVI:
                rho = (items_seen_counts[item] + tau0) ** (-kappa)
                self.a_beta[item] = (1 - rho) * self.a_beta_prev[item] + \
                    rho * self.a_beta[item]
            self.beta[item] = (self.a_beta[item] / self.b_beta)
            self.logbeta[item] = psi(self.a_beta[item]) - log(self.b_beta)

    def update_TF(self, model, data, user_scale=1.0, \
                users_updated=False, iteration=1, tau0=1, kappa=1):
        if users_updated == False:
            users_updated = set(data.users.values())

        if model.trust:
            if model.eta:
                b_tau = self.b_tau.multadd(1- self.eta, 0.3)
            else:
                b_tau = self.b_tau
            self.tau = self.a_tau / b_tau

            #print "a tau 35"
            #print self.a_tau.rows[data.users[35]]
            #print "b tau 35"
            #print self.b_tau.rows[data.users[35]]
            self.logtau = self.a_tau.psi() - log(b_tau)


class triplets:
    def __init__(self):
        self.users = []
        self.items = []
        self.ratings = []
        self.friend_counts = []

class dict_row():
    def __init__(self, val_type, ncols=False):
        self.ncols = ncols

        self.val_type = val_type

        self.cols = defaultdict(val_type)

    def __str__(self):
        i = 0
        rv = '['
        for row in self.rows:
            if i == 4:
                rv += ' ...'
                break
            rv += ' ' + str(row) +':' + str(self.rows[row])
            i += 1
        return rv

    def set(self, col, val):
        if val == 0 and col in self.cols:
            del self.cols[col]
        elif val != 0:
            self.cols[col] = val

    def add_const(self, val):
        for col in self.cols:
            self.cols[col] += val

    def sub_const(self, val):
        for col in self.cols:
            self.cols[col] -= val

    def exp(self):
        for col in self.cols:
            self.cols[col] = exp(self.cols[col])

    def item_add(self, col, val):
        if val != 0: #not (type(val) == int or type(val) == float) or val != 0:
            self.cols[col] += val

    def sum(self):
        rv = 0
        for col in self.cols:
            rv += self.cols[col]
        return rv

    def densedense(self):
        return self.cols.values()

    def pre_multiply(self, other):
        #TODO: check for shape/size; this presumes both are rows
        #TODO: scalar multiplication
        # make a matrix
        rv = dict_matrix(self.val_type, other.shape[0])
        for col in self.cols:
            for row in xrange(other.shape[0]):
                #print (row, col, self.cols[col], other[row], self.cols[col] * other[row])
                rv.set(row, col, self.cols[col] * other[row])
        return rv

    def const_multiply(self, const):
        rv = dict_row(self.val_type, self.ncols)
        for col in self.cols:
            rv.set(col, self.cols[col] * const)
        return rv

    def exp(self):
        rv = dict_row(self.val_type, self.ncols)
        for col in self.cols:
            rv.set(col, exp(self.cols[col]))
        return rv

    def todense(self):
        rv = np.zeros(self.ncols)
        for col in self.cols:
            rv[col] = self.cols[col]
        return rv

    def get_sums(self, mult):
        phi_MT_sum = 0.0
        phi_MT1 = np.zeros(len(mult))
        phi_MT0 = np.zeros(len(self.cols))

        i = 0
        for v in self.cols.values():
            c = mult * v
            cs = sum(c)
            phi_MT += cs
            phi_MT0[i] += cs
            phi_MT1 += c
            i += 1

        return phi_MT_sum, phi_MT1, phi_MT0



class dict_matrix():
    def __init__(self, val_type, nrows=False, ncols=False):

        self.val_type = val_type

        if val_type==float:
            self.rows = defaultdict(ddf)
            #self.cols = defaultdict(ddf)
        elif val_type==int:
            self.rows = defaultdict(ddi)
            #self.cols = defaultdict(ddi)
        elif val_type==bool:
            self.rows = defaultdict(ddb)
            #self.cols = defaultdict(ddb)
        else:
            print "UHOH"

    def __str__(self):
        i = 0
        rv = ''
        for row in self.rows:
            if i > 9:
                rv += '...'
                break
            rv += str(row) + ': ['
            j = 0
            for col in self.rows[row]:
                if j > 3:
                    rv += ' ...'
                    break
                rv += ' ' + str(col) +':' + str(self.rows[row][col])
                j += 1
            rv += ' ]\n'
            i += 1
        return rv

    def set(self, row, col, val):
        if val == 0 and row in self.rows and col in self.rows[row]:
            del self.rows[row][col]
            #del self.cols[col][row]
        elif val != 0:
            self.rows[row][col] = val
            #self.cols[col][row] = val

    def get(self, row, col):
        return self.rows[row][col]

    def item_add(self, row, col, val):
        if val != 0:
            #print "hi?", self.rows[row][col], val
            self.rows[row][col] += val
            #self.cols[col][row] += val
            #print "hi!", self.rows[row][col], val

    def item_sub(self, row, col, val):
        if val != 0:
            self.rows[row][col] -= val
            #self.cols[col][row] -= val

    def item_div(self, row, col, val):
        if val != 0:
            self.rows[row][col] /= val

    def copy(self, mult=1.0):
        rv = dict_matrix(self.val_type)
        for row in self.rows:
            for col in self.rows[row]:
                rv.set(row, col, self.rows[row][col] * mult)
        return rv

    def psi(self):
        rv = dict_matrix(self.val_type)
        for row in self.rows:
            for col in self.rows[row]:
                val = psi(self.rows[row][col])
                rv.set(row, col, val)
        return rv


    def __add__(self, other):#, val_type=self.val_type):
        #if self.nrows != other.nrows or self.ncols != other.ncols:
        #    raise ValueError("Not the same shape! [self %dx%d; other %dx%d" % \
        #        (self.nrows, self.ncols, other.nrows, other.ncols))

        rv = dict_matrix(self.val_type)#, self.nrows, self.ncols)
        for row in self.rows:
            for col in self.rows[row]:
                rv.set(row, col, self.rows[row][col])
        for row in other.rows:
            for col in other.rows[row]:
                rv.item_add(row, col, other.rows[row][col])
        return rv

    def __sub__(self, other):#, val_type=self.val_type):
        #if self.nrows != other.nrows or self.ncols != other.ncols:
        #    raise ValueError("Not the same shape! [self %dx%d; other %dx%d" % \
        #        (self.nrows, self.ncols, other.nrows, other.ncols))

        rv = dict_matrix(self.val_type)#, self.nrows, self.ncols)
        for row in self.rows:
            for col in self.rows[row]:
                rv.set(row, col, self.rows[row][col])
        for row in other.rows:
            for col in other.rows[row]:
                rv.item_sub(row, col, other.rows[row][col])
        return rv

    def __iadd__(self, other):
        #if self.nrows != other.nrows or self.ncols != other.ncols:
        #    raise ValueError("Not the same shape! [self %dx%d; other %dx%d" % \
        #        (self.nrows, self.ncols, other.nrows, other.ncols))

        for row in other.rows:
            for col in other.rows[row]:
                if other.rows[row][col] != 0:
                    self.item_add(row, col, other.rows[row][col])
        return self

    def row_add(self, row_id, other):
        for col in self.rows[row_id]:
            self.rows[row_id][col] += other.cols[col]
            #self[row_id][col] += other[col] #TODO: do this form

    '''def row_add(self, row_id, other, indexes=None, mult=1):
        #for col_id in other.rows[row_id]:
        #    self.item_add(row_id, col_id, other.rows[row_id][col_id])
        if indexes:
            for c in xrange(len(indexes)):
                self.item_add(row_id, indexes[c], other[c] * mult)
        else:
            for col_id in other.cols:
                self.item_add(row_id, col_id, other.cols[col_id] * mult)
    '''

    def multiply(self, other):
        rv = dict_matrix(self.val_type)#, self.nrows, self.ncols)
        for row in self.rows:
            for col in set(self.rows[row]) & set(other.rows[row]):
                rv.set(row, col, self.rows[row][col] * other.rows[row][col])
        return rv

    #def __mult__(self, const):
    def const_multiply(self, const):
        rv = dict_matrix(self.val_type)#, self.nrows, self.ncols)
        for row in self.rows:
            for col in self.rows[row]:
                rv.set(row, col, self.rows[row][col] * const)
        return rv

    def multadd(self, constA, constB):
        rv = dict_matrix(self.val_type)#, self.nrows, self.ncols)
        print 'CONST A AND B', constA, constB
        for row in self.rows:
            for col in self.rows[row]:
                rv.set(row, col, self.rows[row][col] * constA + constB)
        return rv

    def double_multiply(self, in_row, in_col):
        rv = dict_matrix(val_type)#, self.nrows, self.ncols)
        for row in self.rows:
            for col in self.rows[row]:
                rv.set(row, col, self.rows[row][col] * \
                    in_row[col] * in_col[row]) #TODO:chaneg row to row_id and in_col to col
        return rv

    def multiply_row(self, self_row_id, other, other_row_id, const_div=1, printer=False):
        rv = dict_row(self.val_type)
        for col in self.rows[self_row_id]:
            if printer:
                print "multiply row:", col, self.rows[self_row_id][col], other.rows[other_row_id][col], const_div
            rv.set(col, self.rows[self_row_id][col] * \
                other.rows[other_row_id][col] / const_div)
        return rv

    def multiply_row_select(self, self_row_id, other, other_row_id, const_div=1):
        rv = dict_row(self.val_type)
        for col in self.rows[self_row_id]:
            if other_row_id in other[col]:
                rv.set(col, self.rows[self_row_id][col] / const_div)
        return rv

    def __div__(self, other):
        rv = dict_matrix(self.val_type)#, self.nrows, self.ncols)
        for row in self.rows:
            for col in set(self.rows[row]) & set(other.rows[row]):
                rv.set(row, col, self.rows[row][col] / other.rows[row][col])
        return rv

    def sums(self):
        rvA = dict_row(self.val_type)#, self.ncols)
        rvB = dict_row(self.val_type)#, self.nrows)
        for row in self.rows:
            for col in self.rows[row]:
                rvA.item_add(col, self.rows[row][col])
                rvB.item_add(row, self.rows[row][col])
        return rvA, rvB

    def sum(self, axis=None):
        if axis == 0:
            rv = dict_row(self.val_type)
            for row in self.rows:
                for col in self.rows[row]:
                    rv.item_add(col, self.rows[row][col])
            return rv

        if axis == 1:
            rv = dict_row(self.val_type)#, self.nrows)
            for row in self.rows:
                for col in self.rows[row]:
                    rv.item_add(row, self.rows[row][col])
            return rv

        if not axis:
            rv = 0
            for row in self.rows:
                for col in self.rows[row]:
                    rv += self.rows[row][col]
            return rv

    def get_ave(self):
        rv = 0
        rc = 0
        for row in self.rows:
            for col in self.rows[row]:
                rv += self.rows[row][col]
                rc += 1
        return rv / rc


    def log(self):
        rv = dict_matrix(float)#, self.nrows, self.ncols)
        for row in self.rows:
            for col in self.rows[row]:
                rv.set(row, col, np.log(self.rows[row][col]))
        return rv

    def mean(self):
        rv = 0.
        counter = 0
        for row in self.rows:
            for col in self.rows[row]:
                rv += self.rows[row][col]
                counter += 1
        return rv / counter

    def log_gamma_sum(self):
        rv = 0
        for row in self.rows:
            for col in self.rows[row]:
                rv += gammaln(self.rows[row][col])
        return rv

    def set_row(self, row_id, row):
        for col in self.rows[row_id]:
            self.set(row_id, col, row[col])

def ddi():
    return defaultdict(int)
def ddf():
    return defaultdict(float)
def ddb():
    return defaultdict(bool)
class dataset:
    def __init__(self, users, items, binary, directed):
        self.users = users # hashes for users and items
        self.items = items

        self.directed = directed
        self.binary = binary

        self.train_triplets = []

        self.training = triplets()
        self.validation = triplets()

        self.item_counts = defaultdict(int)
        self.item_count = 0
        self.user_count = 0

        if not binary:
            self.sparse_ratings = dict_matrix(int)
            self.log_sparse_ratings = dict_matrix(float)
            self.sparse_vratings = dict_matrix(int)

        self.friends = defaultdict(list)
        self.friend_counts = defaultdict(ddi)
        self.friend_matrix = dict_matrix(bool)

        self.user_data = defaultdict(list)
        self.item_data = defaultdict(list)

        self.connection_count = 0
        self.rating_count = [0,0,0]

    def sorec_copy(self):
        dataset = dataset(self.users.copy(), self.items.copy(), self.binary, \
            self.directed)

        print "TODO"
        return dataset

    def has_rating(self, user, item):
        #binary only
        if not self.binary:
            print 'Problem!!!'
        return (item in self.user_data[user] or item in self.user_datav[user])

    def shares(self, user, item):
        f = set()
        # for binary only
        if self.binary:
            for friend in self.friends[user]:
                if item in self.user_data[friend]:
                    f.add(friend)
        else:
            for friend in self.friends[user]:
                if item in set([i for i,r in self.user_data[friend]]):
                    f.add(friend)
        return f

    def read_ratings(self, filename):
        i = 0
        user_count = len(self.users)
        item_count = len(self.items)
        self.training.size = 0
        '''counts = defaultdict(int)
        for line in open(filename, 'r'):
            triplet = tuple([int(x.strip()) for x in line.split('\t')])
            user, item, rating = triplet
            counts[item] += 1'''

        for line in open(filename, 'r'):
            if i % 500000 == 0:
                print i
            i += 1
            triplet = tuple([int(x.strip()) for x in line.split('\t')])
            user, item, rating = triplet

            if rating == 0:# or counts[item] < 2:
                continue
            self.rating_count[0] += 1
            if self.binary:
                #self.train_triplets.append((self.users[user], model.items[item]))
                self.train_triplets.append((user,item))
            else:
                self.train_triplets.append(triplet)
                #self.train_triplets.append( \
                #    (self.users[user], model.items[item], rating))

            if user not in self.users:
                self.users[user] = user_count#len(self.users)
                user_count += 1

            if item not in self.items:
                self.items[item] = item_count#len(self.items)
                item_count += 1

            self.item_counts[item] += 1
            if not self.binary:
                self.sparse_ratings.set(self.items[item], self.users[user], rating)
                self.log_sparse_ratings.set(self.items[item], self.users[user], log(rating))

                self.user_data[self.users[user]].append((self.items[item], rating))
                self.item_data[self.items[item]].append((self.users[user], rating))
            else:
                self.user_data[self.users[user]].append(self.items[item])
                #self.item_data[self.items[item]].append(self.users[user])

        self.user_count = user_count
        self.item_count = item_count


    def read_validation(self, filename, user_set=False, item_set=False):
        users = []
        items = []
        ratings = []
        friends = []
        self.user_datav = defaultdict(list)
        for line in open(filename, 'r'):
            user, item, rating = \
                tuple([int(x.strip()) for x in line.split('\t')])
            if user not in self.users or item not in self.items:
                continue
            if (user_set and user not in user_set) or (item_set and item not in item_set):
                continue
            self.rating_count[1] += 1
            users.append(self.users[user])
            items.append(self.items[item])
            if self.binary:
                ratings.append(1)
            else:
                ratings.append(rating)
            friends.append(self.friend_counts[self.users[user]][self.items[item]])

            if not self.binary:
                self.user_datav[self.users[user]].append((self.items[item], rating))
                self.sparse_vratings.set(self.items[item], self.users[user], rating)
            else:
                self.user_datav[self.users[user]].append(self.items[item])

        self.validation.users = users
        self.validation.items = items
        self.validation.ratings = ratings
        self.validation.friend_counts = friends
        self.validation.size = len(users)


    def read_network(self, filename):
        print "reading network...."

        for line in open(filename, 'r'):
            if ',' in line:
                user, friend = tuple([int(x.strip()) for x in line.split(',')])
            else:
                user, friend = tuple([int(x.strip()) for x in line.split('\t')])
            if user not in self.users or friend not in self.users:
                continue
            f = "f"+str(friend)
            u = "f"+str(friend)
            duple = (user, f)
            duple2 = (friend, u)
            triplet = (user, f, 1)
            triplet2 = (friend, u, 1)
            user = self.users[user]
            friend = self.users[friend]

            '''
            if model.sorec:
                if f not in self.items:
                    self.items[f] = self.item_count
                    self.item_count += 1
                item = self.items[f]
                if not self.directed:
                    if u not in self.items:
                        self.items[u] = self.item_count
                        self.item_count += 1
                    item2 = self.items[u]

                if self.binary:
                    self.train_triplets.append(duple)
                    if not self.directed:
                        self.train_triplets.append(duple2)
                else:
                    self.train_triplets.append(triplet)
                    if not self.directed:
                        self.train_triplets.append(triplet2)

                self.item_counts[item] += 1
                if not self.directed:
                    self.item_counts[item2] += 1
                if not self.binary:
                    self.sparse_ratings.set(item, user, 1)

                    self.user_data[user].append((item, 1))
                    self.item_data[item].append((user, 1))
                    if not self.directed:
                        self.sparse_ratings.set(item2, friend, 1)

                        self.user_data[friend].append((item2, 1))
                        self.item_data[item2].append((friend, 1))
                else:
                    self.user_data[user].append(item)
                    if not self.directed:
                        self.user_data[friend].append(item2)
            '''
            if friend not in self.friends[user] and user != friend:
                self.friends[user].append(friend)
                self.friend_matrix.set(user, friend, True)

                self.connection_count += 1

                if not self.directed:
                    self.friends[friend].append(user)
                    self.friend_matrix.set(friend, user, True)

                if self.binary:
                    for item in self.user_data[friend]:
                        self.friend_counts[user][item] += 1
                else:
                    for item, rating in self.user_data[friend]:
                        self.friend_counts[user][item] += 1
                if not self.directed:
                    if self.binary:
                        for item in self.user_data[user]:
                            self.friend_counts[friend][item] += 1
                    else:
                        for item, rating in self.user_data[user]:
                            self.friend_counts[friend][item] += 1

        print "  network done"

    def read_test(self, filename):
        if self.binary:
            self.test_user_data = defaultdict(set)
        else:
            self.test_user_data = defaultdict(dict)

        print "READING TEST"
        for line in open(filename, 'r'):
            user, item, rating = \
                 tuple([int(x.strip()) for x in line.split('\t')])

            if user == 838:
                print "HOOOOLLLA"
            if user not in self.users or item not in self.items:
                continue
            if user == 838:
                print "PASSED"

            self.rating_count[2] += 1

            if self.items[item] not in self.user_data[self.users[user]] and self.items[item] not in self.user_datav[self.users[user]]:
                if user == 838:
                    print "MADE IT", user, item, rating
                if self.binary:
                    self.test_user_data[user].add(item)
                else:
                    self.test_user_data[user][item] = rating

    def heldout_count(self, user):
        return len(self.test_user_data[user])

    def friend_count(self, user):
        return len(self.friends[self.users[user]])

    def interconnectivity(self, user):
        friends = set(self.friends[self.users[user]])
        count = len(friends)
        for friend in friends:
            f = set(self.friends[friend])
            count += len(f & friends)
        return count

    # TODO: these functions should be more organiszed (group by test/train)
    def rating(self, user, item):
        if self.binary:
            return 1 if item in self.user_data[user] else 0
        return self.sparse_ratings.get(item, user)


    def test_rating(self, user, item):
        if self.binary:
            return int(item in self.test_user_data[user])
        else:
            if item in self.test_user_data[user]:
                return self.test_user_data[user][item]
            else:
                return 0

    def items_in_common(self, user, friend):
        return len(set(self.user_data[user]) & set(self.user_data[friend]))
