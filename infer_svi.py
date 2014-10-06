# Varaiational Inference for Social Poisson Factorization
import argparse
import sys
import scipy
from scipy.misc import factorial
from scipy.special import digamma, gamma, gammaln
import numpy as np
import random #TODO: just use one source for random, not both np and this
from collections import defaultdict
import time
from time import clock

import os
from os.path import isfile, join, exists

import pygsl.sf

import ptfstore
from ptf import *

tau0 = 1024
kappa = 0.7
user_sample_count = 500
user_scale = 1

def save_state(dire, iteration, model, params):
    # find old state file
    oldstate = ''
    # TODO: make this work for local dump (i.e. dire == '')
    for f in [ f for f in os.listdir(dire) if isfile(join(dire,f)) ]:
        if f.startswith('params-'):
            oldstate = f
            break

    #print "Saving state!"
    fname = "%s-iter%d.dat" % (dire + '/params', iteration)
    ptfstore.dump(fname, model, params)

    # get rid of old state after
    if oldstate != '':
        os.remove(join(dire, oldstate))


def log_state(f, iteration, params, likelihood):
    f.write("%d\t%e\t%e\t%e\t%e\t%e\t%e\n" % (iteration, likelihood, \
        params.theta.mean(), params.beta.mean(), params.tau.mean(),
        params.eta, params.inter.mean()))


# infer latent variables
def infer(model, priors, params, data, dire='', rnd=False):
    if not rnd:
        rnd = random.Random()
    old_C = 1.0
    delta_C = 1e12
    delta_C_thresh = 1e-5 #TODO: find good default value and make it a command arg
    if data.binary:
        delta_C_thesh = 1e-10
    if model.SVI:
        delta_C = [delta_C] * 10
        delta_C_thresh /= 100

    logf = open(join(dire, 'log.tsv'), 'w+')
    logf.write("iteration\tC\ttheta\tbeta\ttau\teta\tintercept\n") # ave values

    iteration = 0

    global user_scale

    items_seen_counts = defaultdict(int)
    batch_size = min(100000, len(data.train_triplets)) #0.1M

    MF_converged = not model.MF
    while (not model.SVI and delta_C > delta_C_thresh) or \
        (model.SVI and sum(delta_C)/10 > delta_C_thresh): # not converged
        if (delta_C[0] if model.SVI else delta_C) < \
            10**(log(delta_C_thresh)/log(10)/2) and not MF_converged:
            MF_converged = True
            print "MF converged"

        #user_sample_count = len(model.users)
        if model.SVI:
            users_updated_orig = set(rnd.sample(data.users.keys(), \
                user_sample_count))
            user_scale = len(data.users) / user_sample_count
        else:
            users_updated_orig = set(data.users.keys())
        users_updated = set([data.users[u] for u in users_updated_orig])

        #print len(users_updated)
        training_batch = [datum for datum in data.train_triplets\
            if datum[0] in users_updated_orig]
        random.shuffle(training_batch)

        if len(training_batch) == 0:
            continue

        items_updated = set()

        for datum in training_batch:
            if data.binary:
                user, item = datum
                rating = 1
            else:
                user, item, rating = datum
            u = user
            i = item
            user = data.users[user]
            item = data.items[item]

            if item not in items_updated:
                items_updated.add(item)
                items_seen_counts[user] += 1
            params.update_shape(user, item, rating, model, data, MF_converged, \
                user_scale) # only update trust shapes after MF converged

        start = clock()
        params.update_MF(model, data, user_scale, \
            users_updated, \
            items_updated, items_seen_counts, tau0, kappa)

        # only need to update tau from default (sum of ratings)
        # when we have an interaction term
        if MF_converged:
            params.update_TF(model, data, user_scale, \
                users_updated, iteration, tau0, kappa)

        if model.intercept:
            start = clock()
            itms = list(items_updated)
            itms_L = np.array([items_seen_counts[itm] for itm in itms])
            rho = (itms_L + tau0) ** (-kappa)
            for i in xrange(len(itms)):
                if items_seen_counts[itms[i]] == 1:
                    rho[i] = 1
            if model.SVI:
                params.inter[itms] * (1-rho) + \
                    rho * (params.a_inter[itms] / params.b_inter[itms])
            else:
                params.inter[itms] = params.a_inter[itms] / params.b_inter[itms]

        C = get_elbo(model, priors, params, data) #TODO: rename; it's not the elbo!
        if model.SVI:
            delta_C.pop(0)
            delta_C.append(abs((old_C - C) / old_C))
        else:
            delta_C = abs((old_C - C) / old_C)
        old_C = C

        # save state regularly
        if iteration % 10 == 0: # 50 == 0:
            if iteration != 0:
                save_state(dire, iteration, model, params)
            tau_ave = 0 if type(params.tau)==type(params.inter) else params.tau.get_ave()
            if model.SVI:
                print iteration, C, delta_C[-1], (sum(delta_C)/10)
            else:
                print iteration, C, delta_C
            logf.write("%d\t%f\t%f\t%f\t%f\t%f\t%f\n" % \
                (iteration, C, params.theta.sum()/(model.K*data.user_count), params.beta.sum()/(model.K*data.item_count), \
                tau_ave, \
                params.eta,params.inter.sum()/data.item_count))

        params.set_to_priors(priors)

        iteration += 1

    save_state(dire, iteration, model, params)

    logf.close()

def load_data(args):
    model = model_settings.fromargs(args)
    data = dataset({}, {}, args.binary, args.directed)

    data.read_ratings(args.data + "/train.tsv")
    if model.trust:
        data.read_network(args.data + "/network.tsv")

    data.read_validation(args.data + "/validation.tsv")

    print "data loaded"
    return model, data

def load_model(fit_dir, iteration, model, priors, data):
    fname = "%s-iter%d.dat" % (fit_dir + '/params', iteration)
    print "loading from %s" % fname
    params = ptfstore.load(fname, model, False, priors, data)
    params.set_to_priors(priors)
    return params

def init_params(model, priors, data, spread=0.1):
    params = parameters(model, readonly=False, data=data, priors=priors)
    params.set_to_priors(priors)

    '''import random
    random.seed(0)
    print random.random()
    np.random.seed(0)
    print np.random.rand()'''
    print "pygsl numbers"
    import pygsl.rng as rng
    #rng.rng.set(11)
    r = rng.rng()
    r.set(11)
    # mimic's prem's initialization
    ad = np.ones((data.user_count, model.K)) * 0.3
    for i in range(data.user_count):
        for k in range(model.K):
            a = 0.01 * r.uniform()
            ad[i,k] += a
    params.a_theta = ad
    bd = np.ones(model.K) * 0.3
    for k in range(model.K):
        b = 0.1 * r.uniform()
        bd[k] += b
    params.b_theta = bd
    cd = np.ones((data.item_count, model.K)) * 0.3
    for i in range(data.item_count):
        for k in range(model.K):
            c = 0.01 * r.uniform()
            cd[i,k] += c
    params.a_beta = cd
    dd = np.ones(model.K) * 0.3
    for k in range(model.K):
        d = 0.1 * r.uniform()
        dd[k] += d
    params.b_beta = dd

    b = np.zeros(model.K) # why this and bd?? #TODO: rm bd above; it's only
    d = np.zeros(model.K) # why this and bd?? #TODO: rm bd above; it's only
    #there to mathc prem's code for random number generations, but it isn't
    #actually used

    #set_gamma_exp_init(_acurr, _Etheta, _Elogtheta, _b);
    params.logtheta = np.zeros((data.user_count, model.K))
    for i in range(data.user_count):
        for j in range(model.K):
            b[j] = 0.3 + 0.1 * r.uniform()
            params.theta[i,j] = ad[i,j] / b[j]
            #print params.logtheta[i,j]
            #print pygsl.sf.psi(ad[i,j])
            #print np.log(b[j])
            params.logtheta[i,j] = pygsl.sf.psi(ad[i,j])[0] - np.log(b[j])
            #if i == 0 and j < 3:
            #    print "[%2d,%2d]  (%5f %5f)  %5f (lg)%5f" % (i, j, ad[i,j], b[j], \
            #        params.theta[i,j], params.logtheta[i,j])

    #set_gamma_exp_init(_ccurr, _Ebeta, _Elogbeta, _d);
    params.logbeta = np.zeros((data.item_count, model.K))
    for i in range(data.item_count):
        for j in range(model.K):
            d[j] = 0.3 + 0.1 * r.uniform()
            params.beta[i,j] = cd[i,j] / d[j]
            params.logbeta[i,j] = pygsl.sf.psi(cd[i,j])[0] - np.log(d[j])
            #if i ==0 and j < 3:
            #    print "[%2d,%2d]  (%5f %5f)  %5f (lg)%5f" % (i, j, cd[i,j], d[j], \
            #        params.beta[i,j], params.logbeta[i,j])
    #set_etheta_sum();
    #set_ebeta_sum();
    #set_to_prior_users(_anext, _bnext);
    #set_to_prior_movies(_cnext, _dnext);


    # initialize
    '''
    params.theta = (np.ones((model.user_count, model.K)) * priors['a_theta'] + \
        spread * np.random.rand(model.user_count, model.K)) / priors['b_theta']
    params.theta /= model.K #params.theta.sum(axis=1)[:, np.newaxis]

    params.beta = (np.ones((model.item_count, model.K)) * priors['a_beta'] + \
        spread * np.random.rand(model.item_count, model.K)) / priors['b_beta']
    params.beta /= model.K #params.beta.sum(axis=1)[:, np.newaxis]
    '''
    if model.trust:
        params.b_tau = priors['b_tau'].copy()
    #params.tau = params.a_tau / params.b_tau
    #params.logtau = params.a_tau.psi() - log(params.b_tau)
    #print '***', params.tau.rows[0][1]
    params.set_to_priors(priors)
    return params

def set_priors_args(args, model, data):
    return set_priors(model, data, \
        args.a_theta, args.b_theta, \
        args.a_beta, args.b_beta, \
        args.a_tau, args.b_tau)

def set_priors(model, data, \
    a_theta=0.3, b_theta=0.3, a_beta=0.3, b_beta=0.3, a_tau=0.3, b_tau=0.3):
    priors = {}
    priors['a_theta'] = a_theta
    priors['b_theta'] = b_theta
    priors['a_beta'] = a_beta
    priors['b_beta'] = b_beta

    # this keeps tau small by default
    priors['a_tau'] = data.friend_matrix.const_multiply(a_tau * 1e-6)
    priors['b_tau'] = data.friend_matrix.const_multiply(b_tau * 1e-3)

    #TODO: parse these from args?
    priors['a_inter'] = 1e-30
    priors['b_inter'] = 0.3

    if model.trust:
        print "updating b_tau"
        user_weighted_ratings = dict_matrix(float, data.user_count, data.user_count)

        for user in data.users.values():
            for vser in data.friends[user]:
                if data.binary:
                    for item in data.user_data[vser]:
                        user_weighted_ratings.item_add(user, vser, \
                            1.0 / data.friend_counts[user][item])
                else:
                    for item, rating in data.user_data[vser]:
                        user_weighted_ratings.item_add(user, vser, \
                            1.0 * rating / \
                            data.friend_counts[user][item])
            priors['b_tau'] += user_weighted_ratings

    return priors

def parse_args():
    #TODO: organize this a little better
    parser = argparse.ArgumentParser(description='Infer parameters for PTF.')

    parser.add_argument('data', metavar='data', type=str, \
        help='Directory of data source.  See README for required format.')
    parser.add_argument('--out', dest='fit_dir', type=str, default='', \
        help='Model directory; all output goes here.')
    parser.add_argument('--model', dest='model', type=str, default='PTFv1', \
        help='Model type; options: PF, PTF-only (trust), PTF-simple (PTFv1)')
    parser.add_argument('--K', dest='K', type=int, default=10, \
        help='Number of components for matrix factorization.')
    parser.add_argument('--load', dest='load', action='store_true', \
        default=False, help='Load most recent model from directory.')
    parser.add_argument('--iter', dest='iteration', type=int, \
        help='Iteration number to load.')

    # flags to use an intercept or not
    parser.add_argument('--binary',dest='binary',action='store_true',default=False)
    parser.add_argument('--intercept',dest='intercept',action='store_true')
    parser.add_argument('--no-intercept',dest='intercept',action='store_false')
    parser.set_defaults(intercept=False)

    parser.add_argument('--SVI', dest='svi', action='store_true', \
        default=False, help='Use stochatic VI instead of batch VI.')


    # TODO: implement SVI (look at old stuff?)
    parser.add_argument('--verbose', dest='verbose', action='store_true',
        default=False, help='Give more output!')
    parser.add_argument('--directed', dest='directed', action='store_true',
        default=False, help='Network input is directed (default undirected)')

    # priors
    parser.add_argument('--a_theta', dest='a_theta', type=float,
        default=0.3, help='Gamma shape prior for theta.')
    parser.add_argument('--b_theta', dest='b_theta', type=float,
        default=0.3, help='Gamma rate prior for theta.')
    parser.add_argument('--a_beta', dest='a_beta', type=float,
        default=0.3, help='Gamma shape prior for beta.')
    parser.add_argument('--b_beta', dest='b_beta', type=float,
        default=0.3, help='Gamma rate prior for beta.')
    parser.add_argument('--a_tau', dest='a_tau', type=float,
        default=0.3, help='Gamma shape prior for tau.')
    parser.add_argument('--b_tau', dest='b_tau', type=float,
        default=0.3, help='Gamma rate prior for tau.')

    return parser.parse_args()

def main():
    args = parse_args()

    # load training and validation data
    # note: here, friends is a sparse matrix.....
    model, data = load_data(args)
    model.nofdiv = False

    # define the priors
    priors = set_priors_args(args, model, data)
    print "priors set"

    # create the fit dir if it doesn't exist
    args.fit_dir = args.fit_dir + ('/' if args.fit_dir != '' else '')
    if not exists(args.fit_dir):
        os.mkdir(args.fit_dir)

    # run inference to find best parameters
    params = load_model(args.fit_dir, args.iteration, model, priors, data) \
        if args.load else init_params(model, priors, data)
    print 'all params initialized'
    ptfstore.dump_model(args.fit_dir + 'model_settings.dat', data, model)
    infer(model, priors, params, data, args.fit_dir)


if __name__ == '__main__':
    np.random.seed(42)
    random.seed(43)

    main()

#TODO match pep8 style guide
# read: http://www.python.org/dev/peps/pep-0008/
# regularly chekc with pep8: "pep8 infer.py"
# TODO: read google's style guide
# http://google-styleguide.googlecode.com/svn/trunk/pyguide.html
# TODO: try pylint infer.py
