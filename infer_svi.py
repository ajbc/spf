# PTF SVI
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
from os.path import isfile, join

import ptfstore
from ptf import *

tau0 = 1024
kappa = 0.7
user_sample_count = 100
user_scale = 1

def save_state(dire, iteration, model, params):
    # find old state file
    oldstate = ''
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
def infer(model, priors, params, data, dire=''):
    old_elbo = 0
    elbo_counter = 0

    logf = open(dire + 'log.tsv', 'w+')
    logf.write("iteration\telbo\tave.theta\tave.beta\tave_tau\tave.eta\tave.intercept\n")
    elbologf = open(dire + 'elbo.tsv', 'w+')
    elbologf.write("iteration\tratings\ttheta\tbeta\ttaut\teta\tintercept\ttotal\n")
    
    converged = False
    iteration = 0

    global user_scale
    global user_sample_count #TODO rm this line
    user_scale = 1.0 * len(model.users) / user_sample_count
    print user_scale

    items_seen_counts = defaultdict(int)
    batch_size = min(100000, len(data.train_triplets)) #0.1M
   

    count = 0
    while not converged:
        print "iteration %d" % iteration
       
        #print "    doing an all-observed pass"
        #if iteration==0:
        #    users_updated = set(model.users.values())
        #    training_batch = data.train_triplets
        #    user_scale = 1.0
        #else:
        #    users_updated = set(random.sample(model.users.values(), user_sample_count))
        #    user_scale = len(model.users) / user_sample_count * 1.0

        #    print len(users_updated)
        #    training_batch = [datum for datum in data.train_triplets\
        #        if datum[0] in users_updated]
        #    random.shuffle(training_batch)
        
        
        #user_sample_count = len(model.users)
        users_updated = set(random.sample(model.users.values(), user_sample_count))
        user_scale = len(model.users) / user_sample_count

        print len(users_updated)
        training_batch = [datum for datum in data.train_triplets\
            if datum[0] in users_updated]
        random.shuffle(training_batch)
        
        
        if len(training_batch) == 0:
            continue
        print len(training_batch)
        start = clock()

        items_updated = set()

        for datum in training_batch:
            if model.binary:
                user, item = datum
                rating = 1
            else:
                user, item, rating = datum
            user = model.users[user]
            item = model.items[item]

            if item not in items_updated:
                items_updated.add(item)
                items_seen_counts[user] += 1
            #print "sending user_scale", user_scale
            params.update_shape(user, item, rating, model, data, user_scale)
        print "      total", clock() - start
        
        print "    updating theta then beta"
        start = clock()
        params.update_MF(model, data, user_scale, \
            users_updated, \
            items_updated, items_seen_counts, tau0, kappa)
        print "      theta & beta update time", clock()-start
    

        # only need to update tau from default (sum of ratings)
        # when we have an interaction term
        start = clock()
        print "    updating tau and eta"
        params.update_TF(model, data, user_scale, \
            users_updated, iteration, tau0, kappa)
        print "      tau and eta update time", clock()-start



        if model.intercept: 
            print "    updating intercepts"
            start = clock()
            itms = list(items_updated)
            itms_L = np.array([items_seen_counts[itm] for itm in itms])
            rho = (itms_L + tau0) ** (-kappa)
            for i in xrange(len(itms)):
                if items_seen_counts[itms[i]] == 1:
                    rho[i] = 1
            antes = params.inter[itms][0]
            params.inter[itms] = params.inter[itms] * (1-rho) + \
                rho * (params.a_inter[itms] / params.b_inter[itms])
            print antes, params.inter[itms][0], rho[0], params.a_inter[itms][0],params.b_inter[itms][0]
            print "ave intercept: ", (sum(params.inter) / model.item_count)
            print "      intercept update time", clock()-start

        start = clock()
        if iteration % 100 == 0:
            elbologf.write("%d\t" % iteration)
            elbo = get_elbo(model, priors, params, data, elbologf)
            rating_likelihoods = get_log_likelihoods(model, params, data, data.validation)
            #elbo = rating_likelihoods.sum() #not really the elbo, but okay...
            #elbologf.write("%d\t%f\n" % (iteration,elbo))
        params.set_to_priors(priors)
        # save state regularly
        if iteration % 50 == 0: 
            #print ''
            if iteration != 0:
                print "    saving state"
                save_state(dire, iteration, model, params)
            tau_ave = 0 if type(params.tau)==type(params.inter) else params.tau.get_ave()
            logf.write("%d\t%f\t%f\t%f\t%f\t%f\t%f\n" % \
                (iteration, elbo, params.theta.sum()/(model.K*model.user_count), params.beta.sum()/(model.K*model.item_count), \
                tau_ave, \
                params.eta,params.inter.sum()/model.item_count))
        
        #log_state(logf, iteration, params, elbo)
        
        # assess convergence 
        if count != model.item_count:
            count = sum([1 if items_seen_counts[item] > 1 else 0 \
                for item in items_seen_counts])
            if count < model.item_count:
                print "    %f%% of items seen" % (count * 100.0 / model.item_count)

        if iteration == 10:
            print "    getting likelihood"
        elif iteration > 50 and iteration % 10 == 0 and count == model.item_count:#== len(data.train_triplets)*2:
            print "    assessing convergence"
            #CT = 0.00001 #TODO: move threshold or rename or something
            CT = 1e-6 #TODO: move threshold or rename or something
            stop = False
            if elbo >= old_elbo and old_elbo != 0 \
                and abs((elbo - old_elbo) / old_elbo) < CT and count >= model.item_count:
                stop = True
            elif elbo < old_elbo:
                elbo_counter += 1
            elif elbo > old_elbo:
                elbo_counter = 0

            if stop:#elbo_counter > 3 or stop:
                if elbo_counter > 3:
                    print "stopping due to likelihood counter"
                if stop:
                    print "stopping due to STOP flag"
                save_state(dire, iteration, model, params)
                converged = True

            old_elbo = elbo

        iteration += 1
        print "  everything else (elbo, log, etc.)", clock()-start
    
    logf.close()
    elbologf.close()

def load_data(args):
    model = model_settings.fromargs(args)
    data = dataset(model)
    
    data.read_ratings(model, args.data + "/train.tsv") 
    if model.trust: 
        data.read_network(model, args.data + "/network.tsv")

    data.read_validation(model, args.data + "/validation.tsv")
   
    print "data loaded"
    return model, data

def load_model(fit_dir, iteration, model, priors, data):
    fname = "%s-iter%d.dat" % (fit_dir + '/params', iteration)
    print "loading from %s" % fname
    params = ptfstore.load(fname, model, False, priors, data)
    params.set_to_priors(priors)
    return params

def init_params(args, model, priors, data, spread=0.1):
    params = parameters(model, readonly=False, priors=priors, data=data)
    params.set_to_priors(priors)
    
    # initialize
    params.theta = (np.ones((model.user_count, model.K)) * priors['a_theta'] + \
        spread * np.random.rand(model.user_count, model.K)) / priors['b_theta']
    params.theta /= model.K #params.theta.sum(axis=1)[:, np.newaxis]

    params.beta = (np.ones((model.item_count, model.K)) * priors['a_beta'] + \
        spread * np.random.rand(model.item_count, model.K)) / priors['b_beta']
    params.beta /= model.K #params.beta.sum(axis=1)[:, np.newaxis]
    
    if model.trust:
        params.b_tau = priors['b_tau'].copy()
        
    return params

def set_priors(args, model, data):
    priors = {}
    priors['a_theta'] = args.a_theta
    priors['b_theta'] = args.b_theta
    priors['a_beta'] = args.a_beta
    priors['b_beta'] = args.b_beta

    # this keeps tau small by default
    priors['a_tau'] = data.friend_matrix.const_multiply(args.a_tau * 1e-6)
    priors['b_tau'] = data.friend_matrix.const_multiply(args.b_tau * 1e-3)
    
    priors['a_eta'] = args.a_eta
    priors['b_eta'] = args.b_eta
          
    #TODO: parse these from args?
    priors['a_inter'] = 1e-30
    priors['b_inter'] = 0.3
    
    if model.trust:
        print "updating b_tau"
        user_weighted_ratings = dict_matrix(float, model.user_count, model.user_count)

        for user in model.users.values():
            for vser in data.friends[user]:
                if model.binary:
                    for item in data.user_data[vser]:
                        user_weighted_ratings.item_add(user, vser, \
                            1.0 / data.friend_counts[user][item])
                else:
                    for item, rating in data.user_data[vser]:
                        user_weighted_ratings.item_add(user, vser, \
                            1.0 * rating / \
                            data.friend_counts[user][item])
            priors['b_tau'] += user_weighted_ratings
    print "user 0 and friend 63", priors['b_tau'].rows[0][63]

    return priors

def parse_args():
    #TODO: organize this a little better
    parser = argparse.ArgumentParser(description='Infer parameters for PTF.')
    
    parser.add_argument('data', metavar='data', type=str, \
        help='Directory of data source.  See README for required format.')
    parser.add_argument('--out', dest='fit_dir', type=str, default='', \
        help='Model directory; all output goes here.')
    parser.add_argument('--model', dest='model', type=str, default='PTFv1', \
        help='Model type; options: PF, PTF-only (trust), PTF-simple (PTFv1), PTF-interaction (PTFv2), IATonly, IAT+MF')
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
    parser.add_argument('--directed', dest='undirected', action='store_false', 
        default=True, help='Network input is directed (default undirected)')

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
    parser.add_argument('--a_eta', dest='a_eta', type=float,
        default=1e-15, help='Gamma shape prior for eta.')
    parser.add_argument('--b_eta', dest='b_eta', type=float,
        default=0.3, help='Gamma rate prior for eta.')
    
    return parser.parse_args()

def main():
    args = parse_args()
   
    # load training and validation data 
    # note: here, friends is a sparse matrix.....
    model, data = load_data(args)
    model.nofdiv = False 

    # define the priors
    priors = set_priors(args, model, data)
    print "priors set"
    args.fit_dir = args.fit_dir + ('/' if args.fit_dir != '' else '')

    # run inference to find best parameters
    params = load_model(args.fit_dir, args.iteration, model, priors, data) if args.load else \
        init_params(args, model, priors, data)
    print 'all params initialized'
    ptfstore.dump_model(args.fit_dir + 'model_settings.dat', model)
    infer(model, priors, params, data, args.fit_dir)


if __name__ == '__main__':
    np.random.seed(42)
    random.seed(43)
    
    main()

#sparse_ratings = lil_matrix((192789, 14468), dtype=np.float64) # 1M GR
#sparse_ratings = lil_matrix((454303, 29697), dtype=np.float64) # 2M GR
#TODO match pep8 style guide
# read: http://www.python.org/dev/peps/pep-0008/
# regularly chekc with pep8: "pep8 infer.py"
# TODO: read google's style guide
# http://google-styleguide.googlecode.com/svn/trunk/pyguide.html
# TODO: try pylint infer.py
