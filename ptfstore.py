import ptf
import numpy as np
#from ptf import *
#class ptfstore?

def dump_model(fname, data, model):
    f = open(fname, 'w+')

    f.write("%d,%d,%d,%d,%d,%d,%d,%d,%d\n" % \
        (data.user_count, data.item_count, model.K, model.MF, model.trust, \
         model.intercept, data.directed, data.binary, model.sorec))

    user_mapping = ''
    for user in data.users:
        user_mapping += ' %d:%d' % (user, data.users[user])
    f.write("%s\n" % user_mapping.strip())

    item_mapping = ''
    for item in data.items:
        if not isinstance(item, basestring):
            item_mapping += ' %d:%d' % (item, data.items[item])
        else:
            item_mapping += ' %s:%d' % (item, data.items[item])
    f.write("%s\n" % item_mapping.strip())

    f.close()

def dump(fname, model, params):
    f = open(fname, 'w+')

    # intercepts
    if model.intercept:
        intercepts = ''
        for i in params.inter:
            intercepts += ' %.5e' % i
        f.write("%s\n" % intercepts.strip())


    # tau
    if model.trust:
        tau = ''
        for row in params.tau.rows:
            for col in params.tau.rows[row]:
                tau += ' %d:%d:%.5e' % (row,col,params.tau.rows[row][col])
        f.write("%s\n" % tau.strip())

    # theta
    if model.MF:
        row_id = 0
        for row in params.theta:
            r = 'U%d' % row_id
            for val in row:
                r += " %.5e" % val
            f.write("%s\n" % r)
            row_id += 1

    # beta
    if model.MF:
        row_id = 0
        for row in params.beta:
            r = 'I%d' % row_id
            for val in row:
                r += " %.5e" % val
            f.write("%s\n" % r.strip())
            row_id += 1

    f.close()

#print ptf.parameters

def load_model(fname):
    f = open(fname, 'r')

    user_count, item_count, K, MF, trust, intercept, directed, binary, sorec = \
        [int(token) for token in f.readline().strip().split(',')]

    user_mapping = {}
    for token in f.readline().strip().split(' '):
        a,b = token.split(':')
        user_mapping[int(a)] = int(b)

    item_mapping = {}
    for token in f.readline().strip().split(' '):
        a,b = token.split(':')
        if a.startswith('f'):
            item_mapping[a] = int(b)
        else:
            item_mapping[int(a)] = int(b)

    model = ptf.model_settings(K, MF, trust, intercept, sorec)

    f.close()
    return model, directed, binary, user_count, item_count, user_mapping, item_mapping

def load(fname, model, data, readonly=True, priors=False):
    f = open(fname, 'r')

    print "  in load"
    params = ptf.parameters(model, readonly, data, priors)

    # itercepts
    if model.intercept:
        i = 0
        for intercept in f.readline().strip().split(' '):
            if i >= data.item_count:
                continue
            params.inter[i] = float(intercept)
            i += 1
        print "INTERCEPT (ave %f)" % (sum(params.inter) / len(params.inter))

    # tau
    if model.trust:
        count = 0.0
        val = 0.0
        for tau in f.readline().strip().split(' '):
            if tau.strip() == '':
                continue
            user, friend, trust = tau.split(':')
            count += 1
            val += float(trust)
            #print user, friend, trust
            params.tau.rows[int(user)][int(friend)] = float(trust)
        print "TAU (ave # %f, ave val %f)" % (count/data.user_count, val/count)#count / model.user_count, val / count)

    # theta
    if model.MF:
        print "THETA"
        params.theta = np.zeros((data.user_count, model.K))
        params.beta = np.zeros((data.item_count, model.K))
        aves = np.zeros(model.K)
        for user in xrange(data.user_count):
            i = 0
            L = f.readline()
            #for val in [float(v) for v in f.readline().strip().split(' ')]:
            for val in [float(v) for v in L.strip().split(' ')[1:]]:
                params.theta[user,i] = val
                aves[i] += val
                i += 1
        print "aves: ", (aves / data.user_count)

        # beta
        print "BETA"
        aves = np.zeros(model.K)
        for item in xrange(data.item_count):
            i = 0
            L = f.readline()
            #print L
            #for val in [float(v) for v in f.readline().strip().split(' ')]:
            for val in [float(v) for v in L.strip().split(' ')[1:]]:
                params.beta[item,i] = val
                aves[i] += val
                i += 1
        print "aves: ", (aves / data.user_count)

    f.close()

    return params
