#include <getopt.h>
#include "spf.h"


#include <stdio.h>
//#include <string.h>
//#include "ctr.h"

//gsl_rng * RANDOM_NUMBER = NULL;

void print_usage_and_exit() {
    // print usage information
    printf("********************** Social Poisson Factorization (SPF) **********************\n");
    printf("(c) Copyright 2014 Allison J.B. Chaney  ( achaney@cs.princeton.edu )\n");
    printf("Distributed under ???; see LICENSE file for details.\n");
    
    printf("\nusage:\n");
    printf(" spf [options]\n");
    printf("  --help            print help information\n");

    printf("\n");
    printf("  --out {dir}       save directory, required\n");
    printf("  --data {dir}      data directory, required\n");
    
    printf("\n");
    printf("  --svi             use stochastic VI (instead of batch VI)\n");
    printf("                    default off for < 10M ratings in training\n");
    printf("  --batch           use batch VI (instead of SVI)\n");
    printf("                    default on for < 10M ratings in training\n");
    
    printf("\n");
    printf("  --a_theta {a}     shape hyperparamter to theta (user preferences); default 0.3\n");
    printf("  --b_theta {b}     rate hyperparamter to theta (user preferences); default 0.3\n");
    printf("  --a_beta {a}      shape hyperparamter to beta (item attributes); default 0.3\n");
    printf("  --b_beta {b}      rate hyperparamter to beta (item attributes); default 0.3\n");
    printf("  --a_tau {a}       shape hyperparamter to tau (user influence); default 2\n");
    printf("  --b_tau {b}       rate hyperparamter to tau (user influence); default 5\n");
  
    printf("\n");
    printf("  --social-only     only consider social aspect of factorization (SF)\n");
    printf("  --factor-only     only consider general factors (no social; PF)\n");

    printf("\n");
    printf("  --binary          assume ratings are binary (instead of default integer)\n");
    printf("  --directed        assume network is directed (instead of default undirected)\n");
    
    printf("\n");
    printf("  --seed {seed}     the random seed, default from time\n");
    printf("  --save_lag {lag}  the saving frequency, default 20\n");
    printf("                    -1 means no savings for intermediate results\n");
    printf("  --max_iter {max}  the max number of iterations, default 300\n");
    printf("  --min_iter {min}  the min number of iterations, default 30\n");
    printf("  --converge {c}    the change in log likelihood required for convergence\n");
    printf("                    default 1e-6\n");
    printf("\n");

    printf("  --sample {size}   the stochastic sample size, default 1000\n");
    printf("  --svi_delay {t}   SVI delay >= 0 to down-weight early samples, default 1024\n");
    printf("  --svi_forget {k}  SVI forgetting rate (0.5,1], default 0.75\n");
    printf("\n");

    printf("  --K {K}           the number of general factors, default 100\n");

    printf("********************************************************************************\n");

    exit(0);
}

int main(int argc, char* argv[]) {
    if (argc < 2) print_usage_and_exit();

    char filename[500];
   
    // variables to store command line args + defaults
    string out = "";
    string data = "";

    bool svi = false;
    bool batchvi = false;

    double a_theta = 0.3;
    double b_theta = 0.3;
    double a_beta = 0.3;
    double b_beta = 0.3;
    double a_tau = 2;
    double b_tau = 5;

    // these are really bools, but typed as integers to play nice with getopt
    int social_only = 0;
    int factor_only = 0;
    int binary = 0;
    int directed = 0;

    time_t t; time(&t);
    long   seed = (long) t;
    int    save_lag = 20;
    int    max_iter = 300;
    int    min_iter = 30;
    double converge_delta = 1e-6;
    
    int    sample_size = 1000;
    double svi_delay = 1024;
    double svi_forget = 0.75;

    int    k = 100;

    // ':' after a character means it takes an argument
    const char* const short_options = "ho:d:vb1:2:3:4:5:6:s:l:x:m:c:a:e:f:k:";
    const struct option long_options[] = {
        {"help",            no_argument,       NULL, 'h'},
        {"out",             required_argument, NULL, 'o'},
        {"data",            required_argument, NULL, 'd'},
        {"svi",             required_argument, NULL, 'v'},
        {"batch",           required_argument, NULL, 'b'},
        {"a_theta",         required_argument, NULL, '1'},
        {"b_theta",         required_argument, NULL, '2'},
        {"a_beta",          required_argument, NULL, '3'},
        {"b_beta",          required_argument, NULL, '4'},
        {"a_tau",           required_argument, NULL, '5'},
        {"b_tau",           required_argument, NULL, '6'},
        {"social_only",     no_argument, &social_only, 1},
        {"factor_only",     no_argument, &factor_only, 1},
        {"binary",          no_argument, &binary, 1},
        {"directed",        no_argument, &directed, 1},
        {"seed",            required_argument, NULL, 's'},
        {"save_lag",        required_argument, NULL, 'l'},
        {"max_iter",        required_argument, NULL, 'x'},
        {"min_iter",        required_argument, NULL, 'm'},
        {"converge",        required_argument, NULL, 'c'},
        {"sample",          required_argument, NULL, 'a'},
        {"delay",           required_argument, NULL, 'e'},
        {"forget",          required_argument, NULL, 'f'},
        {"K",               required_argument, NULL, 'k'},
        {NULL, 0, NULL, 0}};

  
    int opt = 0; 
    while(true) {
        opt = getopt_long(argc, argv, short_options, long_options, NULL);
        switch(opt) {
            case 'h':
                print_usage_and_exit();
                break;
            case 'o':
                out = optarg;
                break;
            case 'd':
                data = optarg;
                break;
            case 'v':
                svi = true;
                break;
            case 'b':
                batchvi = true;
                break;
            case '1':
                a_theta = atof(optarg);
                break;
            case '2':
                b_theta = atof(optarg);
                break;
            case '3':
                a_beta = atof(optarg);
                break;
            case '4':
                b_beta = atof(optarg);
                break;
            case '5':
                a_tau = atof(optarg);
                break;
            case '6':
                b_tau = atof(optarg);
                break;
            case 's':
                seed = atoi(optarg);
                break;
            case 'l':
                save_lag = atoi(optarg);
                break;
            case 'x':
                max_iter =  atoi(optarg);
                break;    
            case 'm':
                min_iter =  atoi(optarg);
                break;    
            case 'c':
                converge_delta =  atoi(optarg);
                break;    
            case 'a':
                sample_size = atoi(optarg);
                break;
            case 'e':
                svi_delay = atof(optarg);
                break;
            case 'f':
                svi_forget = atof(optarg);
                break;
            case 'k':
                k = atoi(optarg);
                break;
            case -1:
                break;
            case '?':
                print_usage_and_exit();
                break;
            default:
                break;
        }
        if (opt == -1)
            break;
    }

    // print information
    printf("********************************************************************************\n");

    if (out == "") {
        printf("No output directory specified.  Exiting.\n");
        exit(-1);
    }
    
    if (dir_exists(out)) {
        string rmout = "rm -rf " + out;
        system(rmout.c_str());
    }
    make_directory(out);
    printf("output directory: %s\n", out.c_str());
    
    if (data == "") {
        printf("No data directory specified.  Exiting.\n");
        exit(-1);
    }
    
    if (!dir_exists(data)) {
        printf("data directory %s doesn't exist!  Exiting.\n", data.c_str());
        exit(-1);
    }
    printf("data directory: %s\n", data.c_str());

    if (!file_exists(data + "/train.tsv")) {
        printf("training data file (train.tsv) doesn't exist!  Exiting.\n");
        exit(-1);
    }
    
    if (!file_exists(data + "/validation.tsv")) {
        printf("validation data file (validation.tsv) doesn't exist!  Exiting.\n");
        exit(-1);
    }
    
    if (!factor_only && !file_exists(data + "/network.tsv")) {
        printf("network data file (network.tsv) doesn't exist!  Exiting.\n");
        exit(-1);
    }
    
    if (social_only && factor_only) {
        printf("Model cannot be both social only (SF) and factor only (PF).  Exiting.\n");
        exit(-1);
    }
    
    if (svi && batchvi) {
        printf("Inference method cannot be both stochatic (SVI) and batch.  Exiting.\n");
        exit(-1);
    }
    
    printf("\nmodel specification:\n");
    
    if (social_only) {
        printf("\tsocial factorization (SF)   [ social factors only ]\n");
    } else if (factor_only) {
        printf("\tPoisson factorization (PF)   [ general preference factors only ]\n");
    } else {
        printf("\tsocial Poisson factorization (SPF)\n");
    }
    if (!social_only) {
        printf("\tK = %d   (number of latent factors for general preferences)\n", k);
    }

    printf("\nshape and rate hyperparameters:\n");
    if (!social_only) {
        printf("\ttheta (%.2f, %.2f)\n", a_theta, b_theta);
        printf("\tbeta  (%.2f, %.2f)\n", a_beta, b_beta);
    }
    if (!factor_only) {
        printf("\ttau   (%.2f, %.2f)\n", a_tau, b_tau);
    }
    

    printf("\ndata attributes:\n");
    
    if (binary) {
        printf("\tbinary ratings\n");
    } else {
        printf("\tinteger ratings\n");
    }
    
    if (!factor_only) {
        if (directed) {
            printf("\tdirected network\n");
        } else {
            printf("\tundirected network\n");
        }
    }
    
    printf("\ninference parameters:\n");
    printf("\tseed:                                     %d\n", (int)seed);
    printf("\tsave lag:                                 %d\n", save_lag);
    printf("\tmaximum number of iterations:             %d\n", max_iter);
    printf("\tminimum number of iterations:             %d\n", min_iter);
    printf("\tchange in log likelihood for convergence: %f\n", converge_delta);
    
   
    if (!batchvi) {
        printf("\nStochastic variational inference parameters\n");
        if (!svi)
            printf("  (may not be used, pending dataset size)\n");
        printf("\tsample size:                              %d\n", sample_size);
        printf("\tSVI delay (tau):                          %f\n", svi_delay);
        printf("\tSVI forgetting rate (kappa):              %f\n", svi_forget);
    } else {
        printf("\nusing batch variational inference\n");
    }
    

    model_settings settings;
    settings.set(out, data, svi, a_theta, b_theta, a_beta, b_beta, a_tau, b_tau,
        (bool) social_only, (bool) factor_only, (bool) binary, (bool) directed,
        seed, save_lag, max_iter, min_iter, converge_delta, 
        sample_size, svi_delay, svi_forget, k);

    // read in the data
    printf("********************************************************************************\n");
    printf("reading data\n");
    Data *dataset = new Data(settings.binary, settings.directed);
    printf("\treading training data\t\t...\t");
    dataset->read_ratings(settings.datadir + "/train.tsv");
    printf("done\n");

    if (!factor_only) {
        printf("\treading network data\t\t...\t");
        dataset->read_network(settings.datadir + "/network.tsv");
        printf("done\n");
    }
    printf("\treading validation data\t\t...\t");
    dataset->read_validation(settings.datadir + "/validation.tsv");
    printf("done\n");
    printf("\tsaving data stats\t\t...\t");
    dataset->save_summary(out + "/data_stats.txt");
    printf("done\n");
    
    // save the run settings
    printf("Saving settings\n");
    if (!svi && !batchvi) {
        if (dataset->num_training() > 10000000) {
            settings.set_stochastic_inference(true);
            printf("using SVI (based on dataset size)\n");
        } else {
            printf("using batch VI (based on dataset size)\n");
        }
    }
    printf("user count %d\n", dataset->user_count());
    if (!settings.svi)
        settings.set_sample_size(dataset->user_count());
    printf("sample size %d\n", settings.sample_size);
    
    settings.save(out + "/settings.txt");

    // create model instance; learn!
    printf("\ncreating model instance\n");
    SPF *model = new SPF(&settings, dataset);
    printf("commencing model inference\n");
    model->learn();

    // test the model fit TODO: make this optional (--test_only, --no_test)
    printf("********************************************************************************\n");
    printf("commencing model evaluation\n");
    if (!file_exists(data + "/test.tsv")) {
        printf("testing data file (test.tsv) doesn't exist!  Exiting.\n");
        exit(-1);
    }
    printf("reading testing data\t\t...\t");
    dataset->read_test(settings.datadir + "/test.tsv");
    printf("done\n");
    printf("evaluating model on held-out data\n");
    model->evaluate();
    
    delete model;
    delete dataset;

    return 0;
}
