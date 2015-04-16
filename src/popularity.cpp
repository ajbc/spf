#include <getopt.h>
#include <stdio.h>
#include <list>
#include "utils.h"
#include "data.h"
#include "eval.h"

void print_usage_and_exit() {
    // print usage information
    printf("*************************** Predict by Poupularity *****************************\n");
    printf("(c) Copyright 2014-2015 Allison J.B. Chaney  ( achaney@cs.princeton.edu )\n");
    printf("Distributed under MIT License; see LICENSE file for details.\n");
    
    printf("\nusage:\n");
    printf(" ./pop [options]\n");
    printf("  --help            print help information\n");
    printf("  --verbose         print extra information while running\n");

    printf("\n");
    printf("  --out {dir}       save directory, required\n");
    printf("  --data {dir}      data directory, required\n");
    printf("  --seed {seed}     the random seed, default from time\n");
    
    printf("********************************************************************************\n");

    exit(0);
}

// helper function to write out per-user info
void log_user(FILE* file, Data *data, int user, int heldout, double rmse, double mae,
    double rank, int first, double crr, double ncrr, double ndcg) {
    fprintf(file, "%d\t%d\t%d\t%d\t%d\t%d\t%f\t%f\t%f\t%d\t%f\t%f\t%f\n", user, 
        data->user_id(user), heldout, data->item_count(user), 
        data->neighbor_count(user), data->connectivity(user), 
        rmse, mae, rank, first, crr, ncrr, ndcg);
    return;
}

void log_item(FILE* file, Data *data, int item, int heldout, double rmse, double mae,
    double rank, int first, double crr, double ncrr, double ndcg) {
    fprintf(file, "%d\t%d\t%d\t%d\t%f\t%f\t%f\t%d\t%f\t%f\t%f\n", item, 
        data->item_id(item), data->popularity(item), heldout,
        rmse, mae, rank, first, crr, ncrr, ndcg);
    return;
}

class Popularity: protected Model {
    public:
        double predict(int user, int item) {
            return data->popularity(item) * 5 / data->item_count();
        }

        void evaluate(Data* d, string outdir, bool verbose, long seed) {
            data = d;
            eval(this, &Model::predict, outdir, data, true, seed, verbose, "final", true, false);
        }
};

int main(int argc, char* argv[]) {
    if (argc < 2) print_usage_and_exit();

    // variables to store command line args + defaults
    string outdir = "";
    string datadir = "";
    bool verbose = false;
    long seed = 11;

    // ':' after a character means it takes an argument
    const char* const short_options = "hqo:d:s:";
    const struct option long_options[] = {
        {"help",            no_argument,       NULL, 'h'},
        {"verbose",         no_argument,       NULL, 'q'},
        {"out",             required_argument, NULL, 'o'},
        {"data",            required_argument, NULL, 'd'},
        {"seed",            required_argument, NULL, 's'},
        {NULL, 0, NULL, 0}};

  
    int opt = 0; 
    while(true) {
        opt = getopt_long(argc, argv, short_options, long_options, NULL);
        switch(opt) {
            case 'h':
                print_usage_and_exit();
                break;
            case 'q':
                verbose = true;
                break;
            case 'o':
                outdir = optarg;
                break;
            case 'd':
                datadir = optarg;
                break;
            case 's':
                seed = atoi(optarg);
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

    if (outdir == "") {
        printf("No output directory specified.  Exiting.\n");
        exit(-1);
    }
    
    if (dir_exists(outdir)) {
        string rmout = "rm -rf " + outdir;
        system(rmout.c_str());
    }
    make_directory(outdir);
    printf("output directory: %s\n", outdir.c_str());
    
    if (datadir == "") {
        printf("No data directory specified.  Exiting.\n");
        exit(-1);
    }
    
    if (!dir_exists(datadir)) {
        printf("data directory %s doesn't exist!  Exiting.\n", datadir.c_str());
        exit(-1);
    }
    printf("data directory: %s\n", datadir.c_str());

    if (!file_exists(datadir + "/train.tsv")) {
        printf("training data file (train.tsv) doesn't exist!  Exiting.\n");
        exit(-1);
    }
    
    if (!file_exists(datadir + "/validation.tsv")) {
        printf("validation data file (validation.tsv) doesn't exist!  Exiting.\n");
        exit(-1);
    }
    

    // read in the data
    printf("********************************************************************************\n");
    printf("reading data\n");
    Data *data = new Data(true, false);
    printf("\treading training data\t\t...\t");
    data->read_ratings(datadir + "/train.tsv");
    printf("done\n");

    // read in the network for data stats only
    printf("\treading network data\t\t...\t");
    data->read_network(datadir + "/network.tsv");
    printf("done\n");

    printf("\treading validation data\t\t...\t");
    data->read_validation(datadir + "/validation.tsv");
    printf("done\n");
    
    if (!file_exists(datadir + "/test.tsv")) {
        printf("testing data file (test.tsv) doesn't exist!  Exiting.\n");
        exit(-1);
    }
    printf("\treading testing data\t\t...\t");
    data->read_test(datadir + "/test.tsv");
    printf("done\n");
    
    printf("\tsaving data stats\t\t...\t");
    data->save_summary(outdir + "/data_stats.txt");
    printf("done\n");
    
    printf("********************************************************************************\n");
    printf("commencing model evaluation\n");
    
    // test the final model fit
    Popularity pop = Popularity();
    pop.evaluate(data, outdir, verbose, seed);

    delete data;
    
    return 0;
}
