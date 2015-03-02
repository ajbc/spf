#include <getopt.h>
#include <stdio.h>
#include <list>
#include <map>
#include "utils.h"
#include "data.h"
#include "eval.h"

void print_usage_and_exit() {
    // print usage information
    printf("*************************** Predict by Poupularity *****************************\n");
    printf("(c) Copyright 2014 Allison J.B. Chaney  ( achaney@cs.princeton.edu )\n");
    printf("Distributed under ???; see LICENSE file for details.\n");
    
    printf("\nusage:\n");
    printf(" ./librec_eval [options]\n");
    printf("  --help            print help information\n");
    printf("  --verbose         print extra information while running\n");

    printf("\n");
    printf("  --out {dir}       save directory, required\n");
    printf("  --data {dir}      data directory, required\n");
    
    printf("********************************************************************************\n");

    exit(0);
}

class LibRec: protected Model {
    private:
        map<int,map<int,float> > preds;

    public:
        LibRec(Data* d) {
            data = d;
        }

        void read_preds(string outdir) {
            int user, item;
            float r, prediction;

            FILE* fileptr = fopen((outdir+"/ratings.dat").c_str(), "r");
            printf("about to read ratings from %s\n", (outdir+"/ratings.dat").c_str());
            while (fscanf(fileptr, "%d %d %f %f\n", &user, &item, &r, &prediction) != EOF) {
                preds[user][item] = prediction;
            }
            fclose(fileptr);
        }

        double predict(int user, int item) {
            return preds[user][item];
        }

        void evaluate(string outdir, bool verbose) {
            eval(this, &Model::predict, outdir, data, false, 11, verbose, "final", true);
        }
};


int main(int argc, char* argv[]) {
    if (argc < 2) print_usage_and_exit();

    // variables to store command line args + defaults
    string outdir = "";
    string datadir = "";
    bool verbose = false;

    // ':' after a character means it takes an argument
    const char* const short_options = "hqo:d:";
    const struct option long_options[] = {
        {"help",            no_argument,       NULL, 'h'},
        {"verbose",         no_argument,       NULL, 'q'},
        {"out",             required_argument, NULL, 'o'},
        {"data",            required_argument, NULL, 'd'},
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

    // read in the ratings

    LibRec lr = LibRec(data);
    
    printf("starting to read ratings\n");
    lr.read_preds(outdir);
    
    lr.evaluate(outdir, verbose);
    
    delete data;

    return 0;
}
