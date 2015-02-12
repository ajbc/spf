#include <getopt.h>
#include <stdio.h>
#include <list>
#include <map>
#include "utils.h"
#include "data.h"

void print_usage_and_exit() {
    // print usage information
    printf("*************************** Predict by Poupularity *****************************\n");
    printf("(c) Copyright 2014 Allison J.B. Chaney  ( achaney@cs.princeton.edu )\n");
    printf("Distributed under ???; see LICENSE file for details.\n");
    
    printf("\nusage:\n");
    printf(" spf [options]\n");
    printf("  --help            print help information\n");
    printf("  --verbose         print extra information while running\n");

    printf("\n");
    printf("  --out {dir}       save directory, required\n");
    printf("  --data {dir}      data directory, required\n");
    
    printf("********************************************************************************\n");

    exit(0);
}

// helper function to write out per-user info
void log_user(FILE* file, Data *data, int user, int heldout, double rmse, double mae,
    double rank, int first, double crr, double ncrr, double ndcg) {
    fprintf(file, "%d\t%d\t%d\t%d\t%f\t%f\t%f\t%d\t%f\t%f\t%f\n", user, 
        data->user_id(user), heldout, data->item_count(user), 
        rmse, mae, rank, first, crr, ncrr, ndcg);
    return;
}

// helper function to sort predictions properly
bool prediction_compare(const pair<pair<double,int>, int>& itemA, 
    const pair<pair<double, int>, int>& itemB) {
    // if the two values are equal, sort by popularity!
    if (itemA.first.first == itemB.first.first) {
        if (itemA.first.second == itemB.first.second)
            return itemA.second < itemB.second;
        return itemA.first.second > itemB.first.second;
    }
    return itemA.first.first > itemB.first.first;
}

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
    
    printf("\tsaving data stats\t\t...\t");
    data->save_summary(outdir + "/data_stats.txt");
    printf("done\n");
    
    printf("********************************************************************************\n");
    printf("commencing model evaluation\n");
    if (!file_exists(datadir + "/test.tsv")) {
        printf("testing data file (test.tsv) doesn't exist!  Exiting.\n");
        exit(-1);
    }
    printf("reading testing data\t\t...\t");
    data->read_test(datadir + "/test.tsv");
    printf("done\n");


    // read in the ratings
    printf("starting to read ratings\n");
    map<int,map<int,float> > preds;

    int user, item;
    float r, prediction;

    FILE* fileptr = fopen((outdir+"/ratings.dat").c_str(), "r");
    printf("about to read ratings from %s\n", (outdir+"/ratings.dat").c_str());
    while (fscanf(fileptr, "%d %d %f %f\n", &user, &item, &r, &prediction) != EOF) {
        preds[user][item] = prediction;
    }
    fclose(fileptr);
    
    
    // test the final model fit
    printf("evaluating model on held-out data\n");
    
    FILE* file = fopen((outdir+"/rankings_final.tsv").c_str(), "w");
    fprintf(file, "user.map\tuser.id\titem.map\titem.id\tpred\trank\trating\n");
    
    FILE* user_file = fopen((outdir+"/user_eval_final.tsv").c_str(), "w");
    fprintf(user_file, "user.map\tuser.id\tnum.heldout\tnum.train\tdegree\tconnectivity\trmse\tmae\tave.rank\tfirst\tcrr\tncrr\tndcg\n");
    
    // overall metrics to track
    double rmse = 0;
    double mae = 0;
    double aggr_rank = 0;
    double crr = 0;
    double user_sum_rmse = 0;
    double user_sum_mae = 0;
    double user_sum_rank = 0;
    double user_sum_first = 0;
    double user_sum_crr = 0;
    double user_sum_ncrr = 0;
    double user_sum_ndcg = 0;

    // per user attibutes
    double user_rmse = 0;
    double user_mae = 0;
    int user_heldout = 0;
    double user_rank = 0;
    int first = 0;
    double user_crr = 0;
    double user_ncrr = 0;
    double user_ncrr_normalizer = 0;
    double user_ndcg = 0;
    double user_ndcg_normalizer = 0;

    // helper var for evaluation (used for mulitple metrics)
    double local_metric;

    // helper var to hold predicted rating
    double pred;
        
    // overall attributes to track
    int user_count = 0;
    int heldout_count = 0;
    
    int rating, rank;
    list<pair<pair<double, int>, int> > ratings;
    int total_pred = 0;
    
    for (set<int>::iterator iter_user = data->test_users.begin(); 
        iter_user != data->test_users.end();
        iter_user++){

        user = *iter_user;
        if (verbose) {
            printf("user %d\n", user);
        }
        user_count++;

        user_rmse = 0;
        user_mae = 0;
        user_rank = 0;
        first = 0;
        user_crr = 0;
        user_ncrr_normalizer = 0;
        user_ndcg = 0;
        user_ndcg_normalizer = 0;
        user_heldout = 0;

        for (set<int>::iterator iter_item = data->test_items.begin(); 
            iter_item != data->test_items.end();
            iter_item++){

            item = *iter_item;

            // don't rank items that we've already seen
            if (data->ratings(user, item) != 0 || 
                data->in_validation(user, item))
                continue;

            total_pred++;

            double pred = preds[user][item];

            ratings.push_back(make_pair(make_pair(pred, 
                data->popularity(item)), item));
        }
        
        ratings.sort(prediction_compare);

        rank = 0;
        int test_count = data->num_test(user);
        while (user_heldout < test_count && !ratings.empty()) {
            pair<pair<double, int>, int> pred_set = ratings.front();
            item = pred_set.second;
            rating = data->test_ratings(user, pred_set.second);
            pred = pred_set.first.first;
            rank++;
            if (rank <= 1000) { // TODO: make this threshold a command line arg
                fprintf(file, "%d\t%d\t%d\t%d\t%f\t%d\t%d\n", user, data->user_id(user),
                    item, data->item_id(item), pred, rank, rating);
            }

            // compute metrics only on held-out items
            if (rating != 0) {
                user_heldout++;
                heldout_count++;

                local_metric = pow(rating - pred, 2);
                rmse += local_metric;
                user_rmse += local_metric;
                
                local_metric = abs(rating - pred);
                mae += local_metric;
                user_mae += local_metric;

                aggr_rank += rank;
                user_rank += rank;

                local_metric = 1.0 / rank;
                user_crr += local_metric;
                crr += local_metric;
                user_ncrr_normalizer += 1.0 / user_heldout;

                user_ndcg += rating / log(rank + 1);
                user_ndcg_normalizer += rating / log(user_heldout + 1);

                if (first == 0)
                    first = rank;
            }
            
            ratings.pop_front();
        }
        while (!ratings.empty()){
            ratings.pop_front();
        }

        // log this user's metrics
        user_rmse = sqrt(user_rmse / user_heldout);
        user_mae /= user_heldout;
        user_rank /= user_heldout;
        user_ncrr = user_crr / user_ncrr_normalizer;
        user_ndcg /= user_ndcg_normalizer;
        
        log_user(user_file, data, user, user_heldout, user_rmse, 
            user_mae, user_rank, first, user_crr, user_ncrr, user_ndcg);

        // add this user's metrics to overall metrics
        user_sum_rmse += user_rmse;
        user_sum_mae += user_mae;
        user_sum_rank += user_rank;
        user_sum_first += first;
        user_sum_crr += user_crr;
        user_sum_ncrr += user_ncrr;
        user_sum_ndcg += user_ndcg;
    }
    fclose(user_file);
    fclose(file);
    
    
    // write out results
    file = fopen((outdir+"/eval_summary_final.dat").c_str(), "w");
    fprintf(file, "metric\tuser average\theldout pair average\n");
    fprintf(file, "RMSE\t%f\t%f\n", user_sum_rmse/user_count, 
        sqrt(rmse/heldout_count));
    fprintf(file, "MAE\t%f\t%f\n", user_sum_mae/user_count, mae/heldout_count);
    fprintf(file, "rank\t%f\t%f\n", user_sum_rank/user_count, 
        aggr_rank/heldout_count);
    fprintf(file, "first\t%f\t---\n", user_sum_first/user_count);
    fprintf(file, "CRR\t%f\t%f\n", user_sum_crr/user_count, crr/heldout_count);
    fprintf(file, "NCRR\t%f\t---\n", user_sum_ncrr/user_count);
    fprintf(file, "NDCG\t%f\t---\n", user_sum_ndcg/user_count);
    fclose(file);
    
    
    delete data;

    return 0;
}
