#include "eval.h"

// random generator to break ties
gsl_rng* rand_gen = gsl_rng_alloc(gsl_rng_taus);

// helper function to write out per-user info
void log_user(FILE* file, Data *data, int user, int heldout, double rmse, double mae,
    double rank, int first, double crr, double ncrr, double ndcg, bool stats) {
    if (stats)
        fprintf(file, "%d\t%d\t%d\t%d\t%d\t%d\t%f\t%f\t%f\t%d\t%f\t%f\t%f\n", user, 
            data->user_id(user), heldout, data->item_count(user), 
            data->neighbor_count(user), data->connectivity(user), 
            rmse, mae, rank, first, crr, ncrr, ndcg);
    else
        fprintf(file, "%d\t%d\t%f\t%f\t%f\t%d\t%f\t%f\t%f\n", user, 
            data->user_id(user),
            rmse, mae, rank, first, crr, ncrr, ndcg);
    return;
}

void log_item(FILE* file, Data *data, int item, int heldout, double rmse, double mae,
    double rank, int first, double crr, double ncrr, double ndcg, bool stats) {
    if (stats)
        fprintf(file, "%d\t%d\t%d\t%d\t%f\t%f\t%f\t%d\t%f\t%f\t%f\n", item, 
            data->item_id(item), data->popularity(item), heldout,
            rmse, mae, rank, first, crr, ncrr, ndcg);
    else
        fprintf(file, "%d\t%d\t%f\t%f\t%f\t%d\t%f\t%f\t%f\n", item, 
            data->item_id(item),
            rmse, mae, rank, first, crr, ncrr, ndcg);
    return;
}

// helper function to sort predictions properly
bool prediction_compare(const pair<double,int>& itemA, 
    const pair<double, int>& itemB) {
    // if the two values are equal, sort by popularity!
    if (itemA.first == itemB.first) {
        return gsl_rng_uniform_int(rand_gen, 2) != 0;
    }
    return itemA.first > itemB.first;
}


// take a prediction function as an argument
void eval(Model* model, double (Model::*prediction)(int,int), string outdir, Data* data, bool stats, 
    unsigned long int seed, bool verbose, string label, bool write_rankings) { 
    // random generator to break ties
    gsl_rng_set(rand_gen, seed);

    // test the final model fit
    printf("evaluating model on held-out data\n");
    
    FILE* file = fopen((outdir+"/rankings_" + label + ".tsv").c_str(), "w");
    if (write_rankings)
        fprintf(file, "user.map\tuser.id\titem.map\titem.id\tpred\trank\trating\n");
    
    FILE* user_file = fopen((outdir+"/user_eval_" + label + ".tsv").c_str(), "w");
    if (stats)
        fprintf(user_file, "user.map\tuser.id\tnum.heldout\tnum.train\tdegree\tconnectivity\trmse\tmae\tave.rank\tfirst\tcrr\tncrr\tndcg\n");
    else
        fprintf(user_file, "user.map\tuser.id\trmse\tmae\tave.rank\tfirst\tcrr\tncrr\tndcg\n");
     
    FILE* item_file = fopen((outdir+"/item_eval_" + label + ".tsv").c_str(), "w");
    fprintf(item_file, "item.map\titem.id\tpopularity\theldout\trmse\tmae\tave.rank\tfirst\tcrr\tncrr\tndcg\n");
    
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
    
    int user, item, rating, rank;
    list<pair<double, int> > ratings;
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

            ratings.push_back(make_pair((model->*prediction)(user,item), item));
        }
        
        ratings.sort(prediction_compare);

        rank = 0;
        int test_count = data->num_test(user);
        while (user_heldout < test_count && !ratings.empty()) {
            pair<double, int> pred_set = ratings.front();
            item = pred_set.second;
            rating = data->test_ratings(user, item);
            pred = pred_set.first;
            rank++;
            if (rank <= 1000 && write_rankings) { // TODO: make this threshold a command line arg
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
            user_mae, user_rank, first, user_crr, user_ncrr, user_ndcg, stats);

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
    if (!write_rankings)
        remove((outdir+"/rankings_" + label + ".tsv").c_str());

    
    // per item attibutes
    double item_rmse = 0;
    double item_mae = 0;
    int item_heldout = 0;
    double item_rank = 0;
    double item_crr = 0;
    double item_ncrr = 0;
    double item_ncrr_normalizer = 0;
    double item_ndcg = 0;
    double item_ndcg_normalizer = 0;

    for (set<int>::iterator iter_item = data->test_items.begin(); 
        iter_item != data->test_items.end();
        iter_item++){

        item = *iter_item;
        if (verbose) {
            printf("item %d\n", item);
        }

        item_rmse = 0;
        item_mae = 0;
        item_rank = 0;
        first = 0;
        item_crr = 0;
        item_ncrr_normalizer = 0;
        item_ndcg = 0;
        item_ndcg_normalizer = 0;
        item_heldout = 0;

        for (set<int>::iterator iter_user = data->test_users.begin(); 
            iter_user != data->test_users.end();
            iter_user++){

            user = *iter_user;

            // don't rank items that we've already seen
            if (data->ratings(user, item) != 0 || 
                data->in_validation(user, item))
                continue;

            total_pred++;

            ratings.push_back(make_pair((model->*prediction)(user, item), user));
        }
        
        ratings.sort(prediction_compare);

        rank = 0;
        int test_count = data->num_test_item(item);
        while (item_heldout < test_count && !ratings.empty()) {
            pair<double, int> pred_set = ratings.front();
            user = pred_set.second;
            rating = data->test_ratings(user, item);
            pred = pred_set.first;
            rank++;

            // compute metrics only on held-out items
            if (rating != 0) {
                item_heldout++;

                item_rmse += pow(rating - pred, 2);
                item_mae += abs(rating - pred);
                item_rank += rank;
                item_crr += 1.0 / rank;
                item_ncrr_normalizer += 1.0 / item_heldout;

                item_ndcg += rating / log(rank + 1);
                item_ndcg_normalizer += rating / log(item_heldout + 1);

                if (first == 0)
                    first = rank;
            }
            
            ratings.pop_front();
        }
        while (!ratings.empty()){
            ratings.pop_front();
        }

        // log this item's metrics
        item_rmse = sqrt(item_rmse / item_heldout);
        item_mae /= item_heldout;
        item_rank /= item_heldout;
        item_ncrr = item_crr / item_ncrr_normalizer;
        item_ndcg /= item_ndcg_normalizer;
        
        log_item(item_file, data, item, item_heldout, item_rmse, 
            item_mae, item_rank, first, item_crr, item_ncrr, item_ndcg, stats);
    }
    fclose(item_file);
    
    // write out results
    file = fopen((outdir+"/eval_summary_" + label + ".dat").c_str(), "w");
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
}
