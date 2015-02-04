#include "spf.h"

SPF::SPF(model_settings* model_set, Data* dataset) {
    settings = model_set;
    data = dataset;

    // user influence
    printf("\tinitializing user influence (tau)\n");
    tau = sp_fmat(data->user_count(), data->user_count());
    logtau = sp_fmat(data->user_count(), data->user_count());
    a_tau = sp_fmat(data->user_count(), data->user_count());
    b_tau = sp_fmat(data->user_count(), data->user_count());

    // user preferences
    printf("\tinitializing user preferences (theta)\n");
    theta = fmat(settings->k, data->user_count());
    logtheta = fmat(settings->k, data->user_count());
    a_theta = fmat(settings->k, data->user_count());
    b_theta = fmat(settings->k, data->user_count());

    // item attributes
    printf("\tinitializing item attributes (beta)\n");
    printf("\t%d users and %d items\n", data->user_count(), data->item_count());
    beta  = fmat(settings->k, data->item_count());
    logbeta  = fmat(settings->k, data->item_count());
    a_beta  = fmat(settings->k, data->item_count());
    a_beta_user = sp_fmat(settings->k, data->item_count());
    b_beta  = fmat(settings->k, data->item_count());
   
    // keep track of old a_beta for SVI
    if (settings->svi) {
        a_beta_old  = fmat(settings->k, data->item_count());
        a_beta_old.fill(settings->a_beta);
    }
    
    printf("\tsetting random seed\n");
    rand_gen = gsl_rng_alloc(gsl_rng_taus);
    gsl_rng_set(rand_gen, (long) settings->seed); // init the seed
    
    initialize_parameters(); 

    scale = settings->svi ? data->user_count() / settings->sample_size : 1;
}

void SPF::learn() {
    double old_likelihood, delta_likelihood, likelihood = -1e10; 
    int likelihood_decreasing_count = 0;
    time_t start_time, end_time;
    
    int iteration = 0;
    char iter_as_str[4];
    bool converged = false;
    bool on_final_pass = false;

    while (!converged) {
        time(&start_time);
        iteration++;
        printf("iteration %d\n", iteration);
        
        reset_helper_params();

        // update rate for user preferences
        b_theta.each_col() += sum(beta, 1);

        set<int> items;
        int user, item, rating;
        for (int i = 0; i < settings->sample_size; i++) {
            if (settings->svi)
                user = gsl_rng_uniform_int(rand_gen, data->user_count());
            else
                user = i;
            printf("user %d\t(%d / %d)\n", user,i,settings->sample_size);
       
            bool user_converged = false;
            int user_iters = 0;
            while (!user_converged) {
                user_iters++;
                printf("\t%d\n", user_iters);
                a_beta_user.zeros();
                float user_change = 0;

                // look at all the user's items
                for (int j = 0; j < data->item_count(user); j++) {
                    item = data->get_item(user, j);
                    items.insert(item);
                    rating = 1;
                    //TODO: rating = data->get_train_rating(i);
                    update_shape(user, item, rating);
                }

                // update per-user parameters
                if (!settings->factor_only)
                    user_change += update_tau(user);
                if (!settings->social_only)
                    user_change += update_theta(user);
                if (!settings->social_only && !settings->factor_only) {
                    user_change /= 2;

                    // if the updates are less than 1% change, the local params have converged
                    if (user_change < 0.01)
                        user_converged = true;
                } else {
                    // if we're only looking at social or factor (not combined)
                    // then the user parameters will always have converged with
                    // a single pass (since there's nothing to balance against)
                    user_converged = true;
                }
            }
            printf("%d user %d took %d iters to converge\n", iteration, user, user_iters);
            a_beta += a_beta_user;
        }
    
        if (!settings->social_only) {
            // update rate for item attributes
            b_beta.each_col() += sum(theta, 1);
            
            // update per-item parameters
            set<int>::iterator it;
            for (it = items.begin(); it != items.end(); it++) {
                item = *it;
                if (iter_count[item] == 0)
                    iter_count[item] = 0;
                iter_count[item]++;
                update_beta(item);
            }
        }

        // check for convergence
        if (on_final_pass) {
            converged = true;
        } else if (iteration % settings->conv_freq == 0) {
            old_likelihood = likelihood;
            likelihood = get_ave_log_likelihood();

            if (likelihood < old_likelihood)
                likelihood_decreasing_count += 1;
            else
                likelihood_decreasing_count = 0;
            delta_likelihood = abs((old_likelihood - likelihood) / 
                old_likelihood);
            log_convergence(iteration, likelihood, delta_likelihood);
            printf("delta: %f\n", delta_likelihood);
            printf("old:   %f\n", old_likelihood);
            printf("new:   %f\n", likelihood);
            if (iteration >= settings->min_iter &&
                delta_likelihood < settings->likelihood_delta) {
                printf("Model converged.\n");
                converged = true;
            } else if (iteration >= settings->min_iter &&
                likelihood_decreasing_count >= 2) {
                printf("Likelihood decreasing.\n");
                converged = true;
            } else if (iteration >= settings->max_iter) {
                printf("Reached maximum number of iterations.\n");
                converged = true;
            }
        }
        
        // save intermediate results
        if (!converged && settings->save_freq > 0 && 
            iteration % settings->save_freq == 0) {
            printf(" saving\n");
            sprintf(iter_as_str, "%04d", iteration);
            save_parameters(iter_as_str);
        }

        // intermediate evaluation
        if (!converged && settings->eval_freq > 0 &&
            iteration % settings->eval_freq == 0) {
            sprintf(iter_as_str, "%04d", iteration);
            evaluate(iter_as_str);
        }

        time(&end_time);
        log_time(iteration, difftime(end_time, start_time));

        if (converged && settings->final_pass) {
            printf("final pass on all users\n.");
            on_final_pass = true;
            converged = false;

            // we need to modify some settings for the final pass
            // things should look exactly like batch here 
            settings->set_stochastic_inference(false);
            settings->set_sample_size(data->user_count()); 
            scale = 1;
        }
    }
    
    save_parameters("final");
}

double SPF::predict(int user, int item) {
    double prediction = settings->social_only ? 1e-10 : 0;
    
    prediction += accu(tau.col(user) % data->ratings.col(item));
    if (!settings->social_only)
        prediction += accu(theta.col(user) % beta.col(item));

    return prediction;
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

void SPF::evaluate() {
    evaluate("final", true);
}

void SPF::evaluate(string label) {
    evaluate(label, false);
}

void SPF::evaluate(string label, bool write_rankings) {
    time_t start_time, end_time;
    time(&start_time);
    
    FILE* file;
    if (write_rankings) {
        file = fopen((settings->outdir+"/rankings_"+label+".tsv").c_str(), "w");
        fprintf(file, "user.map\tuser.id\titem.map\titem.id\tpred\trank\trating\n");
    }
    // TODO: add likelihood here
    
    FILE* user_file = fopen((settings->outdir+"/user_eval_"+label+".tsv").c_str(), "w");
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
    
    int user, item, rating, rank;
    list<pair<pair<double, int>, int> > ratings;
    int total_pred = 0;
    
    for (set<int>::iterator iter_user = data->test_users.begin(); 
        iter_user != data->test_users.end();
        iter_user++){

        user = *iter_user;
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

            ratings.push_back(make_pair(make_pair(predict(user,item), 
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
        
        log_user(user_file, user, user_heldout, user_rmse, 
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
    if (write_rankings)
        fclose(file);
    
    
    // write out results
    file = fopen((settings->outdir+"/eval_summary_"+label+".dat").c_str(), "w");
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
    
    time(&end_time);
    log_time(total_pred, difftime(end_time, start_time));
}



/* PRIVATE */

void SPF::initialize_parameters() {
    int user, neighbor, n, item, i, k;
    if (!settings->factor_only) {
        for (user = 0; user < data->user_count(); user++) {
            // user influence
            for (n = 0; n < data->neighbor_count(user); n++) {
                neighbor = data->get_neighbor(user, n);
                tau(neighbor, user) = 1.0;
                logtau(neighbor, user) = log(1.0 + 1e-5);

                double all = settings->b_tau;
                for (i = 0; i < data->item_count(neighbor); i++) { 
                    item = data->get_item(neighbor, i);
                    all += data->ratings(neighbor, item);
                } //TODO: this doeesn't need to be done as much... only one time per user (U), not UxU times
                b_tau(neighbor, user) = all;
            }
        }
    }

    if (!settings->social_only) {
        // user preferences
        for (user = 0; user < data->user_count(); user++) {
            for (k = 0; k < settings->k; k++) {
                theta(k, user) = (settings->a_theta + 
                    gsl_rng_uniform_pos(rand_gen))
                    / (settings->b_theta);
                logtheta(k, user) = log(theta(k, user));
            }
            theta.col(user) /= accu(theta.col(user));
        }
        
        // item attributes
        for (item = 0; item < data->item_count(); item++) {
            for (k = 0; k < settings->k; k++) {
                beta(k, item) = (settings->a_beta +
                    gsl_rng_uniform_pos(rand_gen))
                    / (settings->b_beta);
                logbeta(k, item) = log(beta(k, item));
            }
            beta.col(item) /= accu(beta.col(item));
        }
    }
}

void SPF::reset_helper_params() {
    a_tau = data->network_spmat * settings->a_tau;

    a_theta.fill(settings->a_theta);
    b_theta.fill(settings->b_theta);
    a_beta.fill(settings->a_beta);
    b_beta.fill(settings->b_beta);
}

void SPF::save_parameters(string label) {
    FILE* file;
    if (!settings->factor_only) {
        // save tau
        file = fopen((settings->outdir+"/tau-"+label+".dat").c_str(), "w");
        fprintf(file, "uid\torig.uid\tvid\torig.vid\ttau\n");
        int user, neighbor, n;
        double tau_uv;
        for (user = 0; user < data->user_count(); user++) {
            for (n = 0; n < data->neighbor_count(user); n++) {
                neighbor = data->get_neighbor(user, n);
                tau_uv = tau(neighbor, user);
                fprintf(file, "%d\t%d\t%d\t%d\t%e\n", user, data->user_id(user),
                    neighbor, data->user_id(neighbor), tau_uv);
            }
        }
        fclose(file);
    }
    
    if (!settings->social_only) {
        int k;

        // write out theta
        file = fopen((settings->outdir+"/theta-"+label+".dat").c_str(), "w");
        for (int user = 0; user < data->user_count(); user++) {
            fprintf(file, "%d\t%d", user, data->user_id(user));
            for (k = 0; k < settings->k; k++)
                fprintf(file, "\t%e", theta(k, user));
            fprintf(file, "\n");
        }
        fclose(file);
        
        // write out beta
        file = fopen((settings->outdir+"/beta-"+label+".dat").c_str(), "w");
        for (int item = 0; item < data->item_count(); item++) {
            fprintf(file, "%d\t%d", item, data->item_id(item));
            for (k = 0; k < settings->k; k++)
                fprintf(file, "\t%e", beta(k, item));
            fprintf(file, "\n");
        }
        fclose(file);
    }
}

void SPF::update_shape(int user, int item, int rating) {
    sp_fmat phi_SF = logtau.col(user) % data->ratings.col(item);

    double phi_sum = accu(phi_SF);

    fmat phi_MF;
    // we don't need to do a similar check for factor only because
    // sparse matrices play nice when empty
    if (!settings->social_only) {
        phi_MF = exp(logtheta.col(user) + logbeta.col(item));
        phi_sum += accu(phi_MF);
    }

    if (phi_sum == 0)
        return;

    if (!settings->factor_only) {
        phi_SF /= phi_sum * rating;
        int neighbor;
        for (int n = 0; n < data->neighbor_count(user); n++) {
            neighbor = data->get_neighbor(user, n);
            a_tau(neighbor, user) += phi_SF(neighbor, 0);
        }
    }

    if (!settings->social_only) {
        phi_MF /= phi_sum * rating;
        a_theta.col(user) += phi_MF;
        a_beta_user.col(item)  += phi_MF * scale;
    }
}

float SPF::update_tau(int user) {
    int neighbor, n;
    float old, change, total;
    for (n = 0; n < data->neighbor_count(user); n++) {
        neighbor = data->get_neighbor(user, n);
        
        old = tau(neighbor, user);
        total += tau(neighbor, user);

        tau(neighbor, user) = a_tau(neighbor, user) / b_tau(neighbor, user);
        // fake log!
        logtau(neighbor, user) = exp(gsl_sf_psi(a_tau(neighbor, user)) - log(b_tau(neighbor, user)));
        
        change += abs(old - tau(neighbor, user));
    }

    return change / total;
}

float SPF::update_theta(int user) {
    float change = accu(abs(theta(user) - (a_theta(user) / b_theta(user))));
    float total = accu(theta(user));

    theta(user) = a_theta(user) / b_theta(user);
    for (int k = 0; k < settings->k; k++)
        logtheta(k, user) = gsl_sf_psi(a_theta(k, user));
    logtheta(user) = logtheta(user) - log(b_theta(user));

    return change / total;
}

void SPF::update_beta(int item) {
    if (settings->svi) {
        double rho = pow(iter_count[item] + settings->delay, 
            -1 * settings->forget);
        a_beta(item) = (1 - rho) * a_beta_old(item) + rho * a_beta(item);
        a_beta_old(item) = a_beta(item);
    }
    beta(item)  = a_beta(item) / b_beta(item);
    for (int k = 0; k < settings->k; k++)
        logbeta(k, item) = gsl_sf_psi(a_beta(k, item));
    logbeta(item) = logbeta(item) - log(b_beta(item));
}

double SPF::get_ave_log_likelihood() {
    double likelihood, prediction;
    int user, item, rating;
    for (int i = 0; i < data->num_validation(); i++) {
        user = data->get_validation_user(i);
        item = data->get_validation_item(i);
        rating = data->get_validation_rating(i);

        prediction = predict(user, item);
        
        likelihood +=
            log(prediction) * rating - log(factorial(rating)) - prediction;
    }

    return likelihood / data->num_validation();
}

void SPF::log_convergence(int iteration, double ave_ll, double delta_ll) {
    FILE* file = fopen((settings->outdir+"/log_likelihood.dat").c_str(), "a");
    fprintf(file, "%d\t%f\t%f\n", iteration, ave_ll, delta_ll);
    fclose(file);
}

void SPF::log_time(int iteration, double duration) {
    FILE* file = fopen((settings->outdir+"/time_log.dat").c_str(), "a");
    fprintf(file, "%d\t%.f\n", iteration, duration);
    fclose(file);
}

void SPF::log_user(FILE* file, int user, int heldout, double rmse, double mae,
    double rank, int first, double crr, double ncrr, double ndcg) {
    if (settings->factor_only) {
        fprintf(file, "%d\t%d\t%d\t%d\t%f\t%f\t%f\t%d\t%f\t%f\t%f\n", user, 
            data->user_id(user), heldout, data->item_count(user), 
            rmse, mae, rank, first, crr, ncrr, ndcg);
    
    } else {
        fprintf(file, "%d\t%d\t%d\t%d\t%d\t%d\t%f\t%f\t%f\t%d\t%f\t%f\t%f\n", user, 
            data->user_id(user), heldout, data->item_count(user), 
            data->neighbor_count(user), data->connectivity(user), 
            rmse, mae, rank, first, crr, ncrr, ndcg);
    }
}
