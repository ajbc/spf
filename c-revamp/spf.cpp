#include "spf.h"

SPF::SPF(model_settings* model_set, Data* dataset) {
    settings = model_set;
    data = dataset;

    // user influence
    tau = sp_mat(data->user_count(), data->user_count());
    //logtau = sp_mat(data->user_count(), data->user_count());
    a_tau = sp_mat(data->user_count(), data->user_count());
    b_tau = sp_mat(data->user_count(), data->user_count());

    // user preferences
    theta = mat(settings->k, data->user_count());
    a_theta = mat(settings->k, data->user_count());
    b_theta = mat(settings->k, data->user_count());

    // item attributes
    beta  = mat(settings->k, data->item_count());
    a_beta  = mat(settings->k, data->item_count());
    b_beta  = mat(settings->k, data->item_count());
    
    rand_gen = gsl_rng_alloc(gsl_rng_taus);
    gsl_rng_set(rand_gen, (long) settings->seed); // init the seed
    
    initialize_parameters(); 
}

void SPF::learn() {
    double old_likelihood, delta_likelihood, likelihood = -1e10; 
    int likelihood_decreasing_count = 0;
    
    int iteration = 0;
    char iter_as_str[4];
    bool converged = false;
    while (!converged) {
        iteration++;
        printf("iteration %d\n", iteration);
        
        reset_helper_params();

        int user, item, rating;
        for (int i = 0; i < data->num_training(); i++) {
            user = data->get_train_user(i);
            item = data->get_train_item(i);
            rating = data->get_train_rating(i);
            update_shape(user, item, rating);
        }
    
        if (!settings->social_only)
            update_MF();
        
        if (!settings->factor_only)
            update_SF();

        if (iteration % settings->save_lag == 0) {
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
            } else {
                printf(" saving\n");
                sprintf(iter_as_str, "%04d", iteration);
                save_parameters(iter_as_str);
            }
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
 

bool prediction_compare(const double* first, const double* second) {
    return first[1] > second[1];
}
void SPF::predict_and_rank() {
    printf("predicting ratings and ranking items for each user\n");
    FILE* file = fopen((settings->outdir+"/rankings.tsv").c_str(), "w");
    //fprintf(file, "user\titem\tpred\trank\trating\n");
    int user, item;
    list<pair<double, int> > ratings;
    for (set<int>::iterator iter_user = data->test_users.begin(); 
        iter_user != data->test_users.end();
        iter_user++){

        user = *iter_user;

        for (set<int>::iterator iter_item = data->test_items.begin(); 
            iter_item != data->test_items.end();
            iter_item++){

            item = *iter_item;

            // don't rank items that we've already seen
            if (data->ratings(user, item) != 0 || 
                data->in_validation(user, item))
                continue;

            ratings.push_back(make_pair(predict(user,item), item));
        }
        
        ratings.sort();
        ratings.reverse();

        int rank = 0;
        while (!ratings.empty()) {
            pair<double, int> pred_set = ratings.front();
            fprintf(file, "%d\t%d\t%f\t%d\t%d\n", user, 
                (int)pred_set.second, pred_set.first, ++rank, 
                (int)data->test_ratings(user, pred_set.second));
            ratings.pop_front();
        }
    }
    fclose(file);
}

void SPF::evaluate_rankings() {
    printf("computing evaluation metrics from rankings\n");
    // compute eval metrics from ratings/rankings
    FILE* file = fopen((settings->outdir+"/rankings.tsv").c_str(), "r");
    FILE* user_file = fopen((settings->outdir+"/user_eval.tsv").c_str(), "w");
    fprintf(user_file, "user.id\tnum.heldout\trmse\tmae\n");

    // overall attributes to track
    int user_count = 0;
    int heldout_count = 0;

    // overall metrics to track
    double rmse = 0;
    double mae = 0;
    double user_sum_rmse = 0;
    double user_sum_mae = 0;

    // per user attibutes
    double user_rmse = 0;
    double user_mae = 0;
    int user_heldout = 0;
        
    // per line variables
    int user, item, rating, rank;
    double pred;
    int prev_user = -1;
    double local_metric;

    //fscanf(file, "%s\n");
    while ((fscanf(file, "%d\t%d\t%lf\t%d\t%d\n", &user, &item, &pred, &rank,
        &rating) != EOF)) {
        if (user != prev_user) {
            if (prev_user != -1) {
                user_rmse = sqrt(user_rmse / user_heldout);
                user_mae = user_mae / user_heldout;
                
                log_user(user_file, prev_user, user_heldout, user_rmse, user_mae);
                user_sum_rmse += user_rmse;
                user_sum_mae += user_mae;
            }

            user_count++;
            user_rmse = 0;
            user_mae = 0;

            user_heldout = 0;
            prev_user = user;
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
            //printf("+%f = %f u%d=%f\n", local_metric, mae/heldout_count, user, user_mae);
        }
    }
    user_rmse = sqrt(user_rmse / user_heldout);
    user_mae = user_mae / user_heldout;
    log_user(user_file, user, user_heldout, user_rmse, user_mae);
    // aggregate metrics
    user_sum_rmse += user_rmse;
    user_sum_mae += user_mae;
    user_count++;

    fclose(file);
    fclose(user_file);


    // write out results
    file = fopen((settings->outdir+"/eval_summary.dat").c_str(), "w");
    fprintf(file, "metric\tuser average\theldout pair average\n");
    fprintf(file, "RMSE\t%f\t%f\n", user_sum_rmse/user_count, 
        sqrt(rmse/heldout_count));
    fprintf(file, "MAE\t%f\t%f\n", user_sum_mae/user_count, mae/heldout_count);
    fclose(file);
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
            }
            theta.col(user) /= accu(theta.col(user));
        }
        
        // item attributes
        for (item = 0; item < data->item_count(); item++) {
            for (k = 0; k < settings->k; k++)
                beta(k, item) = (settings->a_beta +
                    gsl_rng_uniform_pos(rand_gen))
                    / (settings->b_beta);
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
    sp_mat phi_SF = tau.col(user) % data->ratings.col(item);

    double phi_sum = accu(phi_SF);

    mat phi_MF;
    // we don't need to do a similar cheeck for factor only because
    // sparse matrices play nice when empty
    if (!settings->social_only) {
        phi_MF = theta.col(user) % beta.col(item);
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
        a_beta.col(item)  += phi_MF;        
    }
}

void SPF::update_MF() {
    b_theta.each_col() += sum(beta, 1);
    theta = a_theta / b_theta;
    
    b_beta.each_col() += sum(theta, 1);
    beta  = a_beta  / b_beta;
}

void SPF::update_SF() {
    int user, neighbor, n;
    double a, b, c;
    for (user = 0; user < data->user_count(); user++) {
        for (n = 0; n < data->neighbor_count(user); n++) {
            neighbor = data->get_neighbor(user, n);
            tau(neighbor, user) = a_tau(neighbor, user) / b_tau(neighbor, user);
        }
    }
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

void SPF::log_user(FILE* file, int user, int heldout, double rmse, double mae) {
    fprintf(file, "%d\t%d\t%f\t%f\n", user, heldout, rmse, mae);
}
