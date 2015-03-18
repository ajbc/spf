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
    beta = fmat(settings->k, data->item_count());
    logbeta = fmat(settings->k, data->item_count());
    a_beta = fmat(settings->k, data->item_count());
    a_beta_user = fmat(settings->k, data->item_count());
    b_beta = fmat(settings->k, data->item_count());

    delta = fvec(data->item_count());
    a_delta = fvec(data->item_count());
    b_delta = settings->b_delta + data->user_count();
    a_delta_user = fvec(data->item_count());
   
    // keep track of old a_beta and a_delta for SVI
    a_beta_old  = fmat(settings->k, data->item_count());
    a_beta_old.fill(settings->a_beta);
    a_delta_old = fvec(data->item_count());
    a_delta_old.fill(settings->a_delta);
    
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
       
            bool user_converged = false;
            int user_iters = 0;
            while (!user_converged) {
                user_iters++;
                a_beta_user.zeros();
                a_delta_user.zeros();

                // look at all the user's items
                for (int j = 0; j < data->item_count(user); j++) {
                    item = data->get_item(user, j);
                    items.insert(item);
                    rating = 1;
                    //TODO: rating = data->get_train_rating(i);
                    update_shape(user, item, rating);
                }

                // update per-user parameters
                double user_change = 0;
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
            if (settings->verbose)
                printf("%d\tuser %d took %d iters to converge\n", iteration, user, user_iters);
            a_beta += a_beta_user;
            a_delta += a_delta_user;
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
                if (settings->item_bias)
                    update_delta(item);
            }
        } else if (settings->item_bias) {
            set<int>::iterator it;
            for (it = items.begin(); it != items.end(); it++) {
                item = *it;
                if (iter_count[item] == 0)
                    iter_count[item] = 0;
                iter_count[item]++;
                if (settings->item_bias)
                    update_delta(item);
            }
        }


        // check for convergence
        if (on_final_pass) {
            printf("Final pass complete\n");
            converged = true;
            
            old_likelihood = likelihood;
            likelihood = get_ave_log_likelihood();
            delta_likelihood = abs((old_likelihood - likelihood) / 
                old_likelihood);
            log_convergence(iteration, likelihood, delta_likelihood);
        } else if (iteration >= settings->max_iter) {
            printf("Reached maximum number of iterations.\n");
            converged = true;
            
            old_likelihood = likelihood;
            likelihood = get_ave_log_likelihood();
            delta_likelihood = abs((old_likelihood - likelihood) / 
                old_likelihood);
            log_convergence(iteration, likelihood, delta_likelihood);
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
            if (settings->verbose) {
                printf("delta: %f\n", delta_likelihood);
                printf("old:   %f\n", old_likelihood);
                printf("new:   %f\n", likelihood);
            }
            if (iteration >= settings->min_iter &&
                delta_likelihood < settings->likelihood_delta) {
                printf("Model converged.\n");
                converged = true;
            } else if (iteration >= settings->min_iter &&
                likelihood_decreasing_count >= 2) {
                printf("Likelihood decreasing.\n");
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

        if (converged && settings->final_pass && !on_final_pass) {
            printf("final pass on all users.\n");
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
    
    if (!settings->social_only) {
        prediction += accu(theta.col(user) % beta.col(item));
    }

    if (settings->item_bias) {
        prediction += delta(item);
    }

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
    
    eval(this, &Model::predict, settings->outdir, data, false, 11,
        settings->verbose, label, write_rankings);
    
    time(&end_time);
    log_time(-1, difftime(end_time, start_time));
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
            delta(item) = data->popularity(item);
        }
    }
}

void SPF::reset_helper_params() {
    a_tau = data->network_spmat * settings->a_tau;

    a_theta.fill(settings->a_theta);
    b_theta.fill(settings->b_theta);
    a_beta.fill(settings->a_beta);
    b_beta.fill(settings->b_beta);
    a_delta.fill(settings->a_delta);
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
    float phi_B = 0;
    // we don't need to do a similar check for factor only because
    // sparse matrices play nice when empty
    if (!settings->social_only) {
        phi_MF = exp(logtheta.col(user) + logbeta.col(item));
        phi_sum += accu(phi_MF);
    }

    if (settings->item_bias) {
        phi_B = delta(item);
        phi_sum += phi_B;
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
        a_beta_user.col(item) += phi_MF * scale;
    }

    if (settings->item_bias) {
        a_delta(item) += (phi_B / (phi_sum * rating)) * scale;
    }
}

double SPF::update_tau(int user) {
    int neighbor, n;
    double old, change, total;
    change = 0;
    total = 0;
    for (n = 0; n < data->neighbor_count(user); n++) {
        neighbor = data->get_neighbor(user, n);
        
        old = tau(neighbor, user);
        total += tau(neighbor, user);

        tau(neighbor, user) = a_tau(neighbor, user) / b_tau(neighbor, user);
        // fake log!
        logtau(neighbor, user) = exp(gsl_sf_psi(a_tau(neighbor, user)) - log(b_tau(neighbor, user)));
        
        change += abs(old - tau(neighbor, user));
    }

    return total==0 ? 0 : change / total;
}

double SPF::update_theta(int user) {
    double change = accu(abs(theta(user) - (a_theta(user) / b_theta(user))));
    double total = accu(theta(user));

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

void SPF::update_delta(int item) {
    if (settings->svi) {
        double rho = pow(iter_count[item] + settings->delay, 
            -1 * settings->forget);
        a_delta(item) = (1 - rho) * a_delta_old(item) + rho * a_delta(item);
        a_delta_old(item) = a_delta(item);
    }
    delta(item) = a_delta(item) / b_delta;
}

double SPF::get_ave_log_likelihood() {
    double prediction, likelihood = 0;
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
    fprintf(file, "%d\t%f\t%f\t%f\t%d\t%f\t%f\t%f\n", user, 
        rmse, mae, rank, first, crr, ncrr, ndcg);
}
