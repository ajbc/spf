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
    int iteration = 0;
    printf("TODO: learn!\n");
    while (iteration < 100) {
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
    }
    
    save_parameters("final");
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

                double overlap = settings->b_tau;
                for (i = 0; i < data->item_count(user); i++) { 
                    item = data->get_item(user, i);
                    overlap += data->ratings(neighbor, item);
                }
                b_tau(neighbor, user) = overlap;
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
