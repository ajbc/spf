#include "spf.h"

SPF::SPF(model_settings* model_set, Data* dataset) {
    settings = model_set;
    data = dataset;

    // user influence
    tau = sp_mat(data->user_count(), data->user_count());

    
    initialize_parameters(); 
    // initialize random number generator (???)
    //RANDOM_NUMBER = new_random_number_generator(random_seed);
}

void SPF::learn() {
    int iteration = 0;
    printf("TODO: learn!\n");
    while (iteration < 10) {
        iteration++;
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
    int user, neighbor, n;
    if (!settings->factor_only) {
        for (user = 0; user < data->user_count(); user++) {
            // user influence
            for (n = 0; n < data->neighbor_count(user); n++) {
                neighbor = data->get_neighbor(user, n);
                tau(neighbor, user) = 1.0;
            }
        }
    }
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
                tau_uv = tau(user, neighbor);
                fprintf(file, "%d\t%d\t%d\t%d\t%f\n", user, data->user_id(user),
                    neighbor, data->user_id(neighbor), tau_uv);
            }
        }
        fclose(file);
    }
}

void SPF::update_shape(int user, int item, int rating) {
    //TODO: phi_MF of size (settings->k)
    
    sp_mat phi_SF = tau.col(user) % data->ratings.col(item); // element-wise *
    //phi_SF.print();
    
    printf("NEXT STEP");
    
}

void SPF::update_MF() {
    printf("TODO\n");
}

void SPF::update_SF() {
    printf("TODO\n");
}
