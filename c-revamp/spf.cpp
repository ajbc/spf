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
    printf("TODO: learn!\n");
}


/* PRIVATE */

void SPF::initialize_parameters() {
    int user, neighbor, n;
    for (user = 0; user < data->user_count(); user++) {
        // user influence
        for (n = 0; n < data->neighbor_count(user); n++) {
            neighbor = data->get_neighbor(user, n);
            tau(user, neighbor) = 1;
        }
    }
}
