#include "spf.h"

SPF::SPF(model_settings* model_set, Data* dataset) {
    settings = model_set;
    data = dataset;

    // user influence
    tau = sp_mat(data->user_count(), data->user_count());

  /// init random numbe generator
  //RANDOM_NUMBER = new_random_number_generator(random_seed);
}

void SPF::learn() {
    printf("TODO: learn!\n");
}
