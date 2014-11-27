#include "data.h"

Data::Data(bool bin, bool dir) {
    binary = bin;
    directed = dir;
}

void Data::read_ratings(string filename) {
    FILE *fileptr = fopen(filename.c_str(), "r");

    int user, item, rating;
    while ((fscanf(fileptr, "%d\t%d\t%d\n", &user, &item, &rating) != EOF)) {
        printf("user %d, item %d, rating %d\n", user, item, rating);

        // map user and item ids
        if (user_ids.count(user) == 0)
            user_ids[user] = user_ids.size(); 
        if (item_ids.count(item) == 0)    
            item_ids[item] = item_ids.size();

    }
    fclose(fileptr);
}

void Data::read_network(string filename) {
}

void Data::read_validation(string filename) {
}

void Data::save_summary(string filename) {
    FILE * file = fopen(filename.c_str(), "w");
    printf("TODO\n");
    
    fprintf(file, "num users:\t%d\n", user_ids.size());
    fprintf(file, "num items:\t%d\n", item_ids.size());
    fprintf(file, "num ratings:\t%d\n", 0);
    fprintf(file, "network connections:\t%d\n", 0);
    fclose(file);
}
