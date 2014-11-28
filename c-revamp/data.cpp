#include "data.h"

Data::Data(bool bin, bool dir) {
    binary = bin;
    directed = dir;
}

void Data::read_ratings(string filename) {
    printf("[TODO]");
    FILE* fileptr = fopen(filename.c_str(), "r");

    int user, item, rating;
    while ((fscanf(fileptr, "%d\t%d\t%d\n", &user, &item, &rating) != EOF)) {
        //printf("user %d, item %d, rating %d\n", user, item, rating);

        // map user and item ids
        if (user_ids.count(user) == 0)
            user_ids[user] = user_count() - 1;
        if (item_ids.count(item) == 0)    
            item_ids[item] = item_count() - 1;

    }
    fclose(fileptr);
}

void Data::read_network(string filename) {
    // initialize network data structures
    network = new vector<int>[user_count()];

    // read in network data from file
    FILE* fileptr = fopen(filename.c_str(), "r");

    int user, neighbor, u, n;
    while ((fscanf(fileptr, "%d\t%d\n", &user, &neighbor) != EOF)) {
        // skip connections in which either user or neighbor is seen in training
        if (user_ids.count(user) == 0 || user_ids.count(neighbor) == 0)
            continue;
        
        u = user_ids[user];
        n = user_ids[neighbor];
        
        if (!has_connection(u, n))
            network[u].push_back(n);
        if (!directed && !has_connection(n, u))
            network[n].push_back(u);
    }
    fclose(fileptr);
}

void Data::read_validation(string filename) {
    printf("[TODO]");
}

void Data::save_summary(string filename) {
    FILE* file = fopen(filename.c_str(), "w");
    printf("[TODO]");
    
    fprintf(file, "num users:\t%d\n", user_count());
    fprintf(file, "num items:\t%d\n", item_count());
    fprintf(file, "num ratings:\t%d\n", 0);

    int nc = 0;
    for (int user = 0; user < user_count(); user++)
        nc += network[user].size();
    if (directed) {
        fprintf(file, "network connections:\t%d directed\n", nc);
    } else {
        fprintf(file, "network connections:\t%d undirected\n", (nc/2));
    }
    fclose(file);
}

int Data::user_count() {
    return user_ids.size();
}

bool Data::has_connection(int user, int neighbor) {
    for (int i = 0; i < network[user].size(); i++) {
        if (network[user][i] == neighbor)
            return true;
    }
    return false;
}

int Data::item_count() {
    return item_ids.size();
}

int Data::neighbor_count(int user) {
    return network[user].size();
}

int Data::get_neighbor(int user, int n) {
    return network[user][n];
}
