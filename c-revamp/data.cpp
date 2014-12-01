#include "data.h"

Data::Data(bool bin, bool dir) {
    binary = bin;
    directed = dir;
    has_network = false;
}

void Data::read_ratings(string filename) {
    // read in training data
    FILE* fileptr = fopen(filename.c_str(), "r");

    int user, item, rating;
    set<long> dupe_checker;
    while ((fscanf(fileptr, "%d\t%d\t%d\n", &user, &item, &rating) != EOF)) {
        // look for duplicate entries; this is not a perfect check, but it's ok
        long dupe_id = item * 100000 + user * 100 +  rating;
        if (dupe_checker.count(dupe_id) != 0)
            continue;
        dupe_checker.insert(dupe_id);

        // map user and item ids
        if (user_ids.count(user) == 0) {
            user_ids[user] = user_count() - 1;
            reverse_user_ids[user_ids[user]] = user;
        }
        if (item_ids.count(item) == 0) {
            item_ids[item] = item_count() - 1;
            reverse_item_ids[item_ids[item]] = item;
        }

        if (rating != 0) {
            train_users.push_back(user_ids[user]);
            train_items.push_back(item_ids[item]);
            train_ratings.push_back(binary ? 1 : rating);
        }
    }
    fclose(fileptr);

    umat locations = umat(2, num_training());
    colvec values = colvec(num_training());
    user_items = new vector<int>[user_count()];
    for (int i = 0; i < num_training(); i++) {
        locations(0,i) = train_users[i]; // row
        locations(1,i) = train_items[i]; // col
        values(i) = train_ratings[i];
        user_items[train_users[i]].push_back(train_items[i]);
    }
    ratings = sp_mat(locations, values, user_count(), item_count());
}

void Data::read_network(string filename) {
    // initialize network data structures
    network = new vector<int>[user_count()];
    network_spmat = sp_mat(user_count(), user_count());
    has_network = true;

    // read in network data from file
    FILE* fileptr = fopen(filename.c_str(), "r");

    int user, neighbor, u, n;
    while ((fscanf(fileptr, "%d\t%d\n", &user, &neighbor) != EOF)) {
        // skip connections in which either user or neighbor is seen in training
        if (user_ids.count(user) == 0 || user_ids.count(neighbor) == 0)
            continue;
        
        u = user_ids[user];
        n = user_ids[neighbor];
        
        if (!has_connection(u, n)) {
            network[u].push_back(n);
            network_spmat(n,u) = 1.0;
        }
        if (!directed && !has_connection(n, u)) {
            network[n].push_back(u);
            network_spmat(n,u) = 1.0;
        }
    }
    fclose(fileptr);
}

void Data::read_validation(string filename) {
    // read in training data
    /*FILE* fileptr = fopen(filename.c_str(), "r");

    int user, item, rating;
    set<long> dupe_checker;
    while ((fscanf(fileptr, "%d\t%d\t%d\n", &user, &item, &rating) != EOF)) {
        // look for duplicate entries; this is not a perfect check, but it's ok
        long dupe_id = item * 100000 + user * 100 +  rating;
        if (dupe_checker.count(dupe_id) != 0)
            continue;
        dupe_checker.insert(dupe_id);

        // map user and item ids
        if (user_ids.count(user) == 0) {
            user_ids[user] = user_count() - 1;
            reverse_user_ids[user_ids[user]] = user;
        }
        if (item_ids.count(item) == 0) {
            item_ids[item] = item_count() - 1;
            reverse_item_ids[item_ids[item]] = item;
        }

        if (rating != 0) {
            train_users.push_back(user_ids[user]);
            train_items.push_back(item_ids[item]);
            train_ratings.push_back(binary ? 1 : rating);
        }
    }
    fclose(fileptr);

    //ratings = sp_mat(user_count(), item_count());
    umat locations = umat(2, num_training());
    colvec values = colvec(num_training());
    user_items = new vector<int>[user_count()];
    for (int i = 0; i < num_training(); i++) {
        //ratings(train_users[i], train_items[i]) = train_ratings[i];
        locations(0,i) = train_users[i]; // row
        locations(1,i) = train_items[i]; // col
        values(i) = train_ratings[i];
        user_items[train_users[i]].push_back(train_items[i]);
    }
    ratings = sp_mat(locations, values, user_count(), item_count());*/
}

void Data::save_summary(string filename) {
    FILE* file = fopen(filename.c_str(), "w");
    
    fprintf(file, "num users:\t%d\n", user_count());
    fprintf(file, "num items:\t%d\n", item_count());
    fprintf(file, "num ratings:\t%d\t%d\n", num_training, num_validation());

    if (has_network) {
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
}

int Data::user_count() {
    return user_ids.size();
}

bool Data::has_connection(int user, int neighbor) {
    if (network_spmat(neighbor, user) == 1)
        return true;
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

int Data::item_count(int user) {
    return user_items[user].size();
}

int Data::get_item(int user, int i) {
    return user_items[user][i];
}

int Data::user_id(int user) {
    return reverse_user_ids[user];
}

int Data::item_id(int item) {
    return reverse_item_ids[item];
}

// training data
int Data::num_training() {
    return train_ratings.size();
}

int Data::get_train_user(int i) {
    return train_users[i];
}

int Data::get_train_item(int i) {
    return train_items[i];
}

int Data::get_train_rating(int i) {
    return train_ratings[i];
}

// validation data
int Data::num_validation() {
    return validation_ratings.size();
}

int Data::get_validation_user(int i) {
    return validation_users[i];
}

int Data::get_validation_item(int i) {
    return validation_items[i];
}

int Data::get_validation_rating(int i) {
    return validation_ratings[i];
}
