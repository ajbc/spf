#ifndef DATA_H
#define DATA_H

#include <string>
#include <stdio.h>
#include <map>
#include <vector>
#include <set>
#include <armadillo>

using namespace std;
using namespace arma;

class Data {
    private:
        bool binary;
        bool directed;
        bool has_network;
        
        map<int,int> user_ids;
        map<int,int> item_ids;
        map<int,int> reverse_user_ids;
        map<int,int> reverse_item_ids;

        vector<int>* network;
        vector<int>* user_items;
        
        // training data
        vector<int> train_users;
        vector<int> train_items;
        vector<int> train_ratings;

        // validation data
        vector<int> validation_users;
        vector<int> validation_items;
        vector<int> validation_ratings;

        // for use in initializing the network data structures only
        bool has_connection_init(int user, int neighbor);


    public:
        sp_mat ratings; //TODO: should these be sp_umat types?
        sp_mat network_spmat;
        
        Data(bool bin, bool dir);
        void read_ratings(string filename);
        void read_network(string filename);
        void read_validation(string filename);
        void save_summary(string filename);

        int user_count();
        int item_count();

        int neighbor_count(int user);
        int get_neighbor(int user, int n);
        
        int item_count(int user);
        int get_item(int user, int i);

        bool has_connection(int user, int neighbor);

        int user_id(int user);
        int item_id(int item);
    
        // training data
        int num_training();
        int get_train_user(int i);
        int get_train_item(int i);
        int get_train_rating(int i);
        
        // validation data
        int num_validation();
        int get_validation_user(int i);
        int get_validation_item(int i);
        int get_validation_rating(int i);
};

#endif
