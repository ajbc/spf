#ifndef DATA_H
#define DATA_H


#include <string>
#include <stdio.h>
#include <map>
#include <vector>
#include <set>

#define ARMA_64BIT_WORD
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

        map<int,int> item_popularity;
        
        // training data
        vector<int> train_users;
        vector<int> train_items;
        vector<int> train_ratings;

        // validation data
        vector<int> validation_users;
        vector<int> validation_items;
        vector<int> validation_ratings;
        sp_fmat validation_ratings_matrix;
        
        // test data
        map<int,int> test_count;
        map<int,int> test_count_item;

        // for use in initializing the network data structures only
        bool has_connection_init(int user, int neighbor);

        // simple summaries
        float mean_rating;
        map<int,float> item_ave_ratings;
        map<int,float> user_ave_ratings;

    public:
        sp_fmat ratings;
        sp_fmat network_spmat;
        
        Data(bool bin, bool dir);
        void read_ratings(string filename);
        void read_network(string filename);
        void read_validation(string filename);
        void read_test(string filename);
        void save_summary(string filename);

        int user_count();
        int item_count();

        int neighbor_count(int user);
        int get_neighbor(int user, int n);

        int connectivity(int user);
        
        int item_count(int user);
        int get_item(int user, int i);

        bool has_connection(int user, int neighbor);

        int user_id(int user);
        int item_id(int item);

        int popularity(int item);
        float ave_rating();
        float item_ave_rating(int item);
        float user_ave_rating(int user);
    
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
        bool in_validation(int user, int item);
        
        // test data
        set<int> test_users;
        set<int> test_items;
        sp_umat test_ratings;
        int num_test();
        int num_test(int user);
        int num_test_item(int item);
};

#endif
