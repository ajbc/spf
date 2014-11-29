#ifndef DATA_H
#define DATA_H

#include <string>
#include <stdio.h>
#include <map>
#include <vector>

using namespace std;

class Data {
    private:
        bool binary;
        bool directed;
        map<int,int> user_ids;
        map<int,int> item_ids;
        map<int,int> reverse_user_ids;
        map<int,int> reverse_item_ids;

        vector<int>* network;
        
        vector<int> train_users;
        vector<int> train_items;
        vector<int> train_ratings;

    public:
        Data(bool bin, bool dir);
        void read_ratings(string filename);
        void read_network(string filename);
        void read_validation(string filename);
        void save_summary(string filename);

        int user_count();
        int item_count();

        int neighbor_count(int user);
        int get_neighbor(int user, int n);

        bool has_connection(int user, int neighbor);

        int user_id(int user);
        int item_id(int item);

        int num_training();

        int get_train_user(int i);
        int get_train_item(int i);
        int get_train_rating(int i);
};

#endif
