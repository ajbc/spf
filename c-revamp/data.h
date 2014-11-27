#ifndef DATA_H
#define DATA_H

#include <string>
#include <stdio.h>
#include <map>
using namespace std;

class Data {
    private:
        bool binary;
        bool directed;
        map<int,int> user_ids;
        map<int,int> item_ids;

    public:
        Data(bool bin, bool dir);
        void read_ratings(string filename);
        void read_network(string filename);
        void read_validation(string filename);
        void save_summary(string filename);
};

#endif
