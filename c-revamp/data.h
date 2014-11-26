#ifndef DATA_H
#define DATA_H

#include <string>
using namespace std;

class Data {
    private:
        bool binary;
        bool directed;

    public:
        Data(bool bin, bool dir);
        void read_ratings(string filename);
        void read_network(string filename);
        void read_validation(string filename);
        void save_summary(string filename);
};


// dataset class
//
// TODO:
//

    /*fout = open(join(args.out_dir, "data_stats.dat"), 'w+')
    fout.write("num users:\t%d\n" % len(data.users))
    fout.write("num items:\t%d\n" % len(data.items))
    fout.write("num ratings:\t%d\t%d\t%d\n" % \
        (data.rating_count[0], data.rating_count[1], data.rating_count[2]))
    fout.write("network connections:\t%d\n" % data.connection_count)
    fout.close()*/
    /*void save(string filename) {
        FILE * file = fopen(filename.c_str(), "w");
        
        fprintf(file, "data directory: %s\n", datadir.c_str());

        fprintf(file, "\nmodel specification:\n");
        if (social_only) {
            fprintf(file, "\tsocial factorization (SF)   [ social factors only ]\n");
        } else if (factor_only) {
            fprintf(file, "\tPoisson factorization (PF)   [ general preference factors only ]\n");
        } else {
            fprintf(file, "\tsocial Poisson factorization (SPF)\n");
        }
        if (!social_only) {
            fprintf(file, "\tK = %d   (number of latent factors for general preferences)\n", k);
        }

    
        fprintf(file, "\nshape and rate hyperparameters:\n");
        if (!social_only) {
            fprintf(file, "\ttheta (%f, %f)\n", a_theta, b_theta);
            fprintf(file, "\tbeta  (%f, %f)\n", a_beta, b_beta);
        }
        if (!factor_only) {
            fprintf(file, "\ttau   (%f, %f)\n", a_tau, b_tau);
        }
        

        fprintf(file, "\ndata attributes:\n");
        
        if (binary) {
            fprintf(file, "\tbinary ratings\n");
        } else {
            fprintf(file, "\tinteger ratings\n");
        }
        
        if (!factor_only) {
            if (directed) {
                fprintf(file, "\tdirected network\n");
            } else {
                fprintf(file, "\tundirected network\n");
            }
        }

        
        fprintf(file, "\ninference parameters:\n");
        fprintf(file, "\tseed:                         %d\n", (int)seed);
        fprintf(file, "\tsave lag:                     %d\n", save_lag);
        fprintf(file, "\tmaximum number of iterations: %d\n", max_iter);
    
        fclose(file);
    }*/
#endif
