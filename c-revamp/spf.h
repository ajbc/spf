#include <iostream>
#include <armadillo>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_sf_psi.h>
#include <list>

#include "utils.h"

using namespace std;
using namespace arma;
#include "data.h"

struct model_settings {
    string outdir;
    string datadir;

    double a_theta;
    double b_theta;
    double a_beta;
    double b_beta;
    double a_tau;
    double b_tau;

    bool social_only;
    bool factor_only;
    bool binary;
    bool directed;
    
    long   seed;
    int    save_lag;
    int    max_iter;
    int    min_iter;
    double likelihood_delta;

    int k;
  
    
    void set(string out, string data, 
             double athe, double bthe, double abet, double bbet, 
             double atau, double btau,
             bool social, bool factor, bool bin, bool dir,
             long rand, int lag, int iter_max, int iter_min, double delta,
             int num_factors) {
        outdir = out;
        datadir = data;
        
        a_theta = athe;
        b_theta = bthe;
        a_beta  = abet;
        b_beta  = bbet;
        a_tau   = atau;
        b_tau   = btau;

        social_only = social;
        factor_only = factor;
        binary = bin;
        directed = dir;

        seed = rand;
        save_lag = lag;
        max_iter = iter_max;
        min_iter = iter_min;
        likelihood_delta = delta;

        k = num_factors;
    }

    void save(string filename) {
        FILE* file = fopen(filename.c_str(), "w");
        
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
        fprintf(file, "\tseed:                                     %d\n", (int)seed);
        fprintf(file, "\tsave lag:                                 %d\n", save_lag);
        fprintf(file, "\tmaximum number of iterations:             %d\n", max_iter);
        fprintf(file, "\tminimum number of iterations:             %d\n", min_iter);
        fprintf(file, "\tchange in log likelihood for convergence: %f\n", likelihood_delta);
    
        fclose(file);
    }
};

class SPF {
    private:
        model_settings* settings;
        Data* data;
       
        // model parameters
        sp_mat tau; // user influence
        sp_mat logtau; // fake "log" user influence
                       // it's really exp(E[log(tau)]) which != E[tau]
        mat theta;  // user preferences
        mat beta;   // item attributes
        mat logtheta;  // log variant of above
        mat logbeta;   // ditto

        // helper parameters
        sp_mat a_tau;
        sp_mat b_tau;
        mat a_theta;
        mat b_theta;
        mat a_beta;
        mat b_beta;
    
        // random number generator
        gsl_rng* rand_gen;

        void initialize_parameters();
        void reset_helper_params();
        void save_parameters(string label);
    
        // parameter updates
        void update_shape(int user, int item, int rating);
        void update_MF();
        void update_SF();

        double get_ave_log_likelihood();
        void log_convergence(int iteration, double ave_ll, double delta_ll);
        void log_user(FILE* file, int user, int heldout, double rmse, 
            double mae, double rank, int first, double crr, double ncrr,
            double ndcg);

        
    public:
        SPF(model_settings* model_set, Data* dataset);
        void learn();
        double predict(int user, int item);
        void evaluate();
};
