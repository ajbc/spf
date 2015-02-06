#include <iostream>
#define ARMA_64BIT_WORD
#include <armadillo>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_sf_psi.h>
#include <list>

#include "utils.h"

using namespace std;
using namespace arma;
#include "data.h"

struct model_settings {
    bool verbose;

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
    int    save_freq;
    int    eval_freq;
    int    conv_freq;
    int    max_iter;
    int    min_iter;
    double likelihood_delta;

    bool svi;
    bool final_pass;
    int    sample_size;
    double delay;
    double forget;

    int k;
  
    
    void set(bool print, string out, string data, bool use_svi,
             double athe, double bthe, double abet, double bbet, 
             double atau, double btau,
             bool social, bool factor, bool bin, bool dir,
             long rand, int savef, int evalf, int convf, 
             int iter_max, int iter_min, double delta,
             bool finalpass, int sample, double svi_delay, double svi_forget,
             int num_factors) {
        verbose = print;

        outdir = out;
        datadir = data;

        svi = use_svi;
        
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
        save_freq = savef;
        eval_freq = evalf;
        conv_freq = convf;
        max_iter = iter_max;
        min_iter = iter_min;
        likelihood_delta = delta;

        final_pass = finalpass;
        sample_size = sample;
        delay = svi_delay;
        forget = svi_forget;

        k = num_factors;
    }
    
    void set_stochastic_inference(bool setting) {
        svi = setting;
    }
    
    void set_sample_size(int setting) {
        sample_size = setting;
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
        fprintf(file, "\tsave frequency:                           %d\n", save_freq);
        fprintf(file, "\tevaluation frequency:                     %d\n", eval_freq);
        fprintf(file, "\tconvergence check frequency:              %d\n", conv_freq);
        fprintf(file, "\tmaximum number of iterations:             %d\n", max_iter);
        fprintf(file, "\tminimum number of iterations:             %d\n", min_iter);
        fprintf(file, "\tchange in log likelihood for convergence: %f\n", likelihood_delta);
        fprintf(file, "\tdo a final pass after convergence:        %s\n", final_pass ? "true" : "false");
        
        if (svi) {
            fprintf(file, "\nStochastic variational inference parameters\n");
            fprintf(file, "\tsample size:                              %d\n", sample_size);
            fprintf(file, "\tSVI delay (tau):                          %f\n", delay);
            fprintf(file, "\tSVI forgetting rate (kappa):              %f\n", forget);
        } else {
            fprintf(file, "\nusing batch variational inference\n");
        }
    
        fclose(file);
    }
};

class SPF {
    private:
        model_settings* settings;
        Data* data;
       
        // model parameters
        sp_fmat tau; // user influence
        sp_fmat logtau; // fake "log" user influence
                       // it's really exp(E[log(tau)]) which != E[tau]
        fmat theta;  // user preferences
        fmat beta;   // item attributes
        fmat logtheta;  // log variant of above
        fmat logbeta;   // ditto

        // helper parameters
        sp_fmat a_tau;
        sp_fmat b_tau;
        fmat a_theta;
        fmat b_theta;
        fmat a_beta;
        sp_fmat a_beta_user;
        fmat a_beta_old;
        fmat b_beta;
    
        // random number generator
        gsl_rng* rand_gen;

        void initialize_parameters();
        void reset_helper_params();
        void save_parameters(string label);
    
        // parameter updates
        void update_shape(int user, int item, int rating);
        double update_tau(int user);
        double update_theta(int user);
        void update_beta(int item);

        double get_ave_log_likelihood();
        void log_convergence(int iteration, double ave_ll, double delta_ll);
        void log_time(int iteration, double duration);
        void log_params(int iteration, double tau_change, double theta_change);
        void log_user(FILE* file, int user, int heldout, double rmse, 
            double mae, double rank, int first, double crr, double ncrr,
            double ndcg);
    
        // define how to scale updates (training / sample size) (for SVI)
        double scale; 
        
        // counts of number of times an item has been seen in a sample (for SVI)
        map<int,int> iter_count;

        void evaluate(string label);
        void evaluate(string label, bool write_rankings);

        
    public:
        SPF(model_settings* model_set, Data* dataset);
        void learn();
        double predict(int user, int item);
        void evaluate();

};
