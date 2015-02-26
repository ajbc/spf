#include <getopt.h>
#include <stdio.h>
#include <list>
#include <gsl/gsl_rng.h>
//#include "utils.h"
#include "data.h"
#include "model.h"

// helper function to write out per-user info
void log_user(FILE* file, Data *data, int user, int heldout, double rmse, double mae,
    double rank, int first, double crr, double ncrr, double ndcg, bool stats);

void log_item(FILE* file, Data *data, int item, int heldout, double rmse, double mae,
    double rank, int first, double crr, double ncrr, double ndcg, bool stats);

// helper function to sort predictions properly
bool prediction_compare(const pair<double,int>& itemA, 
    const pair<double, int>& itemB);

// take a prediction function as an argument
void eval(Model* model, double (Model::*prediction)(int,int), string outdir, Data* data, bool stats, unsigned long int seed, bool verbose);
