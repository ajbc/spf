# Social Poisson Factorization (SPF)

#### Repository Contents
- `src` C++ source code
- `scripts` bash and python scripts for data processing and running experiments


## Data
The input format for data is tab-separted files with integer values:
```
user id    item id    rating
```
The ratings should be separated into training, testing, and validation data; `scripts/process_data.py` 
helps divide data into these different sets.  This script also culls the user network such that only 
connections that have at least one item in common are included.
```
python process_data.py [ratings-file] [network-file] [output-dir]
```


## Running SPF
1. Clone the repo:
    `git clone https://github.com/ajbc/spf.git`
2. Navigate to the `spf/src` directory
3. Compile with `make`
4. Run the executable, e.g.:
    `./spf --data ~/my-data/ --out my-fit`

#### SPF Options
|Option|Arguments|Help|Default|
|---|---|---|---|
|--help||print help information||
|--verbose||print extra information while running|off|
|--out|dir|save directory, required||
|--data|dir|data directory, required||
|--svi||use stochastic VI (instead of batch VI)|off for < 10M ratings in training|
|--batch||use batch VI (instead of SVI)|on for < 10M ratings in training|
|--a_theta|a|shape hyperparamter to theta (user preferences)|0.3|
|--b_theta|b|rate hyperparamter to theta (user preferences)|0.3|
|--a_beta|a|shape hyperparamter to beta (item attributes)|0.3|
|--b_beta|b|rate hyperparamter to beta (item attributes)|0.3|
|--a_tau|a|shape hyperparamter to tau (user influence)|2|
|--b_tau|b|rate hyperparamter to tau (user influence)|5|
|--a_delta|a|shape hyperparamter to delta (item bias)|0.3|
|--b_delta|b|rate hyperparamter to delta (item bias)|0.3|
|--social-only||only consider social aspect of factorization (SF)|include factors|
|--factor-only||only consider general factors (no social; PF)|include social|
|--bias||include a bias term for each item|no bias|
|--binary||assume ratings are binary|integer|
|--directed||assume network is directed|undirected|
|--seed|seed|the random seed|time|
|--save_freq|f|the saving frequency.  Negative value means no savings for intermediate results.|20|
|--eval_freq|f|the intermediate evaluating frequency. Negative means no evaluation for intermediate results.|-1|
|--conv_freq|f|the convergence check frequency|10|
|--max_iter|max|the max number of iterations|300|
|--min_iter|min|the min number of iterations|30|
|--converge|c|the change in rating log likelihood required for convergence|1e-6|
|--final_pass||do a final pass on all users and items|no final pass|
|--sample|sample_size|the stochastic sample size|1000|
|--svi_delay|tau|SVI delay >= 0 to down-weight early samples|1024|
|--svi_forget|kappa|SVI forgetting rate (0.5,1]|default 0.75|
|--K|K|the number of general factors|100|


## Running an Experiment
1. Download and compile code for comparison models:
    `cd scripts/; ./setup.sh; cd ..`
2. Kick off fits for multiple models with the script:
    `./study [data-dir] [output-dir] [K] [directed/undirected]`
