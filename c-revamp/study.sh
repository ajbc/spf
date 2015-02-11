#!/bin/bash

if [ "$#" -ne 4 ]; then
    echo "usage: ./study [data-dir] [output-dir] [K] [directed/undirected]"
    exit
fi


datadir=$1
outdir=$2
K=$3
directed=$4

echo "creating directory structure"
if [ -d $2 ]; then
    rm -rf $2
fi
mkdir $2

mkdir $2/spf
mkdir $2/pf
mkdir $2/sf
mkdir $2/pop

seed=948237247

echo "compiling model + popularity baseline"
make clean
make
make pop

echo " * initializing study of main model (this will launch multiple processes"
echo "   that will continue living after this bash script has completed)"

if [ "$directed" = "directed" ]; then
    # directed
    (./spf --data $1 --out $2/spf --binary --directed --svi --K $K --seed $seed --save_freq 1000 --conv_freq 100 --min_iter 100 --max_iter 9999 --final_pass > $2/spf/out 2> $2/spf/err &)
    (./spf --data $1 --out $2/pf --binary --directed --svi --K $K --seed $seed --save_freq 1000 --conv_freq 100 --factor_only --min_iter 100 --max_iter 9999 --final_pass > $2/pf/out 2> $2/pf/err &)
    (./spf --data $1 --out $2/sf --binary --directed --svi --K $K --seed $seed --save_freq 1000 --conv_freq 100 --social_only --min_iter 100 --max_iter 9999 --final_pass > $2/sf/out 2> $2/sf/err &)
else
    # undirected
    (./spf --data $1 --out $2/spf --binary --svi --K $K --seed $seed --save_freq 1000 --conv_freq 100 --min_iter 100 --max_iter 9999 --final_pass > $2/spf/out 2> $2/spf/err &)
    (./spf --data $1 --out $2/pf --binary --svi --K $K --seed $seed --save_freq 1000 --conv_freq 100 --factor_only --min_iter 100 --max_iter 9999 --final_pass > $2/pf/out 2> $2/pf/err &)
    (./spf --data $1 --out $2/sf --binary --svi --K $K --seed $seed --save_freq 1000 --conv_freq 100 --social_only --min_iter 100 --max_iter 9999 --final_pass > $2/sf/out 2> $2/sf/err &)
fi

(./pop --data $1 --out $2/pop > $2/pop/out 2> $2/pop/err &)

echo ""
echo "*** it's okay if this script fails beyond this point ***"
echo " * trying to build code for Gaussian MF comparison"
#mkdir ctr; cd ctr; wget http://www.cs.cmu.edu/~chongw/software/ctr.tar.gz; tar -xvzf ctr.tar.gz; cd ../
#cd ctr; make; cd ../

echo " * reformatting input for MF comparisons"
python mkdat/to_list_form.py $1
if [ "$directed" = "directed" ]; then
    # directed
    python mkdat/to_sorec_list_form.py $1
else
    # undirected
    python mkdat/to_sorec_list_form.py $1 undir
fi

echo " * fitting MF comparisons"
mkdir $2/MF
mkdir $2/SoRec
./ctr/ctr --directory $2/MF --user $1/users.dat --item $1/items.dat --num_factors $K --b 1 --random_seed $seed #--lambda_u 0 --lambda_v 0
./ctr/ctr --directory $2/SoRec --user $1/users_sorec.dat --item $1/items_sorec.dat --num_factors $K --b 1 --random_seed $seed #--lambda_u 0 --lambda_v 0

echo " * evaluating MF comparisons"
make mf
./mf --data $1 --out $2/MF --K $K
./mf --data $1 --out $2/SoRec --K $K


echo "\n * getting code for librec comparisons"
#mkdir librec; cd librec
#wget http://www.librec.net/release/librec-v1.2-rc1.zip
#unzip librec-v1.2-rc1.zip
#cd ../

jar -cf librec.jar .


echo "all done!"
