#!/bin/bash

if [ "$#" -ne 3 ]; then
    echo "usage: ./study [data-dir] [output-dir] [K]"
    exit
fi


datadir=$1
outdir=$2
K=$3

if [ ! -d $2 ]; then
    mkdir $2
fi

#seed=1412616690
#seed=11
seed=948237247

echo " * initializing study of main model (this will launch multiple processes"
echo "   that will continue living after this bash script has completed)"

#(nice -n 10 python2.7 study.py $1 --out $2 --K $K --seed $seed > $2/py-study.out 2> $2/py-study.err &)
(python2.7 study.py $1 --out $2 --binary --K $K --seed $seed > $2/py-study.out 2> $2/py-study.err &)
#(python2.7 study.py $1 --out $2 --binary --K $K --seed $seed --SVI > $2/py-study.out 2> $2/py-study.err &)

echo ""
echo "*** it's okay if this script fails beyond this point ***"
echo " * trying to build code for a baseline"
#cd ctr; make; cd ../

echo " * reformatting input for baseline"
python dat/src/to_list_form.py $1
python dat/src/to_sorec_list_form.py $1

echo " * running baselines"
mkdir $2/MF
mkdir $2/SoRec
./ctr/ctr --directory $2/MF --user $1/users.dat --item $1/items.dat --num_factors $K --b 1 --random_seed $seed #--lambda_u 0 --lambda_v 0
./ctr/ctr --directory $2/SoRec --user $1/users_sorec.dat --item $1/items_sorec.dat --num_factors $K --b 1 --random_seed $seed #--lambda_u 0 --lambda_v 0

echo " * evaluating baselines"
python pred-n-rank_ctr.py $1 $2/MF $K
python pred-n-rank_sorec.py $1 $2/SoRec $K
python eval.py $2/MF/rankings.out $2/MF
python eval.py $2/SoRec/rankings.out $2/SoRec

echo "all done!"
