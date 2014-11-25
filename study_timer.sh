#!/bin/bash

if [ "$#" -ne 4 ]; then
    echo "usage: ./study_timer [data-dir] [output-dir] [K] [# iterations]"
    exit
fi


datadir=$1
outdir=$2
K=$3
I=$4

if [ ! -d $2 ]; then
    mkdir $2
fi

#seed=1412616690
#seed=11
seed=948237247

echo " * initializing study of main model (this will launch multiple processes"
echo "   that will continue living after this bash script has completed)"

time python2.7 study_pf.py $1 --out $2 --binary --K $K --seed $seed --max_iter $I
time python2.7 study_spf.py $1 --out $2 --binary --K $K --seed $seed --max_iter $I

N=`cat $1/train.tsv $1/test.tsv $1/validation.tsv | cut -f 1 | sort | uniq | wc -l`
M=`cat $1/train.tsv $1/test.tsv $1/validation.tsv | cut -f 2 | sort | uniq | wc -l`

time ~/PFvsMF/hgaprec/src/hgaprec -dir $1 -n $N -m $M -k $K -label timetest -max-iterations $I

echo " * reformatting input for baseline"
python dat/src/to_list_form.py $1

echo " * running baselines"
mkdir $2/MF
time ./ctr/ctr --directory $2/MF --user $1/users.dat --item $1/items.dat --num_factors $K --b 1 --random_seed $seed --max_iter $I

echo "all done!"
