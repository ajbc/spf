#!/bin/bash

if [ "$#" -ne 3 ]; then
    echo "usage: ./study [data-dir] [output-dir] [K]"
    exit
fi


datadir=$1
outdir=$2
K=$3

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

(./spf --data $1 --out $2/spf --binary --directed --svi --K $K --seed $seed --save_freq 1000 --conv_freq 100 --min_iter 40 --max_iter 9999 --final_pass > $2/spf/out 2> $2/spf/err &)
(./spf --data $1 --out $2/pf --binary --directed --svi --K $K --seed $seed --save_freq 1000 --conv_freq 100 --factor_only --min_iter 40 --max_iter 9999 --final_pass > $2/pf/out 2> $2/pf/err &)
(./spf --data $1 --out $2/sf --binary --directed --svi --K $K --seed $seed --save_freq 1000 --conv_freq 100 --social_only --min_iter 40 --max_iter 9999 --final_pass > $2/sf/out 2> $2/sf/err &)
(./pop --data $1 --out $2/pop > $2/pop/out 2> $2/pop/err &)
