#!/bin/bash

if [ "$#" -ne 3 ]; then
    echo "usage: ./study [ratings-file] [network-file] [output-dir]"
    exit
fi


in_rat=$1
in_net=$2
outdir=$3
K=100

echo " * splitting user-items into train, valiation, and test"
lines=`cat $1 | wc -l`
lines=${lines[0]}
echo "$lines lines in ratings file"

if [ ! -d $3 ]; then
    mkdir $3
fi
mkdir $3/data
mkdir $3/fits

train=$((lines*8/10))
testd=$((lines*19/100))
valid=$((lines-testd-train))
validA=$((valid+train))
#echo "$train, $valid, $testd"
#echo $((train+testd+valid))

head -n $train $1 > $3/data/train.tsv
head -n $validA $1 | tail -n $valid > $3/data/validation.tsv
tail -n $testd $1 > $3/data/test.tsv
ln -s $2 $3/data/network.tsv

echo " * initializing study of main model (this will launch multiple processes"
echo "   that will continue living after this bash script has completed)"

(python2.7 study.py $3/data --out $3/fits --binary --K $K > $3/py-study.out 2> $3/py-study.err &)

echo ""
echo "*** it's okay if this script fails beyond this point ***"
echo " * trying to build code for a baseline"
cd ctr; make; cd ../

echo " * reformatting input for baseline"
python dat/src/to_list_form.py $3/data/
python dat/src/to_sorec_list_form.py $3/data/

echo " * running baselines"
mkdir $3/fits/MF
mkdir $3/fits/SoRec
./ctr/ctr --directory $3/fits/MF --user $3/data/users.dat --item $3/data/items.dat --num_factors $K
./ctr/ctr --directory $3/fits/SoRec --user $3/data/users_sorec.dat --item $3/data/items_sorec.dat --num_factors $K

echo " * evaluating baselines"
python pred-n-rank_ctr.py $3/data $3/fits/MF $K
python pred-n-rank_sorec.py $3/data $3/fits/SoRec $K
python eval.py $3/fits/MF/rankings.out $3/fits/MF
python eval.py $3/fits/SoRec/rankings.out $3/fits/SoRec

mv $3/fits/MF/rankings.out $3/data/MF_rankings.out
mv $3/fits/SoRec/rankings.out $3/data/SoRec_rankings.out

echo "all done!"
