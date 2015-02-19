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
    ((time ./spf --data $1 --out $2/spf --directed --svi --K $K --seed $seed --save_freq 1000 --conv_freq 100 --min_iter 100 --max_iter 9999 --final_pass > $2/spf/out 2> $2/spf/err &) > $2/spf/time.out 2> $2/spf/time.err &)
    ((time ./spf --data $1 --out $2/pf --directed --svi --K $K --seed $seed --save_freq 1000 --conv_freq 100 --factor_only --min_iter 100 --max_iter 9999 --final_pass > $2/pf/out 2> $2/pf/err &) > $2/pf/time.out 2> $2/pf/time.err &)
    ((time ./spf --data $1 --out $2/sf --directed --svi --K $K --seed $seed --save_freq 1000 --conv_freq 100 --social_only --min_iter 100 --max_iter 9999 --final_pass > $2/sf/out 2> $2/sf/err &) > $2/sf/time.out 2> $2/sf/time.err &)
else
    # undirected
    ((time ./spf --data $1 --out $2/spf --svi --K $K --seed $seed --save_freq 1000 --conv_freq 100 --min_iter 100 --max_iter 9999 --final_pass > $2/spf/out 2> $2/spf/err &) > $2/spf/time.out 2> $2/spf/time.err &)
    ((time ./spf --data $1 --out $2/pf --svi --K $K --seed $seed --save_freq 1000 --conv_freq 100 --factor_only --min_iter 100 --max_iter 9999 --final_pass > $2/pf/out 2> $2/pf/err &) > $2/pf/time.out 2> $2/pf/time.err &)
    ((time ./spf --data $1 --out $2/sf --svi --K $K --seed $seed --save_freq 1000 --conv_freq 100 --social_only --min_iter 100 --max_iter 9999 --final_pass > $2/sf/out 2> $2/sf/err &) > $2/sf/time.out 2> $2/sf/time.err &)
fi

((./pop --data $1 --out $2/pop > $2/pop/out 2> $2/pop/err &) > $2/pop/time.out 2> $2/pop/time.err &)

echo ""
echo "*** it's okay if this script fails beyond this point ***"
echo " * trying to build code for Gaussian MF comparison"
#mkdir ctr; cd ctr; wget http://www.cs.cmu.edu/~chongw/software/ctr.tar.gz; tar -xvzf ctr.tar.gz; cd ../
#cd ctr; make; cd ../

#echo " * reformatting input for MF comparisons"
#python mkdat/to_list_form.py $1
#if [ "$directed" = "directed" ]; then
#    # directed
#    python mkdat/to_sorec_list_form.py $1
#else
#    # undirected
#    python mkdat/to_sorec_list_form.py $1 undir
#fi
#
#echo " * fitting MF comparisons"
#mkdir $2/MF
#mkdir $2/SoRec
#./ctr/ctr --directory $2/MF --user $1/users.dat --item $1/items.dat --num_factors $K --b 1 --random_seed $seed #--lambda_u 0 --lambda_v 0
#./ctr/ctr --directory $2/SoRec --user $1/users_sorec.dat --item $1/items_sorec.dat --num_factors $K --b 1 --random_seed $seed #--lambda_u 0 --lambda_v 0
#
#echo " * evaluating MF comparisons"
#make mf
#./mf --data $1 --out $2/MF --K $K
#./mf --data $1 --out $2/SoRec --K $K
#
#mv $2/SoRec $2/SoRec-ctr

echo ""
echo " * getting code for librec comparisons"
#mkdir librec; cd librec
#wget http://www.librec.net/release/librec-v1.2-rc1.zip
#unzip librec-v1.2-rc1.zip
#cd ../
make librec

echo ""
echo " * getting data ready for librec comparisons"
if [ "$directed" = "directed" ]; then
    # directed
    python mkdat/to_librec_form.py $1
else
    # undirected
    python mkdat/to_librec_form.py $1 undir
fi

echo " * fitting librec comparisons"
# config files!! (TODO)
for model in SoRec SocialMF TrustMF SoReg RSTE PMF TrustSVD MostPop BiasedMF
do
    echo $model
    echo "dataset.training.lins=$1/ratings.dat" > tmp
    echo "dataset.social.lins=$1/network.dat" >> tmp
    echo "dataset.testing.lins=$1/test.dat" >> tmp
    echo "recommender=$model" >> tmp
    echo "num.factors=$K" >> tmp
    if [ "$model" = "TrustSVD" ]; then
        echo "num.max.iter=50" >> tmp
    else
        echo "num.max.iter=100" >> tmp
    fi
    cat tmp conf/base.conf > conf/tmp.conf
    echo ""
    echo "CONF"
    head conf/tmp.conf
    echo ""
    time java -jar librec/librec.jar -c conf/tmp.conf
    mkdir $2/$model
    tail -n +2 Results/$model*prediction.txt > $2/$model/ratings.dat

    LINECOUNT=`wc -l $2/$model/ratings.dat | cut -f1 -d' '`

    if [[ $LINECOUNT != 0 ]]; then
        time ./librec_eval --data $1 --out $2/$model
    fi
done


echo "all done!"
