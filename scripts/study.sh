#!/bin/bash

if [ "$#" -ne 4 ]; then
    echo "usage: ./study [data-dir] [output-dir] [K] [directed/undirected]"
    exit
fi


datadir=$(readlink -f $1)
outdir=$(readlink -f $2)
K=$3
directed=$4

echo "creating directory structure"
if [ -d $outdir ]; then
    rm -rf $outdir
fi
mkdir $outdir

mkdir $outdir/spf
mkdir $outdir/pf
mkdir $outdir/sf
mkdir $outdir/pop

seed=948237247

cd ../src

echo " * initializing study of main model (this will launch multiple processes"
echo "   that will continue living after this bash script has completed)"

if [ "$directed" = "directed" ]; then
    # directed
    (./spf --data $datadir --out $outdir/spf --directed --bias --svi --K $K --seed $seed --save_freq 1000 --conv_freq 100 --min_iter 2000 --max_iter 2000 --final_pass > $outdir/spf.out 2> $outdir/spf.err &)
    (./spf --data $datadir --out $outdir/pf --directed --bias --svi --K $K --seed $seed --save_freq 1000 --conv_freq 100 --factor_only --min_iter 2000 --max_iter 2000 --final_pass > $outdir/pf.out 2> $outdir/pf.err &)
    (./spf --data $datadir --out $outdir/sf --directed --bias --svi --K $K --seed $seed --save_freq 1000 --conv_freq 100 --social_only --min_iter 2000 --max_iter 2000 --final_pass > $outdir/sf.out 2> $outdir/sf.err &)
else
    # undirected
    (./spf --data $datadir --out $outdir/spf --bias --svi --K $K --seed $seed --save_freq 1000 --conv_freq 100 --min_iter 2000 --max_iter 2000 --final_pass > $outdir/spf.out 2> $outdir/spf.err &)
    (./spf --data $datadir --out $outdir/pf --bias --svi --K $K --seed $seed --save_freq 1000 --conv_freq 100 --factor_only --min_iter 2000 --max_iter 2000 --final_pass > $outdir/pf.out 2> $outdir/pf.err &)
    (./spf --data $datadir --out $outdir/sf --bias --svi --K $K --seed $seed --save_freq 1000 --conv_freq 100 --social_only --min_iter 2000 --max_iter 2000 --final_pass > $outdir/sf.out 2> $outdir/sf.err &)
fi

(./pop --data $datadir --out $outdir/pop > $outdir/pop.out 2> $outdir/pop.err &)

#echo " * reformatting input for MF comparisons"
#python mkdat/to_list_form.py $datadir
#if [ "$directed" = "directed" ]; then
#    # directed
#    python mkdat/to_sorec_list_form.py $datadir
#else
#    # undirected
#    python mkdat/to_sorec_list_form.py $datadir undir
#fi
#
#echo " * fitting MF comparisons"
#mkdir $outdir/MF
#mkdir $outdir/SoRec
#./ctr/ctr --directory $outdir/MF --user $datadir/users.dat --item $datadir/items.dat --num_factors $K --b 1 --random_seed $seed #--lambda_u 0 --lambda_v 0
#./ctr/ctr --directory $outdir/SoRec --user $datadir/users_sorec.dat --item $datadir/items_sorec.dat --num_factors $K --b 1 --random_seed $seed #--lambda_u 0 --lambda_v 0
#
#echo " * evaluating MF comparisons"
#make mf
#./mf --data $datadir --out $outdir/MF --K $K
#./mf --data $datadir --out $outdir/SoRec --K $K
#
#mv $outdir/SoRec $outdir/SoRec-ctr

echo ""

echo ""
echo " * getting data ready for librec comparisons"
if [ "$directed" = "directed" ]; then
    # directed
    python ../scripts/to_librec_form.py $datadir
else
    # undirected
    python ../scripts/to_librec_form.py $datadir undir
fi

echo " * fitting librec comparisons"
# config files!! (TODO)
for model in SoRec SocialMF TrustMF SoReg RSTE PMF TrustSVD MostPop BiasedMF
do
    echo $model
    echo "dataset.training.lins=$datadir/ratings.dat" > tmp
    echo "dataset.social.lins=$datadir/network.dat" >> tmp
    echo "dataset.testing.lins=$datadir/test.dat" >> tmp
    echo "recommender=$model" >> tmp
    echo "num.factors=$K" >> tmp
    if [ "$model" = "TrustSVD" ]; then
        echo "num.max.iter=50" >> tmp
    else
        echo "num.max.iter=100" >> tmp
    fi
    cat tmp ../conf/base.conf > ../conf/tmp.conf
    echo ""
    echo "CONF"
    head ../conf/tmp.conf
    echo ""
    time java -jar librec/librec.jar -c ../conf/tmp.conf
    mkdir $outdir/$model
    tail -n +2 Results/$model*prediction.txt > $outdir/$model/ratings.dat

    LINECOUNT=`wc -l $outdir/$model/ratings.dat | cut -f1 -d' '`

    if [[ $LINECOUNT != 0 ]]; then
        time ./librec_eval --data $datadir --out $outdir/$model
    fi
done


echo "all done!"
