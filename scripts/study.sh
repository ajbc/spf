#!/bin/bash

if [ "$#" -ne 4 ]; then
    echo "usage: ./study [data-dir] [output-dir] [K] [directed/undirected]"
    exit
fi


datadir=$(readlink -f $1)
outdir=$(readlink -f $2)
K=$3
directed=$4
iter=100

echo "creating directory structure"
if [ -d $outdir ]; then
    rm -rf $outdir
fi
mkdir $outdir

mkdir $outdir/spf
mkdir $outdir/pf
mkdir $outdir/sf
mkdir $outdir/pop
mkdir $outdir/rand

seed=948237247

cd ../src

echo " * initializing study of main model (this will launch multiple processes"
echo "   that will continue living after this bash script has completed)"

convf=100
savef=1000
mini=100
maxi=1000

if [ "$directed" = "directed" ]; then
    # directed
    (time (./spf --data $datadir --out $outdir/spf --directed --svi --K $K --seed $seed --save_freq $savef --conv_freq $convf --min_iter $mini --max_iter $maxi --final_pass > $outdir/spf.out 2> $outdir/spf.err) > $outdir/spf.time.out 2> $outdir/spf.time.err &)
    (time (./spf --data $datadir --out $outdir/pf --directed --svi --K $K --seed $seed --save_freq $savef --conv_freq $convf --factor_only --min_iter $mini --max_iter $maxi --final_pass > $outdir/pf.out 2> $outdir/pf.err) > $outdir/pf.time.out 2> $outdir/pf.time.err &)
    (time (./spf --data $datadir --out $outdir/sf --directed --svi --K $K --seed $seed --save_freq $savef --conv_freq $convf --social_only --min_iter $mini --max_iter $maxi --final_pass > $outdir/sf.out 2> $outdir/sf.err) > $outdir/sf.time.out 2> $outdir/sf.time.err &)
else
    # undirected
    (time (./spf --data $datadir --out $outdir/spf --svi --K $K --seed $seed --save_freq $savef --conv_freq $conf --min_iter $mini --max_iter $maxi --final_pass > $outdir/spf.out 2> $outdir/spf.err) > $outdir/spf.time.out 2> $outdir/spf.time.err &)
    (time (./spf --data $datadir --out $outdir/pf --svi --K $K --seed $seed --save_freq $savef --conv_freq $conf --factor_only --min_iter $mini --max_iter $maxi --final_pass > $outdir/pf.out 2> $outdir/pf.err) > $outdir/pf.time.out 2> $outdir/pf.time.err &)
    (time (./spf --data $datadir --out $outdir/sf --svi --K $K --seed $seed --save_freq $savef --conv_freq $conf --social_only --min_iter $mini --max_iter $maxi --final_pass > $outdir/sf.out 2> $outdir/sf.err) > $outdir/sf.time.out 2> $outdir/sf.time.err &)
fi

(time (./pop --data $datadir --out $outdir/pop > $outdir/pop.out 2> $outdir/pop.err) > $outdir/pop.time.out 2> $outdir/pop.time.err &)
(time (./rand --data $datadir --out $outdir/rand > $outdir/rand.out 2> $outdir/rand.err) > $outdir/rand.time.out 2> $outdir/rand.time.err &)



echo " * reformatting input for MF comparisons"
python ../scripts/to_list_form.py $datadir
if [ "$directed" = "directed" ]; then
    # directed
    python ../scripts/to_sorec_list_form.py $datadir
else
    # undirected
    python ../scripts/to_sorec_list_form.py $datadir undir
fi

echo " * fitting MF comparisons"
mkdir $outdir/MF
mkdir $outdir/SoRec
time (./ctr/ctr --directory $outdir/MF --user $datadir/users.dat --item $datadir/items.dat --num_factors $K --b 1 --random_seed $seed) > $outdir/MF.time.out 2> $outdir.MF.time.err
time (./ctr/ctr --directory $outdir/SoRec --user $datadir/users_sorec.dat --item $datadir/items_sorec.dat --num_factors $K --b 1 --random_seed $seed) > $outdir/SoRec-ctr.time.out 2> $outdir/SoRec-ctr.time.err

echo " * evaluating MF comparisons"
make mf
time (./mf --data $datadir --out $outdir/MF --K $K) >$outdir.MF.eval.time.out 2>$outdir.MF.eval.time.err
time (./mf --data $datadir --out $outdir/SoRec --K $K) >$outdir.SoRec-ctr.eval.time.out 2>$outdir.SoRec-ctr.eval.time.err

mv $outdir/SoRec $outdir/SoRec-ctr

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

#for model in SoRec SocialMF TrustMF SoReg RSTE PMF TrustSVD BiasedMF "SVD++"
for model in SoRec SocialMF TrustMF RSTE TrustSVD
do
    rm $outdir/$model/ratings.dat
    for testidx in $(seq -f "%02g" 1 $numtest)
    do
        echo -e "$model\t(test section $testidx)"
        echo "dataset.training.lins=$datadir/ratings.dat" > tmp
        echo "dataset.social.lins=$datadir/network.dat" >> tmp
        echo "dataset.testing.lins=$datadir/test-$testidx.dat" >> tmp
        echo "recommender=$model" >> tmp
        echo "num.factors=$K" >> tmp
        echo "num.max.iter=$iter" >> tmp
        
        if [ "$model" = "TrustSVD" ]; then
            echo "val.reg.social=0.5" >> tmp
            echo "val.learn.rate=0.001" >> tmp
        else
            echo "val.reg.social=1.0" >> tmp
            echo "val.learn.rate=0.01" >> tmp
        fi
        
        cat tmp ../conf/base.conf > ../conf/tmp.conf
        echo ""
        time java -jar librec/librec.jar -c ../conf/tmp.conf 2> $outdir/$model.fit.time.err
        mkdir $outdir/$model
        tail -n +2 Results/$model*prediction.txt >> $outdir/$model/ratings.dat

    done
    
    LINECOUNT=`wc -l $outdir/$model/ratings.dat | cut -f1 -d' '`

    if [[ $LINECOUNT != 0 ]]; then
        time ./librec_eval --data $datadir --out $outdir/$model 2> $outdir/$model.eval.time.err
    fi
done

echo "all done!"
