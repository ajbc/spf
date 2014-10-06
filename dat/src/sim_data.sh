data=$1
thresh=$2

for ((i=0; i <= 100 ; i=i+10)); do
    mkdir $1/amp/$i
    echo "(de)amp $i"
    python adjust_amplification.py $1 $1/amp/$i $i
    #if [ $i -lt $thresh ]; then
    #    echo "deamp $i"
    #    python deamplify_data.py $1 $1/amp/$i $i
    #else
    #    echo "amp $i"
    #    python amplify_data.py $1 $1/amp/$i $i
    #fi 
    ln -s $1/network.tsv $1/amp/$i/network.tsv
done
