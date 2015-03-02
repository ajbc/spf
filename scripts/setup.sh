#!/bin/bash

if [ "$#" -ne 0 ]; then
    echo "usage: ./setup.sh"
    exit
fi

echo "compiling SPF and popularity baseline code"
cd ../src
make clean
make
make pop

echo "setting up Chong's MF code (from CTR)"
echo "  getting src"
rm -rf ctr
mkdir ctr
cd ctr
wget http://www.cs.cmu.edu/~chongw/software/ctr.tar.gz
tar -xvzf ctr.tar.gz
cd ../

echo "  compiling"
cd ctr; make; cd ../
make mf

echo ""
echo "downloading LibRec"
rm -rf librec
mkdir librec; cd librec
wget http://www.librec.net/release/librec-v1.2.zip
unzip librec-v1.2.zip
cd ../

echo "compiling LibRec eval code"
make librec_eval

echo "all done!"
