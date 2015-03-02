#!/bin/bash

if [ "$#" -ne 0 ]; then
    echo "usage: ./setup.sh"
    exit
fi

cd ../src

echo "setting up Chong's MF code (from CTR)"
echo "  getting src"
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
mkdir librec; cd librec
wget http://www.librec.net/release/librec-v1.2.zip
unzip librec-v1.2-rc1.zip
cd ../

echo "compiling LibRec eval code"
make librec

echo "all done!"
