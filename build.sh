#!/bin/bash
basn=`basename $1 .cu`

rm -vf $basn.x
nvcc -Xcompiler -Wall $1 -o $basn.x
echo "Executable: $basn.x"
