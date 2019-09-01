#!/bin/bash

if [ "$(uname)" == "Linux" ]; then
#    ../recursivefilter-build/test_recursivefilter_checkmath
    ../recursivefilter-build/test_recursivefilter_measuretime
else
    ./bin/test_recursivefilter
fi

printf "\n\nRunning authors' implementation...\n"

WIDTH=1024
HEIGHT=1024
NUM_RUNS=2500
BORDER_TYPE=1
BORDER_BLOCKS=0
if [ "$(uname)" == "Linux" ]; then
    # Usage: [width height run-times border-type border-blocks]
    printf "\nsat: "
    gpufilter-build/src/sat $WIDTH $HEIGHT $NUM_RUNS $BORDER_TYPE $BORDER_BLOCKS
    #gpufilter-build/src/alg6_1
    #gpufilter-build/src/alg6_2
    #gpufilter-build/src/alg6_3
    #gpufilter-build/src/alg6_4
    printf "\nalg6_5:\n"
    gpufilter-build/src/alg6_5 $WIDTH $HEIGHT 1 $BORDER_TYPE $BORDER_BLOCKS
else
    ./bin/test_recursivefilter
fi