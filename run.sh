#!/bin/bash

printf "\n\nRunning my implementation...\n"
MY_RUNTIME_MS=-1.0
if [ "$(uname)" == "Linux" ]; then
    ../recursivefilter-build/test_recursivefilter_checkmath
    ../recursivefilter-build/test_recursivefilter_filterimages
    MY_RUNTIME_OUTPUT="$(../recursivefilter-build/test_recursivefilter_measuretime)"
    echo "$MY_RUNTIME_OUTPUT"
    MY_RUNTIME_MS=$(echo "$MY_RUNTIME_OUTPUT" | grep "MY_RUNTIME_MS " | sed 's/MY_RUNTIME_MS //g')
else
    ./build/Release/test_recursivefilter_checkmath
    ./build/Release/test_recursivefilter_filterimages
    MY_RUNTIME_OUTPUT="$(./build/Release/test_recursivefilter_measuretime)"
    echo "$MY_RUNTIME_OUTPUT"
    MY_RUNTIME_MS=$(echo "$MY_RUNTIME_OUTPUT" | grep "MY_RUNTIME_MS " | sed 's/MY_RUNTIME_MS //g')
fi
printf "\nRuntime of my implementation: %s ms\n" $MY_RUNTIME_MS

if [ "$(uname)" == "Linux" ]; then
    WIDTH=1024
    HEIGHT=1024
    NUM_RUNS=1  # Does not print runtime with more than 1 runs
    BORDER_TYPE=0
    BORDER_BLOCKS=0
    AUTHORS_RUNTIME_MS=-1.0
    printf "\n\nRunning Authors' implementation..."
    # Usage: [width height run-times border-type border-blocks]
    AUTHORS_OUTPUT=$(gpufilter-build/src/sat $WIDTH $HEIGHT $NUM_RUNS $BORDER_TYPE $BORDER_BLOCKS | grep "sat_gpu:")
    #printf "%s" $AUTHORS_OUTPUT
    AUTHORS_RUNTIME_S=$(echo $AUTHORS_OUTPUT | grep "sat_gpu: " | sed 's/\[\([^]]*\)\] 100% - sat_gpu: //g' | sed 's# s - [0-9]*.[0-9]* GiP/s##g')
    AUTHORS_RUNTIME_MS=$(echo "scale=6; $AUTHORS_RUNTIME_S*1000.0" | bc)
    printf "\nRuntime of Authors' implementation: %0.06f ms\n" $AUTHORS_RUNTIME_MS
    MY_PER_AUTHORS=$(echo "scale=6; $MY_RUNTIME_MS/$AUTHORS_RUNTIME_MS" | bc)
    MY_PER_AUTHORS=$(echo "scale=6; $MY_PER_AUTHORS*100" | bc)
    printf "\nMy runtime as percentage of Authors' runtime: %.02f%% (compared on table size %dx%d)\n\n" $MY_PER_AUTHORS $WIDTH $HEIGHT
else
    echo "Authors' implementation is only available on Linux."
fi