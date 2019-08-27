
WIDTH=1024
HEIGHT=1024
NUM_RUNS=1
BORDER_TYPE=1
BORDER_BLOCKS=0

if [ "$(uname)" == "Linux" ]; then
    gpufilter-build/src/sat
    #gpufilter-build/src/alg6_1
    #gpufilter-build/src/alg6_2
    #gpufilter-build/src/alg6_3
    #gpufilter-build/src/alg6_4
    gpufilter-build/src/alg6_5
else
    ./bin/test_recursivefilter
fi
