
WIDTH=1024
HEIGHT=1024
NUM_RUNS=1
BORDER_TYPE=1
BORDER_BLOCKS=0

echo ""
./gpufilter/build/src/Debug/alg6_5 $WIDTH $HEIGHT $NUM_RUNS $BORDER_TYPE $BORDER_BLOCKS
echo ""
./gpufilter/build/src/Debug/sat $WIDTH $HEIGHT $NUM_RUNS $BORDER_TYPE $BORDER_BLOCKS