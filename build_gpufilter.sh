PROJECT_NAME=recursivefilter
GPUFILTER_DIR=gpufilter

if [[ "$OSTYPE" == "msys" ]]; then
    GENERATE_PROJECT_TYPE="Visual Studio 15 2017 Win64"
    BUILD_PATH=${GPUFILTER_DIR}/build
else
    GENERATE_PROJECT_TYPE="Eclipse CDT4 - Unix Makefiles"
    BUILD_PATH=../${PROJECT_NAME}/${GPUFILTER_DIR}-build
fi
echo "Generating \"${GENERATE_PROJECT_TYPE}\""
echo "Build path: ${BUILD_PATH}"

export MAKEFLAGS=-j8
cmake -H${GPUFILTER_DIR} -B${BUILD_PATH} \
    -G"${GENERATE_PROJECT_TYPE}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=${BUILD_PATH}
#make
cmake --build ${BUILD_PATH} --target install
