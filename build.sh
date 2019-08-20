PROJECT_NAME=gpuacademy_recursivefilter

DEPS_DIR="C:/work/vs2017-x64"
if [[ "$OSTYPE" == "msys" ]]; then
    GENERATE_PROJECT_TYPE="Visual Studio 15 2017 Win64"
    BUILD_PATH="build"
else
    GENERATE_PROJECT_TYPE="Eclipse CDT4 - Unix Makefiles"
    BUILD_PATH="../${PROJECT_NAME}-build"
fi
echo "Generating \"${GENERATE_PROJECT_TYPE}\""
echo "Build path: ${BUILD_PATH}"

export MAKEFLAGS=-j6
cmake -H. -B${BUILD_PATH} \
    -DCMAKE_CXX_STANDARD=11 \
    -G"${GENERATE_PROJECT_TYPE}" \
    -DCMAKE_PREFIX_PATH="${DEPS_DIR}" \
    -DCMAKE_BUILD_TYPE=Debug
cmake --build ${BUILD_PATH} --target install