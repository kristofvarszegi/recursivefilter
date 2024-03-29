PROJECT_NAME=recursivefilter

if [[ "$OSTYPE" == "msys" ]]; then
    DEPS_PATH="C:/work/vs2017-x64"
    GENERATE_PROJECT_TYPE="Visual Studio 15 2017 Win64"
    BUILD_PATH="build"
    echo "Deps path: ${DEPS_PATH}"
else
    DEPS_PATH=""
    GENERATE_PROJECT_TYPE="Eclipse CDT4 - Unix Makefiles"
    BUILD_PATH="../${PROJECT_NAME}-build"
fi
echo "Generating \"${GENERATE_PROJECT_TYPE}\""
echo "Build path: ${BUILD_PATH}"

export MAKEFLAGS=-j8
cmake -H. -B${BUILD_PATH} \
    -G"${GENERATE_PROJECT_TYPE}" \
    -DCMAKE_PREFIX_PATH="${DEPS_PATH}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=${BUILD_PATH}
LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${BUILD_PATH}
cmake --build ${BUILD_PATH} --target install