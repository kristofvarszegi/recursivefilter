#!/bin/bash

if [ "$(uname)" == "Linux" ]; then
    ../recursivefilter-build/test_recursivefilter
else
    ./bin/test_recursivefilter
fi
