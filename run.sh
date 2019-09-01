#!/bin/bash

if [ "$(uname)" == "Linux" ]; then
#    ../recursivefilter-build/test_recursivefilter_checkmath
    ../recursivefilter-build/test_recursivefilter_measuretime
else
    ./bin/test_recursivefilter
fi
