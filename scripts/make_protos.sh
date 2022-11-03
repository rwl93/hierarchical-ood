#!/bin/bash
./local/protoc/bin/protoc \
    -I=./ \
    --python_out=. \
    ./lib/protos/main.proto
./local/protoc/bin/protoc \
    -I=./ \
    --python_out=. \
    ./lib/protos/ensemble.proto
