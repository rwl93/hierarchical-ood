#!/bin/bash
./local/protobuf-3.18.0-rc2/bin/protoc \
    -I=./ \
    --python_out=. \
    ./lib/protos/main.proto
./local/protobuf-3.18.0-rc2/bin/protoc \
    -I=./ \
    --python_out=. \
    ./lib/protos/ensemble.proto
# ./local/protoc-21.4-osx-aarch_64/bin/protoc \
#     -I=./ \
#     --python_out=. \
#     ./lib/protos/main.proto

# ./local/protoc-21.4-osx-aarch_64/bin/protoc \
#     -I=./ \
#     --python_out=. \
#     ./lib/protos/ensemble.proto
