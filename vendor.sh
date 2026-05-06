#!/bin/bash

set -e

mkdir -p vendor

if [ ! -f "vendor/sqlite3.c" ]; then
    echo "sqlite not found, downloading..."
    curl -fL -o sqlite-amalgamation.zip https://www.sqlite.org/2024/sqlite-amalgamation-3450300.zip
    unzip -o sqlite-amalgamation.zip
    mv sqlite-amalgamation-3450300/* vendor/
    rm -rf sqlite-amalgamation-3450300
    rm -f sqlite-amalgamation.zip
    echo "sqlite downloaded into vendor/"
else
    echo "sqlite already exists, skip downloading."
fi

if [ ! -f "vendor/rapidjson/include/rapidjson/document.h" ]; then
    echo "rapidjson not found, downloading..."
    curl -fL -o rapidjson-v1.1.0.zip https://github.com/Tencent/rapidjson/archive/refs/tags/v1.1.0.zip
    unzip -o rapidjson-v1.1.0.zip
    rm -rf vendor/rapidjson
    mv rapidjson-1.1.0 vendor/rapidjson
    rm -f rapidjson-v1.1.0.zip
    echo "rapidjson downloaded into vendor/rapidjson"
else
    echo "rapidjson already exists, skip downloading."
fi
