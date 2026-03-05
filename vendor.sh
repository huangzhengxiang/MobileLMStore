#!/bin/bash

if [ ! -d "vendor" ]; then
    echo "sqlite not found, downloading..."
    mkdir -p vendor
    curl -L -o sqlite-amalgamation.zip https://www.sqlite.org/2024/sqlite-amalgamation-3450300.zip
    unzip sqlite-amalgamation.zip
    mv sqlite-amalgamation-3450300/* vendor/
    rm -rf sqlite-amalgamation-3450300
    rm sqlite-amalgamation.zip
    echo "sqlite downloaded into vendor/"
else
    echo "sqlite already exists, skip downloading."
fi