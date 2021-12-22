#!/bin/bash

mkdir ./python/data/downloaded
mkdir ./python/data/downloaded/WILLOW
mkdir ./matlab/data

cd ./python/data/downloaded/WILLOW

echo -e "\e[1mGetting WILLOW data\e[0m"
wget http://www.di.ens.fr/willow/research/graphlearning/WILLOW-ObjectClass_dataset.zip
unzip WILLOW-ObjectClass_dataset.zip
echo -e "\e[32m... done\e[0m"

cd ../../../../
cp -r ./python/data/downloaded/WILLOW/WILLOW-ObjectClass ./matlab/data/

