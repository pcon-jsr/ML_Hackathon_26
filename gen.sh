#!/bin/sh
cd classification
python3 generate_p1_dataset.py
cp train.csv ../static/dataset1.csv
cp eval.csv ../static/eval1.csv
cd ..


cd regression
python3 generate_p2_dataset.py
cp train.csv ../static/dataset2.csv
cp eval.csv ../static/eval2.csv
cd ..
