#!/bin/bash

for i in 10 20 30 40 50
do
    for j in $(seq 0 0.05 0.3)
    do 
        # echo "Probability is $j and num agent is $i"
        python3.5 catastrophe_game2.py $i $j
    done
done
