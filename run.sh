export TRAINING_DATA=input/train_folds.csv
export TEST_DATA=input/test_cat.csv

export FOLD=0
#this is to receive "model arg" from sh run.sh
#for ex: sh run.sh randomforest
#then MODEL = randomforest
export MODEL=$1 

python -m src.train

