# Machine Learning Framework

## 1. Conda Environment
```Python
#To create a new conda env
conda create--name myenv_name

#To list env in Conda
conda env list

#To activate & deactivate
conda activate myenv_name
conda deactivate
```
- In the activated conda enviroment, you can install or execute python scrip
```Python
#Once activated, to install
conda install numpy matplotlib scikit-learn

#Once activated, to run the python script
python -m scr.train.py #Run train.py scrip in src folder
sh run.sh randomforest #Run via bash script run.sh
```
