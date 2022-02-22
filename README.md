# Machine Learning Framework

# Table of contents
- [Table of contents](#table-of-contents)
- [1. Conda Environment](#1-conda-environment)
- [Resources](#resources)

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
sh run.sh              #Run via bash script run.sh
sh run.sh randomforest #Run via bash script run.sh and passing "randomforest" as an argument
```

## Resources:
- [Dataset](https://www.kaggle.com/abhishek/aaamlp)

[(Back to top)](#table-of-contents)
