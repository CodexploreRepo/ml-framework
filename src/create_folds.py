import pandas as pd
from sklearn import model_selection

if __name__ == "__main__":
    df = pd.read_csv("input/train.csv")
    df["kfold"] = -1

    #This is to shuffle the data
    #frac=1 means return all rows (in random order)
    #drop=True prevents .reset_index from creating a column containing the old index entries.
    df = df.sample(frac=1).reset_index(drop=True)

    #Generate test sets such that all contain the same distribution of classes, or as close as possible.
    kf = model_selection.StratifiedKFold(n_splits=5, shuffle=False)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X=df, y=df.target.values)):
        print(len(train_idx), len(val_idx))
        df.loc[val_idx, 'kfold'] = fold
    
    df.to_csv("input/train_folds.csv", index = False)