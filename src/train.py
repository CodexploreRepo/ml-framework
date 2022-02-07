import pandas as pd
import os

from sklearn import ensemble
from sklearn import preprocessing
from sklearn import metrics
import joblib

from . import dispatcher

TRAINING_DATA = os.environ.get("TRAINING_DATA")
FOLD = int(os.environ.get("FOLD"))
MODEL = os.environ.get("MODEL")

FOLD_MAPPING = {
    0: [1,2,3,4],
    1: [0,2,3,4],
    2: [1,0,3,4],
    3: [1,2,0,4],
    4: [1,2,3,0]
}
if __name__ == "__main__":
    df = pd.read_csv(TRAINING_DATA)
    train_df = df[df.kfold.isin(FOLD_MAPPING[FOLD])] #train_df =  kfold in [0,2,3,4] 
    valid_df = df[df.kfold == FOLD] #valid_df = (kfold = 1)

    ytrain = train_df.target.values
    yvalid = valid_df.target.values

    #X_train, X_valid
    train_df = train_df.drop(["id", "target", "kfold"], axis = 1)
    valid_df = valid_df.drop(["id", "target", "kfold"], axis = 1)

    #to ensure the column order of train_df and valid_df are the same
    valid_df = valid_df[train_df.columns]

    #Label Encoder: encode values in a Column to value between 0 and n_classes-1.
    label_encoders = [] #create a list to contain all label encoder
    for c in train_df.columns:
        lbl = preprocessing.LabelEncoder()
        #Fit all values in a column
        lbl.fit(train_df[c].values.tolist() + valid_df[c].values.tolist())
        
        train_df.loc[:,c] = lbl.transform(train_df[c].values.tolist())
        valid_df.loc[:,c] = lbl.transform(valid_df[c].values.tolist())

        #Store the label encoder for each column:
        label_encoders.append((c,lbl))


    #Data is ready to train
    clf = dispatcher.MODELS[MODEL]
    clf.fit(train_df, ytrain)
    preds = clf.predict_proba(valid_df)[:,1] #only 1 class
    print(f"AUC Score: {metrics.roc_auc_score(yvalid, preds)}")

    # joblib.dump(label_encoders, f"models/{MODEL}_label_encoder.pkl")
    # joblib.dump(clf, f"models/{MODEL}.pkl")