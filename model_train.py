import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc, precision_score, balanced_accuracy_score
import matplotlib.pyplot as plt
# from sklearn.utils.class_weight import compute_sample_weight


def train():
    # open the log file to record the performance indicators, they are saved in 'log.txt'
    log = open("log.txt", mode="a+", encoding="utf-8")
    # read the files saved previously
    data = pd.read_csv('final_data/train_set.csv')
    data = data.drop(data.columns[0], axis=1)
    # get the positive and negative patient id to better separate the train and validation set
    data_pos = data[data['Outcome'] == 1]
    data_neg = data[data['Outcome'] == 0]
    pos_id = data_pos.ID.unique().tolist()
    neg_id = data_neg.ID.unique().tolist()
    # manually separate the train and validation set by 70% and 30%
    pos_valid_id = pos_id[:int(len(pos_id)*0.3)]
    pos_train_id = pos_id[int(len(pos_id)*0.3):]
    neg_valid_id = neg_id[:int(len(neg_id)*0.3)]
    neg_train_id = neg_id[int(len(neg_id)*0.3):]
    # get the train set
    train_set = data.loc[data['ID'].isin(pos_train_id+neg_train_id)]
    # get the validation set
    validation_set = data.loc[data['ID'].isin(pos_valid_id+neg_valid_id)]
    X_train = train_set.drop(['Outcome'], axis=1)
    y_train = train_set['Outcome']
    X_valid = validation_set.drop(['Outcome'], axis=1)
    y_valid = validation_set['Outcome']
    # weights = compute_sample_weight(class_weight='balanced', y=y_train)

    # get the ratio of the negative samples/positive samples, which parses to model further to resolve the imbalance of
    # the data set
    ratio = sum(y_train == 0)/sum(y_train == 1)

    # the candidate of the params of xgboost model
    param_test1 = {
        'max_depth': range(3, 10, 2),
        'learning_rate': [0.01, 0.02, 0.04, 0.06, 0.08, 0.1],
        'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9]
    }

    # use grid search cv to find the best params from the candidate
    gsearch1 = GridSearchCV(estimator=XGBClassifier(n_estimators=220,scale_pos_weight=ratio), param_grid=param_test1,
                            scoring='roc_auc', n_jobs=-1, cv=5)

    gsearch1.fit(X_train, y_train)

    print("The best params are", gsearch1.best_params_, file=log)
    # initialize the xgboost classifier using the cv results
    xgb_model = XGBClassifier(
        n_estimators=220,
        learning_rate=gsearch1.best_params_['learning_rate'],
        colsample_bytree=gsearch1.best_params_['colsample_bytree'],
        max_depth=gsearch1.best_params_['max_depth'],
        scale_pos_weight=ratio
    )

    # fit the model
    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_valid)

    # some performance indicators
    print("balanced accuracy is :", balanced_accuracy_score(y_valid, y_pred), file=log)
    print("model score is :", xgb_model.score(X_valid, y_valid), file=log)
    print("precision score is:", precision_score(y_valid, y_pred), file=log)

    # plot the ROC curve
    y_pred_proba = xgb_model.predict_proba(X_valid)
    fpr, tpr, thresholds = roc_curve(y_valid,y_pred_proba[:,1], pos_label=1)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=lw,
        label="ROC curve (area = %0.2f)" % roc_auc,
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic example")
    plt.legend(loc="lower right")
    plt.savefig('auc_roc.pdf')
    print('AUC is:', roc_auc, file=log)

    # save the model to json
    xgb_model.save_model("xgb_model.json")
    log.close()

if __name__ == "__main__":
    train()