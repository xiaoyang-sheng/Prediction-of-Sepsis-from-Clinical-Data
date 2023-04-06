from xgboost import XGBClassifier
import pandas as pd

def apply():
    xgb_model = XGBClassifier()
    # load the pre-trained model
    xgb_model.load_model('xgb_model.json')
    data = pd.read_csv('final_data/test_set.csv')
    data = data.drop(data.columns[0], axis=1)
    # predict the outcome using model
    y_test = xgb_model.predict(data)
    y_pred_proba = xgb_model.predict_proba(data)
    data['outcome'] = y_test
    # get the score of each prediction
    data['score'] = y_pred_proba[:, 1]
    test_id = pd.read_csv('final_data/test_nolabel.csv')
    final_res = pd.merge(test_id['ID'], data[['ID', 'outcome', 'score']], how='left', on='ID')
    # save the results
    final_res.to_csv('test_result.csv', index=False)


if __name__ == "__main__":
    apply()