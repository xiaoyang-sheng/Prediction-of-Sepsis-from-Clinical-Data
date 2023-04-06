import pandas as pd
import os


# the sub function that read each txt file, and restructure the dataframe, calculate the summary statistic, i.e. the
# mean and the difference of maximum and minimum of each variable.
def df_restructure(df):
    id = int(df[df['Variable'] == 'ID'].iloc[0]['Value'])
    gender = int(df[df['Variable'] == 'Gender'].iloc[0]['Value'])
    age = df[df['Variable'] == 'Age'].iloc[0]['Value']
    unit1 = df[df['Variable'] == 'Unit1']['Value'] if not df[df['Variable'] == 'Unit1'].empty else pd.NA
    unit2 = df[df['Variable'] == 'Unit2']['Value'] if not df[df['Variable'] == 'Unit2'].empty else pd.NA
    df = df[~(df['Variable'].isin(['ID', 'Age', 'Gender', 'Unit1', 'Unit2']))]
    df = df.pivot(index='Hour', columns='Variable', values='Value')
    df.reset_index(inplace=True)
    df = df.drop(columns=['Hour'])
    df = df.ffill(axis=0)
    tmp = df.mean()
    tmp = tmp.to_frame().transpose()
    tmp = tmp.add_suffix('_mean')
    tmp2 = df.max()-df.min()
    tmp2 = tmp2.to_frame().transpose()
    tmp2 = tmp2.add_suffix('_diff')
    df = pd.concat([tmp, tmp2], axis=1)
    df['ID'] = id
    df['Gender'] = gender
    df['Age'] = age
    df['Unit1'] = unit1
    df['Unit2'] = unit2
    return df


def data_clean():
    file_path = 'final_data/x_all'
    files = os.listdir(file_path)
    final_df = pd.DataFrame()
    # apply the sub function to all the txt file in the files
    for file in files:
        final_df = pd.concat([final_df, df_restructure(pd.read_csv(os.path.join(file_path, file)))], ignore_index=True)

    final_df.to_csv('final_data/final_data_md.csv')

    # drop the following columns because there are too many NaN values.
    final_df = final_df.drop(columns=['Unit1', 'Unit2', 'Bilirubin_direct_mean', 'Bilirubin_direct_diff', 'EtCO2_mean',
                                      'EtCO2_diff', 'TroponinI_mean', 'TroponinI_diff', 'Fibrinogen_mean',
                                      'Fibrinogen_diff'])
    # read the train set and get the ID, separating the train/test set from the final_df
    train_outcome = pd.read_csv('final_data/train_outcome.csv')
    # merge the train set outcome with the final_df
    final_df = pd.merge(final_df, train_outcome, how='left', on='ID')
    df_test = final_df[pd.isna(final_df['Outcome'])]
    df_test = df_test.drop(columns=['Outcome'])
    # write the seperated test set to csv files
    df_test.to_csv('final_data/test_set.csv')
    df_train = final_df[~pd.isna(final_df['Outcome'])]
    # write the seperated train set to csv files
    df_train.to_csv('final_data/train_set.csv')


if __name__ == "__main__":
    data_clean()
