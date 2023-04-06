import data_clean_merge
import model_train
import model_apply


if __name__ == "__main__":
    print("start the data clean...")
    data_clean_merge.data_clean()
    print("start the model training...")
    model_train.train()
    print("apply the model to test set...")
    model_apply.apply()
