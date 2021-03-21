TRAIN_FILE = "data/train.csv"
TEST_FILE = "data/test.csv"
N_CLASS = 7
INPUT_DIM = 54
BATCH_SIZE = 64
KAGGLE_COMMAND = """kaggle competitions submit -c forest-cover-type-prediction -f submission.csv -m "{}" """
KAGGLE_SUBMISSIONS = "kaggle competitions submissions -c forest-cover-type-prediction"
