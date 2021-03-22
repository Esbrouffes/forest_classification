TRAIN_FILE = "data/train.csv"
TEST_FILE = "data/test.csv"
N_CLASS = 7
BATCH_SIZE = 64
# Command to submit results to the kaggle competition
KAGGLE_COMMAND = """kaggle competitions submit -c forest-cover-type-prediction -f submission.csv -m "{}" """
# Command to show the submissions to the competition
KAGGLE_SUBMISSIONS = "kaggle competitions submissions -c forest-cover-type-prediction"
