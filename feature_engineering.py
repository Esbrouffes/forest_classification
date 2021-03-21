from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import constants as CN
import numpy as np
import pandas as pd
import torch


def feature_engineering(feature_data):
    """
    This function creates new columns which are linear combinations of
     others and which make sense
    """
    feature_data.drop(['Soil_Type7', 'Soil_Type15'], axis=1, inplace=True)

    feature_data['Real_Distance_To_Hydrology'] = round(np.sqrt(
        feature_data['Horizontal_Distance_To_Hydrology'] ** 2 + feature_data[
            'Vertical_Distance_To_Hydrology'] ** 2), 1)
    feature_data['Mean_Hillshade'] = round((feature_data['Hillshade_9am'] + feature_data[
        'Hillshade_Noon'] + feature_data['Hillshade_3pm']) / 3, 1)
    feature_data['Horizontal_Distance_Mean_Pos'] = round((feature_data[
                                                              'Horizontal_Distance_To_Hydrology'] +
                                                          feature_data[
                                                              'Horizontal_Distance_To_Roadways'] +
                                                          feature_data[
                                                              'Horizontal_Distance_To_Fire_Points']) / 3,
                                                         1)
    feature_data['Horizontal_Distance_Mean_Neg'] = round((feature_data[
                                                              'Horizontal_Distance_To_Hydrology'] -
                                                          feature_data[
                                                              'Horizontal_Distance_To_Roadways'] -
                                                          feature_data[
                                                              'Horizontal_Distance_To_Fire_Points']) / 3,
                                                         1)
    feature_data['Horizontal_Distance_HR_Pos'] = feature_data['Horizontal_Distance_To_Hydrology'] + \
                                                 feature_data['Horizontal_Distance_To_Roadways']
    feature_data['Horizontal_Distance_HR_Neg'] = feature_data['Horizontal_Distance_To_Hydrology'] - \
                                                 feature_data['Horizontal_Distance_To_Roadways']
    feature_data['Horizontal_Distance_RF_Pos'] = feature_data['Horizontal_Distance_To_Roadways'] + \
                                                 feature_data['Horizontal_Distance_To_Fire_Points']
    feature_data['Horizontal_Distance_RF_Neg'] = feature_data['Horizontal_Distance_To_Roadways'] - \
                                                 feature_data['Horizontal_Distance_To_Fire_Points']
    feature_data['Horizontal_Distance_HF_Pos'] = feature_data['Horizontal_Distance_To_Hydrology'] + \
                                                 feature_data['Horizontal_Distance_To_Fire_Points']
    feature_data['Horizontal_Distance_HF_Neg'] = feature_data['Horizontal_Distance_To_Hydrology'] - \
                                                 feature_data['Horizontal_Distance_To_Fire_Points']
    feature_data['Elevation_Of_Hydrology'] = feature_data['Elevation'] + feature_data[
        'Vertical_Distance_To_Hydrology']
    feature_data['Difference_Elevation_Higher_Lowest_Point'] = feature_data['Elevation'] * np.tan(
        feature_data['Slope'] * np.pi / 180)
    cols = feature_data.columns.tolist()
    cols = cols[:10] + cols[-12:] + cols[10:-12]
    feature_data = feature_data[cols]

    feature_data['Aspect'] = [
        'NORTH' if 0 <= x < 45 else 'EST' if 45 <= x < 135 else 'SOUTH' if 135 <= x < 225 else 'WEST'
        for x in feature_data['Aspect']]
    feature_data = pd.get_dummies(feature_data,
                                  columns=['Aspect'])  ### transforming using OneHotEncoding
    cols_asp = feature_data.columns.tolist()
    cols_asp = cols_asp[:21] + cols_asp[-4:] + cols_asp[21:-4]
    feature_data = feature_data[cols_asp]

    return feature_data


def get_dataloaders(train_df):
    labels = train_df["Cover_Type"]
    labels_array = labels.values
    labels_array -= 1
    samples = train_df.drop(columns=["Cover_Type", "Id"])
    train_samples = torch.Tensor(samples.values)
    train_labels = torch.Tensor(labels_array)
    X_train, X_val, y_train, y_val = train_test_split(train_samples, train_labels, test_size=0.20)
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=CN.BATCH_SIZE, drop_last=True)
    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=CN.BATCH_SIZE, drop_last=True)
    print(f"Training data has {len(samples.columns)} dimensions")
    return train_loader, val_loader


def get_test_data(test_df):
    df_no_id = test_df.drop(columns="Id")
    return test_df, torch.Tensor(df_no_id.values)
