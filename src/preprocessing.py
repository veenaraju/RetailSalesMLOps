import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(file_path):

    df = pd.read_csv(file_path)

    print("\nDataset Preview:")
    print(df.head())

    print("\nDataset Shape:")
    print(df.shape)

    print("\nColumn Information:")
    print(df.info())

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nStatistical Summary:")
    print(df.describe())

    return df


def preprocess_data(df, target_column, test_size, random_state):

    # Remove rows with missing target values
    df = df.dropna(subset=[target_column])

    # Separate input features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print("\nTraining Records:", len(X_train))
    print("Testing Records:", len(X_test))

    return X_train, X_test, y_train, y_test


def save_processed_data(df, file_path):

    df.to_csv(file_path, index=False)

    print("\nProcessed dataset saved:", file_path)
