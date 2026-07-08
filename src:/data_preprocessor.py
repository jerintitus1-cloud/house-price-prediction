import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def load_and_split_data(data_path, target_column='Price', test_size=0.2, random_state=42):
    """Loads dataset and splits it into training and testing features."""
    df = pd.read_csv(data_path)
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def build_preprocessor_pipeline():
    """Creates an isolated data preprocessor transformation matrix."""
    num_features = ['Area', 'Bedrooms', 'Bathrooms', 'Age']
    cat_features = ['Location', 'Property_Type']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
        ])
    return preprocessor