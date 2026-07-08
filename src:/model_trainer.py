import os
import pickle
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

def train_production_model(X_train, y_train, preprocessor):
    """Assembles the preprocessor and estimator model factory, then fits data."""
    production_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    production_pipeline.fit(X_train, y_train)
    return production_pipeline

def export_model(model_pipeline, output_directory, filename='best_model.pkl'):
    """Safely saves the trained model binary out into the system workspace."""
    os.makedirs(output_directory, exist_ok=True)
    target_path = os.path.join(output_directory, filename)
    
    with open(target_path, 'wb') as file:
        pickle.dump(model_pipeline, file)
    print(f"✅ Production engine exported successfully to: {target_path}")