# 🛠️ Code Base API Reference Manual

Detailed technical specifications for the internal programmatic execution modules located inside the `src/` package.

---

### 📦 Module: `data_preprocessor.py`

#### `load_and_split_data(data_path, target_column='Price', test_size=0.2)`
Loads a target spreadsheet matrix file and splits features away from targeted evaluation variables.
* **Parameters**:
  * `data_path` (*str*): The absolute system disk path locating the target CSV.
  * `test_size` (*float*): Proportion of rows saved for model validation (default: `0.2`).
* **Returns**: `X_train`, `X_test`, `y_train`, `y_test` (*DataFrames/Series*).

#### `build_preprocessor_pipeline()`
Configures static transformations for incoming numerical and categorical column metrics.
* **Returns**: `ColumnTransformer` object instance factory.

---

### 📦 Module: `model_trainer.py`

#### `train_production_model(X_train, y_train, preprocessor)`
Assembles and optimizes structural estimators to fit training input historical metrics.
* **Parameters**:
  * `preprocessor` (*ColumnTransformer*): Configured data transformations.
* **Returns**: Fully fitted and operational `Pipeline` object instance model.

#### `export_model(model_pipeline, output_directory, filename='best_model.pkl')`
Serializes the operational pipeline factory straight down into a physical file on your machine.