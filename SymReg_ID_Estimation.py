
#This script currently computes the Global Intrinsic Dimension estimates for each of the Symbolic Regression Datasets within the Penn Machine Learning Benchmark libary.
import skdim
from skdim.id import lPCA, MLE, DANCo, TwoNN
from skdim.id import ESS, CorrInt, FisherS, KNN
from skdim.id import MADA, MiND_ML, MOM, TLE
import numpy as np
import pandas as pd
import random
from pmlb import fetch_data
from pmlb import dataset_names, regression_dataset_names
from sklearn.model_selection import train_test_split
import time
from tqdm import tqdm
import skdim
from skdim.id import lPCA, MLE, DANCo, TwoNN
from skdim.id import ESS, CorrInt, FisherS, KNN
from skdim.id import MADA, MiND_ML, MOM, TLE
import numpy as np
import pandas as pd
import random
from pmlb import fetch_data
from pmlb import dataset_names, regression_dataset_names
from sklearn.model_selection import train_test_split
import time
from tqdm import tqdm

# Initialize intrinsic dimension estimators
estimators = {
    'lPCA': lPCA(),
    'MLE': MLE(),
    'DANCo': DANCo(),
    'TwoNN': TwoNN(),
    'ESS': ESS(),
    'CorrInt': CorrInt(),
    'FisherS': FisherS(),
    'KNN': KNN(),
    'MADA': MADA(),
    'MiND_ML': MiND_ML(),
    'MOM': MOM(),
    'TLE': TLE()
}

# Create a DataFrame to store the results
df_results = pd.DataFrame(columns=['dataset_name'] + list(estimators.keys()) + ['num_params', 'num_train_rows'])

#run ID estimation on subset
regression_datasets = regression_dataset_names[:10] 

# Iterate over each dataset in the regression_dataset_names list with a progress bar
for dataset_name in tqdm(regression_datasets, desc="Processing datasets"):
    start_time = time.time()  # Start time for the current dataset
    try:
        data = fetch_data(dataset_name)
        X = data.drop('target', axis=1)  # Features
        y = data['target']  # Target

        # Check for and remove duplicate rows
        X = X.drop_duplicates()
        y = y.loc[X.index]  # Keep target values consistent with the cleaned feature set

        # Convert to numpy arrays
        X = X.values
        y = y.values

        # Split the data into training and testing sets (70-30 split)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Subsample the training data if it has more than 100,000 rows
        num_train_rows = X_train.shape[0]  # Get number of rows in X_train
        if num_train_rows > 10000:
            X_train = X_train[np.random.choice(X_train.shape[0], 1000, replace=False), :]

        num_params = X_train.shape[1]  # Get number of feature columns in X_train

        # Compute intrinsic dimension estimates on the training data
        results = {}
        for name, estimator in estimators.items():
            estimate = estimator.fit_transform(X_train)
            results[name] = estimate

        # Prepare the data to be added to the DataFrame
        data_to_add = {'dataset_name': dataset_name}
        for estimator_name, estimate in results.items():
            data_to_add[estimator_name] = estimate
        data_to_add['num_params'] = num_params
        data_to_add['true_num_train_rows'] = num_train_rows

        # Append the data to the DataFrame
        df_results = pd.concat([df_results, pd.DataFrame([data_to_add])], ignore_index=True)
    except Exception as e:
        print(f"Error processing dataset {dataset_name}: {e}")
    end_time = time.time()  # End time for the current dataset
    print(f"Processing time for dataset {dataset_name}: {end_time - start_time:.2f} seconds")

# Save the DataFrame to a CSV file
df_results.to_csv('SymRegDataIDs.csv', index=False)
# Optionally, print a message indicating the file has been saved
print("Results have been saved to 'SymRegDataIDs.csv'")