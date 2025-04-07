# Credit Card Fraud Detection - Exploratory Data Analysis (EDA)

This notebook explores a dataset of credit card transactions with the goal of identifying fraudulent transactions.

### Dataset Availability:
The dataset used in this analysis is the "Credit Card Fraud Detection" dataset from Kaggle. It is available publicly, and we loaded it from Google Drive at the following path: `/content/drive/MyDrive/creditcard.csv`.

### Interpretation of Variables:
- `Time`: The number of seconds elapsed since the first transaction.
- `V1-V28`: Anonymized features that contain information about the transactions.
- `Amount`: The transaction amount.
- `Class`: Target variable (0 = legitimate, 1 = fraudulent).

## Profiling Report
We will generate a profile report for an in-depth look at the dataset's features, including missing values, correlations, and distribution of features.

### Profiling Report:

```python
from ydata_profiling import ProfileReport
import pandas as pd

# Load the dataset from Google Drive
file_path = "/content/drive/MyDrive/creditcard.csv"
df = pd.read_csv(file_path)

# Generate the profile report
profile = ProfileReport(df, title="Credit Card Fraud Detection Profiling Report", explorative=True)

# Save the profile report
profile_output_path = "/content/drive/MyDrive/BenHajBraiek-creditfraud-eda.html"
profile.to_file(profile_output_path)
