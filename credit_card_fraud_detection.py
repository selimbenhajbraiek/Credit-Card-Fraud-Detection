



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

file_path = '/content/drive/MyDrive/creditcard.csv'

# Load dataset
df = pd.read_csv(file_path)

# Generate profiling report
profile = ProfileReport(df, title="Credit Card Fraud Detection Profiling Report", explorative=True)
profile_output_path = "/content/drive/MyDrive/BenHajBraiek-creditfraud-eda.html"
profile.to_file(profile_output_path)

# Basic dataset info
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("Missing values:", df.isnull().sum().sum())
print("Fraud distribution:\n", df["Class"].value_counts(normalize=True))

#  Visual EDA

# 1. Target distribution
sns.countplot(x='Class', data=df)
plt.title('Distribution of Transactions (0 = Legit, 1 = Fraud)')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()

# 2. Distribution of transaction amount
sns.histplot(df['Amount'], bins=50, kde=True)
plt.title('Distribution of Transaction Amount')
plt.xlabel('Amount')
plt.ylabel('Frequency')
plt.show()

# 3. Distribution of transaction time
sns.histplot(df['Time'], bins=50, kde=True)
plt.title('Distribution of Transaction Time')
plt.xlabel('Time (seconds)')
plt.ylabel('Frequency')
plt.show()

# 4. Correlation heatmap
plt.figure(figsize=(12, 10))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, cmap='coolwarm_r', center=0)
plt.title("Feature Correlation Heatmap")
plt.show()

# 5. Top features correlated with fraud
cor_target = abs(corr_matrix["Class"]).sort_values(ascending=False)
print("Top features correlated with fraud:\n", cor_target[1:6])

# 6. Boxplots of top correlated features
top_features = cor_target[1:4].index.tolist()
for feature in top_features:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x="Class", y=feature, data=df)
    plt.title(f'{feature} vs Class')
    plt.show()