# Import necessary libraries
import os
import gdown
import pandas as pd

# Define a folder name in the current working directory
folder_name = 'data'  # or any folder name you prefer
folder_path = os.path.join(os.getcwd(), folder_name)

# Create the directory if it doesn't exist
os.makedirs(folder_path, exist_ok=True)

# Google Drive file URL for downloading data (replace with your own file IDs)
file_urls = {
    'Adani Enterprises': 'https://drive.google.com/uc?id=1hjfIe3z1zbGwtawYeXuqItljIMAzvd7G',
    'Adani Ports': 'https://drive.google.com/uc?id=1kPysVUJ2oAM3DS9t9q3gnK3SzAiWY8nG',
    'Bank of Baroda': 'https://drive.google.com/uc?id=1qU5wdFCBmvy1jIDvqv1NmxRyKftm4jis',
    'Bharti Airtel': 'https://drive.google.com/uc?id=1VJE65JcFDHQjmrurlZshtvfVAqK9M1yO',
    'Hindalco Industries': 'https://drive.google.com/uc?id=1kLUHgG1-0krGRITcvpyPwoNFs0FKiGd6'
    # Add more companies as needed
}

# Dictionary to store dataframes
stock_data = {}

# Download each CSV file from Google Drive and store it in the stock_data dictionary
for company, file_url in file_urls.items():
    output_file = os.path.join(folder_path, f"{company}.csv")
    gdown.download(file_url, output_file, quiet=False)
    
    # Read the downloaded CSV file
    stock_data[company] = pd.read_csv(output_file)

# Output the stock data keys (company names)
print(stock_data.keys())

# 1. Split Date and Time
for company, df in stock_data.items():
    df['date'] = pd.to_datetime(df['date'])  # Convert to datetime objects
    df['Trade_Date'] = df['date'].dt.date  # Extract date
    df['Trade_Time'] = df['date'].dt.time  # Extract time

# 2. Find Common Trading Dates (Ignoring Time)
common_dates = set(stock_data['Adani Enterprises']['Trade_Date'])
for company, df in stock_data.items():
    common_dates = common_dates.intersection(set(df['Trade_Date']))
common_dates = sorted(list(common_dates), reverse=True)

# 3. Narrow Down to Last 30 Trading Days
common_dates = common_dates[:30]

# 4. Filter DataFrames to the Last 30 Trading Days
for company, df in stock_data.items():
    stock_data[company] = df[df['Trade_Date'].isin(common_dates)].reset_index(drop=True)

# 5. Feature Engineering
for company, df in stock_data.items():
    # Absolute difference between high and low
    df['High_Low_Diff'] = abs(df['high'] - df['low'])

    # Absolute difference between open and close
    df['Open_Close_Diff'] = abs(df['open'] - df['close'])

    # Positive or negative change (1 for positive, 0 for negative)
    df['Change_Type'] = (df['close'] - df['open']).apply(lambda x: 1 if x > 0 else 0)

    # Percentage change from the open value
    df['Pct_Change'] = ((df['close'] - df['open']) / df['open']) * 100

# Prepare Data for Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Combine all stocks data into one dataframe
all_stocks = pd.concat([stock_data[stock] for stock in stock_data.keys()], ignore_index=True)

# Select relevant features and target variable
features = ['open', 'high', 'low', 'close', 'volume', 'High_Low_Diff', 'Open_Close_Diff', 'Pct_Change']
target = 'Change_Type'

# Prepare the feature matrix (X) and target vector (y)
X = all_stocks[features]
y = all_stocks[target]

# Split the data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the feature data (optional but recommended for Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the Logistic Regression model
model = LogisticRegression()

# Train the model on the training data
model.fit(X_train_scaled, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Print the results
print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Classification Report:\n{class_report}")

# Create the 'model' folder if it doesn't exist
if not os.path.exists('model'):
    os.makedirs('model')

# After training the model
model.save('model/stock_price_model.h5')
print("Model saved successfully.")

