import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pyarrow.parquet as pq
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import cv2  # OpenCV for image processing
# Load train events data
train_events = pd.read_csv('/kaggle/input/child-mind-institute-detect-sleep-states/train_events.csv')

# Convert timestamps to datetime
train_events['timestamp'] = pd.to_datetime(train_events['timestamp'])

# Sort events for accurate time difference calculation
train_events = train_events.sort_values(['series_id', 'night', 'timestamp'])

# Calculate time difference in minutes
train_events['time_diff'] = train_events.groupby(['series_id', 'night'])['timestamp'].diff().dt.total_seconds() / 60

# Add 'risky' flag based on short sleep durations (<10 mins)
train_events['risky'] = train_events['time_diff'].apply(lambda x: 1 if x and x < 10 else 0)

print(train_events.head())
# Load train events data
train_events = pd.read_csv('/kaggle/input/child-mind-institute-detect-sleep-states/train_events.csv')

# Convert timestamps to datetime
train_events['timestamp'] = pd.to_datetime(train_events['timestamp'])

# Sort events for accurate time difference calculation
train_events = train_events.sort_values(['series_id', 'night', 'timestamp'])

# Calculate time difference in minutes
train_events['time_diff'] = train_events.groupby(['series_id', 'night'])['timestamp'].diff().dt.total_seconds() / 60

# Add 'risky' flag based on short sleep durations (<10 mins)
train_events['risky'] = train_events['time_diff'].apply(lambda x: 1 if x and x < 10 else 0)

print(train_events.head())
# Load time-series movement data
train_series = pq.read_table('/kaggle/input/child-mind-institute-detect-sleep-states/train_series.parquet').to_pandas()

# Detect sudden movements based on large angle changes (>30 degrees)
train_series['dangerous_movement'] = train_series['anglez'].apply(lambda x: 1 if abs(x) > 30 else 0)

print(train_series.head())
# Risky Events Distribution
sns.countplot(data=train_events, x='risky')
plt.title('Risky Events Distribution')
plt.show()

# Sudden Movement Detection
sns.histplot(train_series['anglez'], kde=True)
plt.title('Distribution of Anglez Values')
plt.show()
# Drop unnecessary columns
train_events_new = train_events.drop(['series_id', 'timestamp'], axis='columns')

# Encode categorical variables
label_encoder = LabelEncoder()
for col in train_events_new.select_dtypes(include='object'):
    train_events_new[col] = label_encoder.fit_transform(train_events_new[col])

print(train_events_new.head())
# Define CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(2, activation='softmax')  # Output: 'safe' or 'dangerous'
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print(model.summary())
# Example: Load and preprocess image data (replace with your dataset)
X_train = []  # List to hold image data
y_train = []  # List to hold labels (0 = safe, 1 = dangerous)

# Load images and preprocess (assuming images are in 'train_images' directory)
for img_name, label in [('img1.jpg', 0), ('img2.jpg', 1)]:  # Replace with actual data loop
    img = cv2.imread(f'/path/to/train_images/{img_name}')
    img = cv2.resize(img, (128, 128)) / 255.0  # Normalize pixel values
    X_train.append(img)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Train the CNN model
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)
def predict_movement(image_path, anglez):
    # Load and preprocess the image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128)) / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict using the CNN model
    prediction = model.predict(img)
    position = 'safe' if np.argmax(prediction) == 0 else 'dangerous'

    # Combine with movement risk from anglez
    movement_risk = 'dangerous' if abs(anglez) > 30 else 'safe'

    print(f"Position: {position}, Movement Risk: {movement_risk}")

# Test the prediction function
predict_movement("/path/to/test_image.jpg", anglez=35)
