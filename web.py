import os
import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from flask import Flask, render_template, request

app = Flask(__name__)

# Step 1: Data Preparation

dataset_dir = r"/Users/satyasanjanarama/Desktop/Internship Code/Plant_Database"
csv_file_path = r"/Users/satyasanjanarama/Desktop/Internship Code/dataset/dataset4.csv"

# Load plant data from the CSV file into a DataFrame
plant_data_df = pd.read_csv(csv_file_path)

# Convert the 'Plant name' column to lowercase in the DataFrame
plant_data_df['Plant name'] = plant_data_df['Plant name'].str.lower()

# Assuming the column names are "Plant name," "Description," and "Uses" as you mentioned
plant_data = dict(zip(plant_data_df['Plant name'], plant_data_df[['Description', 'Uses']].values))

# Load and preprocess the image for model training
images = []
labels = []

for plant_class in os.listdir(dataset_dir):
    if plant_class == '.DS_Store':
        continue  # Skip the .DS_Store file

    class_dir = os.path.join(dataset_dir, plant_class)
    for image_file in os.listdir(class_dir):
        if image_file == '.DS_Store':
            continue  # Skip the .DS_Store file

        image_path = os.path.join(class_dir, image_file)
        img = cv2.imread(image_path)
        img = cv2.resize(img, (128, 128))
        img = img / 255.0
        images.append(img)
        labels.append(plant_class)
images = np.array(images)
labels = np.array(labels)

# Feature Extraction and Random Forest Classification

X_train, y_train = images, labels  # Use the entire dataset for simplicity

X_train_rf = X_train.reshape(X_train.shape[0], -1)

rf_classifier = RandomForestClassifier(n_estimators=100)
rf_classifier.fit(X_train_rf, y_train)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', message='No file part')

        file = request.files['file']

        if file.filename == '':
            return render_template('index.html', message='No selected file')

        if file:
            try:
                # Process the uploaded image
                image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
                image = cv2.resize(image, (128, 128))
                image = image / 255.0
                image_rf = image.reshape(1, -1)

                # Predict the plant name
                predicted_plant = rf_classifier.predict(image_rf)[0]

                # Convert the predicted plant name to lowercase
                predicted_plant = predicted_plant.lower()

                # Get the Description and Uses based on the predicted plant name
                plant_info = plant_data.get(predicted_plant, ('Description not available', 'Uses not available'))

                return render_template('index.html', predicted_plant=predicted_plant, 
                                       plant_description=plant_info[0], plant_uses=plant_info[1])

            except Exception as e:
                return render_template('index.html', message=f'Error: {str(e)}')

    return render_template('index.html')
@app.route('/favicon.ico')
def favicon():
    return '', 204

if __name__ == '__main__':
    app.run()