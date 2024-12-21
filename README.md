
Human Activity Recognition using Mobile Phone Sensors
This program utilizes data from mobile phone sensors, specifically the gyroscope and accelerometer, to classify human activity as either walking or running. The classification model is evaluated using the Receiver Operating Characteristic (ROC) curve and the Area Under the Curve (AUC) score.

Data Collection
The program collects data from the mobile phone's gyroscope and accelerometer sensors. The gyroscope measures the device's orientation and rotation, while the accelerometer measures the device's acceleration.

Feature Extraction
The program extracts relevant features from the sensor data, including:

- Accelerometer features:
    - Mean acceleration
    - Standard deviation of acceleration
    - Peak acceleration
- Gyroscope features:
    - Mean angular velocity
    - Standard deviation of angular velocity
    - Peak angular velocity

Classification Model
The program uses a machine learning classification model to classify the human activity as either walking or running. The model is trained using the extracted features and evaluated using the ROC curve and AUC score.

ROC Curve and AUC Score
The program plots the ROC curve for each of the following scenarios:

1. All features (accelerometer and gyroscope)
2. Accelerometer features only
3. Gyroscope features only

The AUC score is calculated for each scenario, providing a quantitative measure of the model's performance.

Comparison of Accelerometer-Only and Gyroscope-Only Models
The program compares the performance of the accelerometer-only model and the gyroscope-only model. This comparison provides insights into the relative importance of each sensor modality in human activity recognition.

Code Implementation
The program is implemented using Python and utilizes libraries such as NumPy, Pandas, and Scikit-learn for data manipulation, feature extraction, and classification. The ROC curve and AUC score are calculated using the Scikit-learn library.

Example Use Cases
This program has various applications, including:

- Fitness tracking: The program can be used to track physical activity and provide personalized fitness recommendations.
- Healthcare monitoring: The program can be used to monitor patients with mobility issues or chronic conditions.
- Sports analytics: The program can be used to analyze athlete performance and provide insights for training and improvement.
