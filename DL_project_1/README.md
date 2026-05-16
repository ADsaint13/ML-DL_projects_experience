# 🩺 Breast Cancer Classification with Neural Networks

This project implements a Deep Learning Neural Network to classify breast cancer tumors as either **Malignant** or **Benign** based on cytology data (features extracted from fine-needle aspirates of breast masses). It includes a Jupyter Notebook for exploratory data analysis and model training, as well as a fully interactive Streamlit web application for real-time predictions.

## 🚀 Project Workflow

The project is structured into two main phases:

### Phase 1: Model Training (Jupyter Notebook)
1. **Data Loading**: We load the Breast Cancer Wisconsin (Diagnostic) dataset from `sklearn.datasets`. The dataset contains 30 numerical features (mean, standard error, and worst dimensions of cell nuclei).
2. **Data Preprocessing**: A `StandardScaler` is fitted to the training data to normalize the features, ensuring the Neural Network converges effectively.
3. **Neural Network Architecture**: A feedforward Artificial Neural Network (ANN) is built using **TensorFlow/Keras**, consisting of:
   - A Flatten input layer (30 features).
   - A Dense hidden layer with 20 neurons (ReLU activation).
   - A Dense output layer with 2 neurons (Sigmoid activation) for binary classification.
4. **Exporting Artifacts**: The fully trained model is saved natively as `breast_cancer_model.keras`. The fitted scaler is exported as `scaler.pkl` using the `pickle` library so the Streamlit app can accurately standardize incoming user data.

### Phase 2: Web Application Deployment (Streamlit)
The `app.py` script serves as the frontend dashboard:
- It securely loads the pre-trained Keras model (`breast_cancer_model.keras`) and the StandardScaler (`scaler.pkl`).
- It generates a sleek sidebar UI where users can tweak the 30 tumor features. To improve user experience, all 30 inputs are grouped into collapsible sections and pre-filled with the dataset averages.
- When the user clicks **Predict**, the data is standardized through the loaded scaler, passed through the neural network, and the probability/confidence score of the diagnosis (Malignant vs. Benign) is instantly displayed on the screen.

---

## 🛠️ How to Use This Project

### 1. Prerequisites
Ensure you have the required Python libraries installed:
```bash
pip install pandas numpy scikit-learn tensorflow streamlit
```

### 2. Training the Model (Optional)
If you wish to retrain the model, open the Jupyter Notebook:
```bash
jupyter notebook "Breast cancer classification with NN.ipynb"
```
Execute all the cells. The notebook will automatically train the model and export the `breast_cancer_model.keras` and `scaler.pkl` files to the project directory.

### 3. Running the Streamlit Web App
To launch the interactive frontend dashboard, run the following command in your terminal:
```bash
streamlit run app.py
```
This will start a local server. Open the provided Local URL (typically `http://localhost:8501`) in your web browser. 

### 4. Making Predictions
1. On the web app, navigate to the **Input Features** sidebar.
2. Expand the *Mean*, *Error*, and *Worst* feature categories to adjust the numeric values of the tumor cytology.
3. Click the **🔍 Predict Tumor Type** button.
4. View the diagnosis and confidence score displayed dynamically on the main screen!
