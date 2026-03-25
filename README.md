# Stock-Price-Prediction
# 📈 VNM Stock Price Prediction using LSTM

This project implements a Deep Learning model to forecast the closing prices of **Vinamilk (VNM)**, one of Vietnam's leading dairy companies, using historical data from 2013 to 2023.

## 🚀 Overview
The goal of this project is to leverage **Long Short-Term Memory (LSTM)** networks to capture temporal dependencies in stock market data and provide accurate price predictions.

## 🛠️ Methodology

### 1. Data Acquisition & Environment 📚
* **Framework**: Built using `TensorFlow/Keras` for deep learning architecture.
* **Data Handling**: Utilized `pandas` for CSV manipulation and `numpy` for array processing.
* **Normalization**: Applied `MinMaxScaler` to scale stock prices between [0, 1], ensuring faster model convergence.

### 2. Preprocessing & Feature Engineering 🧹
* **Cleaning**: Removed non-essential columns such as "Volume" and "% Change" to focus on price action.
* **Time-Series Sorting**: Converted the "Date" column to datetime format and sorted the dataset chronologically.
* **Dataset Splitting**: Divided data into a **Training Set** (first 1,500 records) and a **Test Set** (remaining records).
* **Sliding Window**: Implemented a windowing technique using the **50 previous days** to predict the next day's closing price.

### 3. Model Architecture 🏗️
The model is constructed using a `Sequential` API with the following layers:
* **LSTM Layer 1**: 128 units with `return_sequences=True` to feed into the next layer.
* **LSTM Layer 2**: 64 units to extract deeper temporal features.
* **Dropout (0.5)**: Integrated to prevent overfitting by randomly deactivating neurons during training.
* **Dense Layer**: A single output neuron representing the predicted closing price.
* **Compilation**: Optimized using the **Adam** algorithm and **Mean Absolute Error (MAE)** loss function.

### 4. Training & Checkpointing ⚙️
* **Model Checkpoint**: Configured to automatically save the best-performing model based on loss reduction.
* **Parameters**: Trained for **100 epochs** with a **batch size of 50**.

---

## 📊 Results & Evaluation

### Performance Metrics ✅
The model performance is evaluated using:
* **R2 Score**: To measure the goodness of fit.
* **MAPE (Mean Absolute Percentage Error)**: To quantify the prediction error as a percentage.
  <img width="1971" height="703" alt="image" src="https://github.com/user-attachments/assets/d381e2b0-83e1-4ac3-89e0-f06ded39d5a2" />

  <img width="987" height="225" alt="image" src="https://github.com/user-attachments/assets/8af96a1a-b1df-4ea3-866d-d4e879a14e32" />



The predictive power of the model is validated through the following key metrics:

* **High Accuracy Rate**: The model achieves a **98% accuracy rate** (based on $1 - MAPE$), indicating that the average prediction error is only **2%**.
* **Volatility Capture**: This demonstrates that the **LSTM architecture** effectively captures the historical price volatility of **VNM**.
* **Statistical Reliability**: Furthermore, the high **$R^2$ Score** confirms that the model explains most of the variance in the stock's movement.



### Visualization 🖼️
The project generates a comprehensive plot comparing:
* **Actual Prices** (Red Line)
* **Train Predictions** (Green Line)
* **Test Predictions** (Blue Line)

### Future Forecasting 🔮
* The model successfully predicts the **next day's price** immediately following the dataset's timeframe.
* A comparison table is provided to show the **Predicted Price** vs. the **Last Actual Price**.

---

## 📂 Project Structure
* `StockPredictVNM.ipynb`: Main Jupyter Notebook containing the source code.
* `Output.keras`: The saved best-performing model file.
* `Dataset/`: Folder containing the historical VNM CSV data.

## 📝 Requirements
* Python 3.x
* TensorFlow / Keras
* Pandas, Numpy, Matplotlib
* Scikit-learn
