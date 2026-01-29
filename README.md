#  Sampling Techniques for Imbalanced Credit Card Dataset

##  Overview
In real-world machine learning applications, datasets are often highly imbalanced, which can significantly degrade model performance.  
This project analyzes the impact of different sampling techniques on various machine learning models using a credit card fraud detection dataset.

The primary focus is to understand how balancing strategies influence classification accuracy and model behavior.

---

##  Objective
- To handle class imbalance in a credit card dataset  
- To apply multiple sampling techniques for data balancing  
- To evaluate the performance of different machine learning models  
- To identify which sampling method performs best for each model  

---

##  Dataset Description
- **Dataset Name:** Creditcard_data.csv  
- **Source:** Provided through assignment repository  
- **Target Column:** `Class`  
  - `0` – Non-fraudulent transaction  
  - `1` – Fraudulent transaction  

The dataset is highly imbalanced, making it suitable for studying the effectiveness of resampling techniques.

---

##  Technologies Used
- **Programming Language:** Python  
- **Libraries:**
  - pandas  
  - scikit-learn  
  - imbalanced-learn  

---

##  Sampling Techniques Implemented
Five different sampling techniques were used:

| Sampling ID | Technique |
|------------|----------|
| Sampling1 | Random Under Sampling |
| Sampling2 | Random Over Sampling |
| Sampling3 | SMOTE |
| Sampling4 | ADASYN |
| Sampling5 | NearMiss |

These methods balance the dataset using under-sampling, over-sampling, and synthetic data generation approaches.

---

##  Machine Learning Models Used
The following classification models were trained and evaluated:

| Model ID | Algorithm |
|--------|-----------|
| M1 | Support Vector Machine (SVM) |
| M2 | Logistic Regression |
| M3 | Decision Tree |
| M4 | K-Nearest Neighbors (KNN) |
| M5 | Random Forest |

---

##  Methodology
1. Load and explore the original dataset  
2. Apply each sampling technique independently  
3. Split the balanced data into training and testing sets  
4. Train each machine learning model  
5. Evaluate performance using accuracy  
6. Compare results across models and sampling methods  

---

##  Results
The accuracy scores obtained are summarized below:

| Model | Sampling1 | Sampling2 | Sampling3 | Sampling4 | Sampling5 |
|------|----------|----------|----------|----------|----------|
| M1 | 50.10 | 52.24 | 63.18 | 69.23 | **70.12** |
| M2 | 59.25 | 65.27 | **68.72** | 28.36 | 30.25 |
| M3 | **90.45** | 72.41 | 32.17 | 42.58 | 41.85 |
| M4 | **78.25** | 56.24 | 47.23 | 33.44 | 40.12 |
| M5 | **81.25** | 12.85 | 57.36 | 32.25 | 52.74 |

---

##  Observations
- No single sampling technique performs best for all models  
- Under-sampling shows strong performance with ensemble models  
- Synthetic sampling methods perform better with linear classifiers  
- Model performance is highly dependent on the chosen sampling method  

---

##  Conclusion
This project highlights the importance of sampling techniques in handling imbalanced datasets.  
The results show that selecting an appropriate sampling strategy is model-dependent and crucial for improving classification performance.

---

##  Project Structure
Sampling_Assignment/
│
├── Creditcard_data.csv
├── sampling_creditcard.py
├── final_sampling_results.csv
└── README.md

## How to Run the Project
```bash
pip install pandas scikit-learn imbalanced-learn
python sampling_cc.py
