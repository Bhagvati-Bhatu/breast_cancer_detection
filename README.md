# Breast Cancer Detection 🩺🔬

This project demonstrates **Breast Cancer Detection** using machine learning algorithms on the **Breast Cancer Wisconsin Dataset** (UCI Repository).  
The notebook walks through data preprocessing, exploratory analysis, model training, and evaluation.

---

## 📂 Dataset
- **Source:** [UCI Breast Cancer Wisconsin Dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/)  
- **Features:**
  - Clump thickness
  - Uniform cell size & shape
  - Marginal adhesion
  - Single epithelial cell size
  - Bare nuclei
  - Bland chromatin
  - Normal nucleoli
  - Mitoses
- **Target:**  
  - `2` → Benign  
  - `4` → Malignant  

---

## ⚙️ Project Workflow
1. **Import Libraries** → NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn  
2. **Load Dataset** → From UCI repository  
3. **Data Preprocessing**  
   - Dropping irrelevant columns (`id`)  
   - Handling missing values  
   - Label encoding for target class  
4. **Exploratory Data Analysis (EDA)**  
   - Shape and distribution of data  
   - Visualization of features  
5. **Model Training & Evaluation**  
   - Algorithms: Decision Tree, KNN, Naïve Bayes, SVM  
   - Evaluation using:
     - Accuracy
     - Confusion Matrix
     - Classification Report
     - Cross-validation (K-Fold, GridSearchCV for tuning)  

---

## 🛠️ Dependencies
Install required libraries:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

---

## 🚀 Usage
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/breast-cancer-detection.git
   ```
2. Open the Jupyter Notebook:
   ```bash
   jupyter notebook Breast_Cancer_Detection.ipynb
   ```
3. Run all cells to see preprocessing, model training, and evaluation.

---

## 📊 Results
- Multiple models are compared to identify the most accurate classifier.  
- Metrics include **Accuracy**, **Precision**, **Recall**, and **F1-score**.  
- The notebook concludes with the best-performing model for breast cancer classification.

---

## 📌 Future Work
- Implement deep learning models (e.g., Neural Networks).  
- Use feature selection for dimensionality reduction.  
- Deploy the model as a web app using Flask/Streamlit.  

---

## 🙌 Acknowledgments
- Dataset: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Original))  
- Libraries: Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn  
