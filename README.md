# 🐊 Crocodile Species Classification

A machine learning project that classifies crocodile species using Random Forest algorithm based on physical characteristics, habitat, and geographic data.

## 📊 Project Overview

This project implements a Random Forest classifier to predict crocodile species (Common Name) based on various features including physical measurements, geographic location, habitat type, and conservation status. The model aims to assist wildlife researchers and conservationists in species identification and ecological studies.

## 🎯 Objectives

- Develop an accurate machine learning model for crocodile species classification
- Analyze feature importance to understand which characteristics are most predictive
- Provide data visualization and exploratory analysis of crocodile populations
- Create a reproducible pipeline for species classification

## 📁 Dataset Description

The project uses a comprehensive [crocodile dataset](https://www.kaggle.com/datasets/zadafiyabhrami/global-crocodile-species-dataset)
 (`crocodile_dataset.csv`) containing **1,000+ observations** with the following features:

### Features
- **Physical Characteristics:**
  - Observed Length (m)
  - Observed Weight (kg)
  - Age Class (Hatchling, Juvenile, Subadult, Adult)
  - Sex (Male, Female, Unknown)

- **Geographic & Environmental:**
  - Country/Region
  - Habitat Type (Rivers, Swamps, Mangroves, etc.)
  - Date of Observation

- **Conservation:**
  - Conservation Status (Least Concern, Vulnerable, Endangered, etc.)

### Target Variable
- **Common Name** - The species to be predicted

## 🛠️ Technologies Used

- **Python 3.x**
- **Data Science Libraries:**
  - `pandas` - Data manipulation and analysis
  - `numpy` - Numerical computing
  - `scikit-learn` - Machine learning algorithms
  - `matplotlib` - Data visualization
  - `seaborn` - Statistical visualization

- **Machine Learning:**
  - Random Forest Classifier
  - One-Hot Encoding for categorical features
  - Pipeline preprocessing
  - Train-test split with stratification

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/RyanWoodzell/Gator-Classification.git
cd Gator-Classification
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Launch Jupyter Notebook:
```bash
jupyter notebook Crocodile_Random_Forest.ipynb
```

## 📈 Model Performance

The Random Forest classifier is configured with:
- **200 estimators** for robust predictions
- **Stratified sampling** to maintain class balance
- **80/20 train-test split**
- **One-hot encoding** for categorical variables

### Key Features by Importance:
1. **Observed Weight (kg)** - Most predictive feature
2. **Observed Length (m)** - Strong physical indicator
3. **Geographic Location** - Regional distribution patterns

## 📊 Visualizations

The notebook includes comprehensive visualizations:
### Initial Data Exploration
<img width="528" height="332" alt="image" src="https://github.com/user-attachments/assets/724bfbf0-bd8d-4ddb-bdef-d0e7d5b95009" />

### Confusion Matrix - Classification Performance
<img width="1044" height="1045" alt="image" src="https://github.com/user-attachments/assets/aa122b16-cbb9-42ef-8b61-7820f8da6c7c" />


### Model Confusion Analysis
My model frequently confuses the following species: 
- New Guinea vs. Hall's New Guinea
- Nile vs Mugger (Marsh)
I wanted to take a deeper look into this confusion, so I made a function to compare two types of crocodile using various matplotlib graphs. Here is an example of the New Guinea vs. Hall's New Guinea Comparison: 
<img width="1721" height="1142" alt="image" src="https://github.com/user-attachments/assets/7d8b752d-a378-4a28-962d-f09e6ed1d3a3" />

It is eviden that Weight and Length are very similar between these species, likely contributing to the confusion. In my feature importance analysis, I also uncovered that Weight and Length were the most influential features for the models predictions:
<img width="1142" height="961" alt="image" src="https://github.com/user-attachments/assets/1b407a7b-9849-4e9d-a719-30bb42a15985" />
Based on this insight, I plan to do the followwing:
- Model Fine Tuning: Adjusting hyperparameters to reduce misclassifcation
- Look to Alternative Models: I would like to combine random forest with other classifiers to better seperate mistaken species.
## 🔬 Methodology

### 1. Data Preprocessing
- Date feature extraction (Year, Month, Season)
- Age class numerical mapping
- Missing value handling
- Feature scaling and encoding

### 2. Feature Engineering
- Temporal features from observation dates
- Categorical encoding with One-Hot Encoder
- Feature selection and importance analysis

### 3. Model Training
- Random Forest with hyperparameter tuning
- Pipeline implementation for reproducibility
- Cross-validation and performance evaluation

### 4. Evaluation Metrics
- Accuracy score
- Classification report (Precision, Recall, F1-score)
- Confusion matrix analysis

## 📂 Project Structure

```
Gator-Classification/
├── crocodile_dataset.csv           # Main dataset
├── Crocodile_Random_Forest.ipynb   # Main analysis notebook
├── README.md                       # Project documentation
├── LICENSE                         # MIT License
└── requirements.txt                # Python dependencies
```

## Improvements
As I contintinue to progress my knowledge in machine learning, I plan on circling back to this project and implementing different models to test and see if I can up my model's accuracy

### Areas for Improvement:
- Additional machine learning algorithms (SVM, Neural Networks)
- Hyperparameter optimization
- Feature engineering improvements
- Data visualization enhancements
- Model deployment pipeline

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Ryan Woodzell**
- GitHub: [@RyanWoodzell](https://github.com/RyanWoodzell)



