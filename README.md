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

The project uses a comprehensive crocodile dataset (`crocodile_dataset.csv`) containing **1,000+ observations** with the following features:

### Features
- **Physical Characteristics:**
  - Observed Length (m)
  - Observed Weight (kg)
  - Age Class (Hatchling, Juvenile, Subadult, Adult)
  - Sex (Male, Female, Unknown)

- **Taxonomic Information:**
  - Scientific Name
  - Family
  - Genus

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
3. **Conservation Status** - Important ecological factor
4. **Geographic Location** - Regional distribution patterns

## 📊 Visualizations

The notebook includes comprehensive visualizations:

- **Feature Importance Plot** - Bar chart showing model feature rankings
- **Confusion Matrix** - Classification performance heatmap
- **Data Distribution Analysis** - Species weight and size distributions
- **Correlation Heatmaps** - Feature relationship analysis

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
- Feature importance rankings

## 📂 Project Structure

```
Gator-Classification/
├── crocodile_dataset.csv           # Main dataset
├── Crocodile_Random_Forest.ipynb   # Main analysis notebook
├── README.md                       # Project documentation
├── LICENSE                         # MIT License
└── requirements.txt                # Python dependencies
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Areas for Contribution:
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

## 🙏 Acknowledgments

- Wildlife conservation organizations for data collection efforts
- Scientific community for crocodilian research
- Open-source contributors to the libraries used in this project

## 📚 References

- Crocodilian species identification guides
- Conservation status databases
- Machine learning best practices for ecological data

---

*This project is part of the GatorTalk initiative aimed at advancing crocodilian research through machine learning and data science.*
