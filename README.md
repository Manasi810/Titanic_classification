# Titanic Survival Prediction

This project is a machine learning classification task aimed at predicting the survival of passengers aboard the Titanic. Using the Titanic dataset from Kaggle, the model classifies whether a passenger survived or not based on features like age, gender, ticket class, and more.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Data Preprocessing](#data-preprocessing)
4. [Feature Engineering](#feature-engineering)
5. [Model Selection](#model-selection)
6. [Evaluation](#evaluation)
7. [How to Run](#how-to-run)
8. [Results](#results)
9. [Contributing](#contributing)
10. [License](#license)

## Project Overview

The Titanic classification project utilizes various machine learning algorithms to predict the survival of passengers. The goal is to experiment with different models, compare their performance, and choose the one that best fits the problem.

## Dataset

The dataset used in this project is the famous Titanic dataset available on [Kaggle](https://www.kaggle.com/c/titanic/data). It contains data on 891 passengers with features such as:

- PassengerId: Unique identifier
- Pclass: Passenger class (1, 2, or 3)
- Name: Passenger name
- Sex: Gender
- Age: Age in years
- SibSp: Number of siblings or spouses aboard the Titanic
- Parch: Number of parents/children aboard the Titanic
- Ticket: Ticket number
- Fare: Passenger fare
- Cabin: Cabin number
- Embarked: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)
  
## Data Preprocessing

To handle missing data and prepare the dataset for machine learning models, several preprocessing steps were taken:

- Imputation of missing values in the 'Age' and 'Embarked' columns.
- Conversion of categorical features such as 'Sex' and 'Embarked' into numerical format using one-hot encoding.
- Feature scaling for continuous variables.

## Feature Engineering

Additional features were created to enhance the predictive power of the model, such as:

- Title extraction from the 'Name' column to create a new feature.
- Family size calculation by combining 'SibSp' and 'Parch' columns.
- Binning the 'Age' and 'Fare' columns to reduce noise.

## Model Selection

Several machine learning models were tested, including:

- Logistic Regression
- Support Vector Machines (SVM)
- Random Forest
- Gradient Boosting
- K-Nearest Neighbors (KNN)
  
Cross-validation was used to select the best model based on accuracy and other performance metrics like precision, recall, and F1 score.

## Evaluation

The model performance was evaluated using:

- Confusion Matrix
- Accuracy Score
- Precision, Recall, F1 Score
- ROC-AUC Curve

## How to Run

To run this project, follow these steps:

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/titanic-classification.git
    cd titanic-classification
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Run the Jupyter notebook or Python script for training and evaluation:

    ```bash
    jupyter notebook titanic_classification.ipynb
    ```

## Results

The best-performing model achieved an accuracy of XX% on the test dataset. Below is a summary of the model performance:

| Model                | Accuracy | Precision | Recall | F1 Score |
|----------------------|----------|-----------|--------|----------|
| Logistic Regression   | XX%      | XX%       | XX%    | XX%      |
| Random Forest         | XX%      | XX%       | XX%    | XX%      |
| Gradient Boosting     | XX%      | XX%       | XX%    | XX%      |

## Contributing

Contributions are welcome! Please create an issue or submit a pull request with your changes.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
