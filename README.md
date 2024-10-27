# Advancing Ensemble Learning Models for Residential Building Electricity Consumption Forecasting

Welcome to the repository accompanying our paper: **"Advancing Ensemble Learning Models for Residential Building Electricity Consumption Forecasting"** by **J. Moon et al., 2024**. This repository provides the datasets and code necessary to reproduce the results presented in the paper and to facilitate further research in the field of residential building electricity consumption forecasting.

## Overview

Accurate forecasting of electricity consumption in residential buildings is crucial for energy management and planning. This repository contains:

- **Datasets**: Two comprehensive datasets for residential building electricity consumption forecasting.
- **Notebooks**: Jupyter notebooks implementing ensemble learning models and deep learning models.

## Repository Structure

```plaintext
PLoS_ONE/
├── Appliances Energy Prediction.csv
├── University Residential Complex.csv
├── ensemble_models_notebook.ipynb
├── deep_learning_energy_prediction.ipynb
├── explainable_AI_energy_consumption_models.ipynb
├── README.md
```

- The previous folders (`code`, `data`, `predict`) can be ignored as they have been replaced with updated datasets and notebooks.

## Datasets

The datasets used in our study are included in this repository:

1. **Appliances Energy Prediction.csv**
   - **Description**: Contains energy consumption data from household appliances. This dataset is instrumental in understanding the consumption patterns within a household.
   - **Notes**: Detailed information about this dataset is provided in **Table 1** of our paper.

2. **University Residential Complex.csv**
   - **Description**: Includes energy consumption data from a university residential complex. This dataset helps in analyzing consumption patterns in a larger residential setting.
   - **Notes**: Detailed information about this dataset is also provided in **Table 1** of our paper.

## Notebooks

### 1. ensemble_models_notebook.ipynb

- **Purpose**: This notebook selects the optimal hyperparameters for five ensemble learning models: Random Forest, GBM, XGBoost, LightGBM, and CatBoost. It calculates evaluation metrics such as MAPE, CVRMSE, NMAE, and HM on the evaluation set, presenting the metrics from the perspectives of weekdays, holidays, and overall.
- **What You'll Learn**:
  - How to tune hyperparameters for ensemble models.
  - How to evaluate models using various metrics.
  - How to analyze model performance across different time periods.
- **Tips**:
  - **Dependencies**: Ensure you have the required libraries installed (`scikit-learn`, `xgboost`, `lightgbm`, `catboost`, etc.).
  - **Execution Order**: Run the cells sequentially to avoid any dependency issues.
  - **Understanding Outputs**: Pay attention to the evaluation metrics provided at the end of the notebook to compare model performances.

### 2. deep_learning_energy_prediction.ipynb

- **Purpose**: This notebook explores benchmark deep learning models for energy consumption forecasting, including LSTM, Bi-LSTM, GRU, Bi-GRU, 1D-CNN, and TCN. Additionally, we explore hybrid models such as LSTM-TCN, BiLSTM-TCN, GRU-TCN, and BiGRU-TCN.
- **What You'll Learn**:
  - How to implement and train different types of deep learning models for time series forecasting.
  - How to handle data preprocessing for deep learning models.
  - How to evaluate deep learning models using relevant metrics.
- **Tips**:
  - **Environment Setup**: Deep learning models may require specific versions of libraries like TensorFlow or PyTorch. Ensure your environment is configured correctly.
  - **Package Compatibility**: Due to rapid updates in deep learning libraries, you might encounter compatibility issues. Be prepared to adjust the code or install specific library versions.
    - For example, you might need to install TensorFlow 2.x or PyTorch 1.x depending on the code.
  - **Hardware Requirements**: Training deep learning models can be resource-intensive. If possible, use a machine with a GPU.
  - **Updates**: There may be updates or slight code modifications due to the deep learning environment and package compatibility issues. Readers should be aware of this.

### 3. explainable_AI_energy_consumption_models.ipynb

- **Purpose**: This notebook applies VIP, PDP, and SHAP analyses on the ensemble learning models configured with optimal hyperparameters. The results allow for interpretation of the models, and users can adjust options as needed to interpret the models in their preferred way.
- **What You'll Learn**:
  - How to interpret ensemble learning models using XAI techniques.
  - How to generate and analyze VIP, PDP, and SHAP plots.
  - How to derive insights from model interpretations to understand the factors influencing electricity consumption.
- **Tips**:
  - **Dependencies**: Ensure you have `shap`, `matplotlib`, `seaborn`, and other necessary libraries installed.
  - **Customization**: Feel free to adjust the plots and parameters to focus on variables of interest in your analysis.
  - **Model Interpretation**: Use the insights from the XAI techniques to inform energy management strategies or further research.

## Getting Started

To get the most out of this repository, we recommend the following steps:

1. **Clone the Repository**:

```plaintext
git clone https://github.com/sodayeong/PLoS_ONE.git
```


Ensure Datasets Are in Place:

Both datasets are included in the repository. They should be in the same directory as the notebooks.
Run the Notebooks:

Order of Execution: We recommend running the notebooks in the following order for optimal understanding and setup:
- ensemble_models_notebook.ipynb
- deep_learning_energy_prediction.ipynb
- explainable_AI_energy_consumption_models.ipynb

Note: Ensure you run the notebooks in order to properly set up the environment and dependencies.

## Important Notes

- **Data Privacy**: When using real-world datasets, ensure compliance with data privacy laws and regulations.

- **Citations**: If you find this repository helpful, please consider citing our paper:

```plaintext
J. Moon et al., "Advancing Ensemble Learning Models for Residential Building Electricity Consumption Forecasting," PLOS ONE, 2024. (Please fill in volume, issue, and page numbers when available.)
```


## Issues and Contributions

- If you encounter any issues or have suggestions, feel free to open an issue on GitHub.
- Contributions to improve the code or add new features are welcome.

## Additional Tips

- **Understanding the Models**: Take the time to understand how each model works and how the hyperparameters affect performance.
- **Experimentation**: Don't hesitate to experiment with different settings or models to see how they impact the results.
- **Learning Resources**: If you're new to ensemble learning or XAI techniques, consider reviewing tutorials or courses to strengthen your understanding.
- **Model Interpretation**: The XAI notebook allows you to interpret models in your preferred way. Adjust the code and options to focus on the aspects most relevant to your research or application.

## Contact

For any questions or inquiries, please contact:

- **First Author**: Jihoon Moon, Ph.D.
- **Email**: johnnyone89@gmail.com
- **Institution**: Soonchunhyang University

By providing comprehensive datasets, detailed code, and guidance, we aim to support researchers and practitioners in advancing the field of residential building electricity consumption forecasting. We hope this repository serves as a valuable resource for your work.

## Disclaimer

Due to potential updates in software packages and dependencies, there might be slight differences in results or required adjustments in the code. We encourage users to verify their environment and make necessary changes.

Thank you for your interest in our work!
