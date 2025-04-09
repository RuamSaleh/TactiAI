# Tactical Decision Predictor

This project aims to predict the tactical decisions made during tennis matches using machine learning techniques. The dataset used for training and testing the model is derived from various tennis matches, containing features related to player performance and match statistics.

## Project Structure

```
tactical-decision-predictor
├── data
│   └── tennis_matches_with_tactics.csv  # Dataset containing match statistics and tactical decisions
├── src
│   ├── data_preprocessing.py             # Data loading, cleaning, and preparation functions
│   ├── train_model.py                     # Code to train the machine learning model
│   ├── evaluate_model.py                  # Functions to evaluate model performance
│   └── utils.py                           # Utility functions for visualization and model handling
├── models
│   └── model.pkl                          # Saved trained model
├── notebooks
│   └── exploratory_analysis.ipynb         # Jupyter notebook for exploratory data analysis
├── requirements.txt                       # List of required Python packages
├── .gitignore                             # Files and directories to ignore in Git
└── README.md                              # Project documentation
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd tactical-decision-predictor
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```


3. Accuracy: 60%

## Usage

1. **Data Preprocessing**: Run the `data_preprocessing.py` script to load and preprocess the dataset.
2. **Model Training**: Execute the `train_model.py` script to train the model using the preprocessed data.
3. **Model Evaluation**: Use the `evaluate_model.py` script to assess the model's performance on the test dataset.
4. **Exploratory Analysis**: Open the `exploratory_analysis.ipynb` notebook for insights and visualizations of the dataset.

## License

This project is licensed under the MIT License.
## Resources
- https://www.tennisabstract.com/blog/2019/12/03/an-introduction-to-tennis-elo/
- https://github.com/JeffSackmann/tennis_misc/blob/master/fiveSetProb.py
- https://github.com/JeffSackmann/tennis_atp
