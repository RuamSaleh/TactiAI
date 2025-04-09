def save_model(model, filename):
    import joblib
    joblib.dump(model, filename)

def load_model(filename):
    import joblib
    return joblib.load(filename)

def plot_feature_importances(importances, feature_names):
    import matplotlib.pyplot as plt
    import numpy as np

    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), np.array(feature_names)[indices], rotation=90)
    plt.xlim([-1, len(importances)])
    plt.tight_layout()
    plt.show()