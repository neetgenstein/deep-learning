# pip install pycm

import seaborn as sns
import pandas as pd
from pycm import ConfusionMatrix
import matplotlib.pyplot as plt

confusion_matrix = {
    "Red Disease": {"Red Disease": 88, "Aeromoniasis": 4, "Gill Disease": 2, "Saprolegniasis": 3, "Healthy Fish": 0, "Parasitic Disease": 1, "White Tail Disease": 2},
    "Aeromoniasis": {"Red Disease": 2, "Aeromoniasis": 90, "Gill Disease": 5, "Saprolegniasis": 1, "Healthy Fish": 2, "Parasitic Disease": 0, "White Tail Disease": 0},
    "Gill Disease": {"Red Disease": 3, "Aeromoniasis": 1, "Gill Disease": 91, "Saprolegniasis": 0, "Healthy Fish": 1, "Parasitic Disease": 2, "White Tail Disease": 2},
    "Saprolegniasis": {"Red Disease": 6, "Aeromoniasis": 0, "Gill Disease": 1, "Saprolegniasis": 87, "Healthy Fish": 2, "Parasitic Disease": 4, "White Tail Disease": 0},
    "Healthy Fish": {"Red Disease": 2, "Aeromoniasis": 1, "Gill Disease": 0, "Saprolegniasis": 0, "Healthy Fish": 97, "Parasitic Disease": 0, "White Tail Disease": 0},
    "Parasitic Disease": {"Red Disease": 0, "Aeromoniasis": 4, "Gill Disease": 6, "Saprolegniasis": 2, "Healthy Fish": 3, "Parasitic Disease": 82, "White Tail Disease": 3},
    "White Tail Disease": {"Red Disease": 5, "Aeromoniasis": 1, "Gill Disease": 2, "Saprolegniasis": 1, "Healthy Fish": 4, "Parasitic Disease": 2, "White Tail Disease": 85}
}

df = pd.DataFrame(confusion_matrix).T  # Transpose for correct orientation
plt.figure(figsize=(25, 25))
sns.heatmap(df, annot=True, fmt="d", cmap="YlGnBu")
plt.title("Confusion Matrix for MobileNetV2 based Classifier Model")
plt.ylabel("Predicted Disease")
plt.xlabel("Actual Disease")
plt.show()
