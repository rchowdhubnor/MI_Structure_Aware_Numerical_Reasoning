import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Load data
datasame = np.load()
DataT = datasame.astype(int)
activation = np.load()
activation27 = activation[:, 1, :, :]   # shape: (5000, 32, 4096)
activationLast = activation[:, -1, :, :]  # control condition: (5000, 32, 4096)

print("Activation shapes:", activation.shape, activation27.shape, activationLast.shape)

# Extract scalar labels (last first difference)
tkindlst = list(range(7, 14, 2))
toknext = tkindlst[-1]
labels = np.zeros(5000)
for ind in range(5000):
    DiffIndices = np.diff(DataT[0, ind, 0:(16 + toknext + 1)])
    labels[ind] = DiffIndices[-1]

# Prepare storage
mae_exp, r2_exp = np.zeros(32), np.zeros(32)
mae_ctrl, r2_ctrl = np.zeros(32), np.zeros(32)

# Train & evaluate per layer
for layer in range(32):
    # Experimental input: token 27
    X_exp = activation27[:, layer, :]
    # Control input: last token
    X_ctrl = activationLast[:, layer, :]
    y = labels

    # Train/test split
    X_train_exp, X_test_exp, y_train, y_test = train_test_split(X_exp, y, test_size=0.1, random_state=42)
    X_train_ctrl, X_test_ctrl, _, _ = train_test_split(X_ctrl, y, test_size=0.1, random_state=42)

    # Experimental probe
    model_exp = LinearRegression()
    model_exp.fit(X_train_exp, y_train)
    y_pred_exp = model_exp.predict(X_test_exp)
    mae_exp[layer]=mean_absolute_error(y_test, y_pred_exp)
    r2_exp[layer]=r2_score(y_test, y_pred_exp)

    # Control probe
    model_ctrl = LinearRegression()
    model_ctrl.fit(X_train_ctrl, y_train)
    y_pred_ctrl = model_ctrl.predict(X_test_ctrl)
    mae_ctrl[layer]=mean_absolute_error(y_test, y_pred_ctrl)
    r2_ctrl[layer]=r2_score(y_test, y_pred_ctrl)

