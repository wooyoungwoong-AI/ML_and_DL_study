## Housing-Price-Forecast-Regression-Issues
A regression AI model that predicts median home prices with data such as crime rates and local tax rates in the suburbs of Boston in the mid-1970s.


### Reason for production
This artificial intelligence model was created to improve skills and gain experience in artificial intelligence development as a personal project.


### Dataset
The dataset used in the AI model has 506 data points, 404 of which are divided into training samples and 102 into test samples. There are 13 numerical characteristics in total. These characteristics include crime rate per person, average number of rooms per house, and highway accessibility.
The target is the median price of a house, in units of $1,000.

### model
Since the number of samples is small, we used a small model with two intermediate layers with 64 units. In general, the smaller the number of training data, the easier it is to overfit, so using a small model is one way to avoid overfitting.

```python
from tensorflow import keras
from tensorflow.keras import layers

def build_model():
  model = keras.Sequential([
      layers.Dense(64, activation="relu"),
      layers.Dense(64, activation="relu"),
      layers.Dense(1)
  ])
  model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
  return model
```

### Training Validation Using K-Layer Validation
In this model, it was an opportunity to practice K-layers cross-validation. K-layers cross-validation is a method of dividing the data into K divisions, creating each of the K models, training them in K - 1 divisions, and evaluating them in the remaining divisions. The model's validation score is the average of K validation scores.

```python
import numpy as np

k = 4
num_val_samples = len(train_data) // k
num_epochs = 100
all_scores = []
for i in range(k):
    print(f"Processing fold #{i}")
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
         train_data[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
         train_targets[(i + 1) * num_val_samples:]],
        axis=0)
    model = build_model()
    model.fit(partial_train_data, partial_train_targets,
              epochs=num_epochs, batch_size=16, verbose=0)
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)
```

---

#### Colab link
<https://colab.research.google.com/drive/1oebRkye3-sDO3GdFAXUiTLcHnpk3exnw?usp=drive_link>
