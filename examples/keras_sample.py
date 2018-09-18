import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import ngraph

model = ResNet50(weights='imagenet')

img = np.random.rand(1,224,224,3)
preds = model.predict(preprocess_input(img))
print('Predicted:', decode_predictions(preds, top=3)[0])
