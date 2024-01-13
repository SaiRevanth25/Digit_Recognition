import tensorflow as tf
import sklearn as sk
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from Model import x_train, y_train


model = tf.keras.models.load_model('model.h5')

config = model.get_config() # Returns pretty much every information about your model
print(config["layers"][0]["config"]["batch_input_shape"])

data, labels = x_train, y_train

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.35, random_state=42)
labelNames = "0123456789"
labelNames = [l for l in labelNames]    

predictions = model.predict(X_test, batch_size=128)
print(classification_report(y_test, predictions.argmax(axis=1), target_names=labelNames))