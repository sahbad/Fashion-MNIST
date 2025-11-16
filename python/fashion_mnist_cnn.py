#importing libraries and defining the class skeleton
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class FashionMNISTClassifier:
    def __init__(self):
        # load data
        # preprocess
        # build model
        pass

    def _preprocess(self):
        pass

    def _build_model(self):
        pass

    def train(self, epochs=5, batch_size=64):
        pass

    def evaluate(self):
        pass

    def predict_samples(self, indices=None):
        pass

#Implement data loading in __init__:
class FashionMNISTClassifier:
    def __init__(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = keras.datasets.fashion_mnist.load_data()

        # Optional: keep class names as attribute
        self.class_names = [
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
        ]

        self._preprocess()
        self.model = self._build_model()

#Implement _preprocess:
def _preprocess(self):
    # Convert to float and normalize to [0, 1]
    self.x_train = self.x_train.astype("float32") / 255.0
    self.x_test = self.x_test.astype("float32") / 255.0

    # Reshape to (28, 28, 1)
    self.x_train = self.x_train.reshape((-1, 28, 28, 1))
    self.x_test = self.x_test.reshape((-1, 28, 28, 1))

#Implement _build_model with exactly 6 layers (excluding input):
def _build_model(self):
    # 6-layer CNN: Conv → Pool → Conv → Pool → Flatten → Dense
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),  # 1
        layers.MaxPooling2D((2, 2)),                                             # 2
        layers.Conv2D(64, (3, 3), activation="relu"),                           # 3
        layers.MaxPooling2D((2, 2)),                                             # 4
        layers.Flatten(),                                                       # 5
        layers.Dense(10, activation="softmax")                                  # 6
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

#Implement train method:
def train(self, epochs=5, batch_size=64):
    history = self.model.fit(
        self.x_train,
        self.y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=2
    )
    return history
#Implement evaluate method:
def evaluate(self):
    test_loss, test_acc = self.model.evaluate(self.x_test, self.y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")
    return test_loss, test_acc

#Implement predict_samples for at least 2 images:
def predict_samples(self, indices=None):
    if indices is None:
        indices = [0, 1]

    images = self.x_test[indices]
    labels = self.y_test[indices]

    predictions = self.model.predict(images)
    predicted_classes = predictions.argmax(axis=1)

    for idx, true_label, pred_label in zip(indices, labels, predicted_classes):
        print(f"Image index: {idx}")
        print(f"  True label:      {self.class_names[true_label]}")
        print(f"  Predicted label: {self.class_names[pred_label]}")
        print("-" * 40)

    return predicted_classes

#Add the script entry point:
if __name__ == "__main__":
    classifier = FashionMNISTClassifier()
    classifier.train(epochs=5, batch_size=64)
    classifier.evaluate()
    classifier.predict_samples(indices=[0, 1])
