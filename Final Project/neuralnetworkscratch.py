# Reimplementation of without PyTorch
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from data_loader import ImageLabelDataset

# --- Hyperparameters and Constants ---
DIGIT_IMG_HEIGHT = 28
DIGIT_IMG_WIDTH = 28
FACE_IMG_HEIGHT = 70
FACE_IMG_WIDTH = 60

INPUT_SIZE_DIGITS = DIGIT_IMG_HEIGHT * DIGIT_IMG_WIDTH
INPUT_SIZE_FACES = FACE_IMG_HEIGHT * FACE_IMG_WIDTH

HIDDEN1_SIZE = 128
HIDDEN2_SIZE = 64
INITIAL_LEARNING_RATE = 0.01
BATCH_SIZE = 64
NUM_EPOCHS = 30
MOMENTUM = 0.9

NUM_CLASSES_DIGITS = 10
NUM_CLASSES_FACES = 2

TRAINING_PERCENTAGES = np.arange(0.1, 1.1, 0.1)
NUM_RUNS_PER_PERCENTAGE = 5

def one_hot(y, num_classes):
    return np.eye(num_classes)[y]

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

def cross_entropy_loss(predictions, targets):
    m = targets.shape[0]
    log_likelihood = -np.log(predictions[range(m), targets] + 1e-9)
    return np.sum(log_likelihood) / m

def compute_accuracy(preds, labels):
    return np.mean(preds == labels)

class ThreeLayerNet:
    def __init__(self, input_size, hidden1_size, hidden2_size, num_classes):
        self.W1 = np.random.randn(input_size, hidden1_size) * np.sqrt(2. / input_size)
        self.b1 = np.zeros((1, hidden1_size))
        self.W2 = np.random.randn(hidden1_size, hidden2_size) * np.sqrt(2. / hidden1_size)
        self.b2 = np.zeros((1, hidden2_size))
        self.W3 = np.random.randn(hidden2_size, num_classes) * np.sqrt(2. / hidden2_size)
        self.b3 = np.zeros((1, num_classes))

        self.vW1 = np.zeros_like(self.W1)
        self.vb1 = np.zeros_like(self.b1)
        self.vW2 = np.zeros_like(self.W2)
        self.vb2 = np.zeros_like(self.b2)
        self.vW3 = np.zeros_like(self.W3)
        self.vb3 = np.zeros_like(self.b3)

    def forward(self, x):
        self.Z1 = x.dot(self.W1) + self.b1
        self.A1 = relu(self.Z1)
        self.Z2 = self.A1.dot(self.W2) + self.b2
        self.A2 = relu(self.Z2)
        self.Z3 = self.A2.dot(self.W3) + self.b3
        self.output = softmax(self.Z3)
        return self.output

    def backward(self, x, y, lr):
        m = y.shape[0]
        y_one_hot = one_hot(y, self.output.shape[1])

        dZ3 = self.output - y_one_hot
        dW3 = self.A2.T.dot(dZ3) / m
        db3 = np.sum(dZ3, axis=0, keepdims=True) / m

        dA2 = dZ3.dot(self.W3.T)
        dZ2 = dA2 * relu_derivative(self.Z2)
        dW2 = self.A1.T.dot(dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        dA1 = dZ2.dot(self.W2.T)
        dZ1 = dA1 * relu_derivative(self.Z1)
        dW1 = x.T.dot(dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        self.vW3 = MOMENTUM * self.vW3 - lr * dW3
        self.vb3 = MOMENTUM * self.vb3 - lr * db3
        self.vW2 = MOMENTUM * self.vW2 - lr * dW2
        self.vb2 = MOMENTUM * self.vb2 - lr * db2
        self.vW1 = MOMENTUM * self.vW1 - lr * dW1
        self.vb1 = MOMENTUM * self.vb1 - lr * db1

        self.W3 += self.vW3
        self.b3 += self.vb3
        self.W2 += self.vW2
        self.b2 += self.vb2
        self.W1 += self.vW1
        self.b1 += self.vb1

def batch_generator(data, labels, batch_size):
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    for start in range(0, len(data), batch_size):
        end = start + batch_size
        batch_idx = indices[start:end]
        yield data[batch_idx], labels[batch_idx]

def train_model(model, train_data, train_labels, num_epochs):
    start = time.time()
    lr = INITIAL_LEARNING_RATE
    for epoch in range(num_epochs):
        if epoch != 0 and epoch % 10 == 0:
            lr *= 0.5  # decay learning rate every 10 epochs
        for X_batch, y_batch in batch_generator(train_data, train_labels, BATCH_SIZE):
            model.forward(X_batch)
            model.backward(X_batch, y_batch, lr)
    return time.time() - start

def evaluate_model(model, test_data, test_labels):
    predictions = model.forward(test_data)
    pred_classes = np.argmax(predictions, axis=1)
    return 1 - compute_accuracy(pred_classes, test_labels)

def run_experiment(task_name, train_dataset, test_dataset, input_size, num_classes):
    print(f"\n--- Running Experiment for: {task_name} ---")
    results_time = {}
    results_error_mean = {}
    results_error_std = {}

    X_test, y_test = test_dataset[:]
    X_test = X_test.numpy()
    y_test = y_test.numpy()

    for percent in TRAINING_PERCENTAGES:
        subset_size = int(len(train_dataset) * percent)
        print(f"\nTraining on {percent*100:.0f}% of data ({subset_size} samples)")

        percent_times = []
        percent_errors = []

        for _ in range(NUM_RUNS_PER_PERCENTAGE):
            idx = np.random.choice(len(train_dataset), subset_size, replace=False)
            X_train, y_train = train_dataset.image_data[idx].numpy(), train_dataset.label_data[idx].numpy()
            model = ThreeLayerNet(input_size, HIDDEN1_SIZE, HIDDEN2_SIZE, num_classes)
            t = train_model(model, X_train, y_train, NUM_EPOCHS)
            error = evaluate_model(model, X_test, y_test)
            percent_times.append(t)
            percent_errors.append(error)

        results_time[percent] = np.mean(percent_times)
        results_error_mean[percent] = np.mean(percent_errors)
        results_error_std[percent] = np.std(percent_errors)

        print(f"Avg Time: {results_time[percent]:.2f}s, Error: {results_error_mean[percent]:.4f}, Std: {results_error_std[percent]:.4f}")

    return results_time, results_error_mean, results_error_std

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))

    digit_train_dataset = ImageLabelDataset(
        os.path.join(base_dir, 'digitdata/trainingimages'),
        os.path.join(base_dir, 'digitdata/traininglabels'),
        DIGIT_IMG_HEIGHT, DIGIT_IMG_WIDTH)

    digit_test_dataset = ImageLabelDataset(
        os.path.join(base_dir, 'digitdata/testimages'),
        os.path.join(base_dir, 'digitdata/testlabels'),
        DIGIT_IMG_HEIGHT, DIGIT_IMG_WIDTH)

    face_train_dataset = ImageLabelDataset(
        os.path.join(base_dir, 'facedata/facedatatrain'),
        os.path.join(base_dir, 'facedata/facedatatrainlabels'),
        FACE_IMG_HEIGHT, FACE_IMG_WIDTH)

    face_test_dataset = ImageLabelDataset(
        os.path.join(base_dir, 'facedata/facedatatest'),
        os.path.join(base_dir, 'facedata/facedatatestlabels'),
        FACE_IMG_HEIGHT, FACE_IMG_WIDTH)

    digit_times, digit_errors, digit_stds = run_experiment("Digit Classification",
                                                           digit_train_dataset,
                                                           digit_test_dataset,
                                                           INPUT_SIZE_DIGITS,
                                                           NUM_CLASSES_DIGITS)

    face_times, face_errors, face_stds = run_experiment("Face Detection",
                                                        face_train_dataset,
                                                        face_test_dataset,
                                                        INPUT_SIZE_FACES,
                                                        NUM_CLASSES_FACES)

    # Plotting results
    percentages_100 = [p * 100 for p in TRAINING_PERCENTAGES]

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(percentages_100, list(digit_times.values()), marker='o', label='Digits')
    plt.plot(percentages_100, list(face_times.values()), marker='s', label='Faces')
    plt.xlabel('Training Data (%)')
    plt.ylabel('Time (s)')
    plt.title('Training Time vs Data Size')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.errorbar(percentages_100, list(digit_errors.values()), yerr=list(digit_stds.values()), fmt='-o', label='Digits', capsize=4)
    plt.errorbar(percentages_100, list(face_errors.values()), yerr=list(face_stds.values()), fmt='-s', label='Faces', capsize=4)
    plt.xlabel('Training Data (%)')
    plt.ylabel('Error Rate')
    plt.title('Error vs Data Size (with Std Dev)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('numpy_nn_performance_curves.png')
    plt.show()
