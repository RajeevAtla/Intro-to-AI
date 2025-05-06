import numpy as np
import time
import os
import matplotlib.pyplot as plt  # For plotting results
from data_loader import ImageLabelDataset_Perceptron as ImageLabelDataset  # Import the custom dataset from data_loader.py

# --- 1. Constants and Hyperparameters ---
# TODO: Determine these from your data files!
DIGIT_IMG_HEIGHT = 28  # Example
DIGIT_IMG_WIDTH = 28  # Example
FACE_IMG_HEIGHT = 70  # Example
FACE_IMG_WIDTH = 60  # Example

# Calculate Input Size
INPUT_SIZE_DIGITS = DIGIT_IMG_HEIGHT * DIGIT_IMG_WIDTH
INPUT_SIZE_FACES = FACE_IMG_HEIGHT * FACE_IMG_WIDTH

# Model Hyperparameters (tune these)
LEARNING_RATE = 0.001
BATCH_SIZE = 64
NUM_EPOCHS = 20  # Adjust as needed

# Output sizes
NUM_CLASSES_DIGITS = 10
NUM_CLASSES_FACES = 2  # Using 2 output neurons and CrossEntropyLoss

# Data Paths (TODO: Update these after unzipping)
DATA_DIR = ''
# Assumes data.zip is unzipped into a 'data' directory
# Example paths - adjust based on actual file names/structure
DIGIT_TRAIN_IMAGES_PATH = os.path.join(DATA_DIR, 'digitdata/trainingimages')
DIGIT_TRAIN_LABELS_PATH = os.path.join(DATA_DIR, 'digitdata/traininglabels')
DIGIT_TEST_IMAGES_PATH = os.path.join(DATA_DIR, 'digitdata/testimages')
DIGIT_TEST_LABELS_PATH = os.path.join(DATA_DIR, 'digitdata/testlabels')

FACE_TRAIN_IMAGES_PATH = os.path.join(DATA_DIR, 'facedata/facedatatrain')
FACE_TRAIN_LABELS_PATH = os.path.join(DATA_DIR, 'facedata/facedatatrainlabels')
FACE_TEST_IMAGES_PATH = os.path.join(DATA_DIR, 'facedata/facedatatest')
FACE_TEST_LABELS_PATH = os.path.join(DATA_DIR, 'facedata/facedatatestlabels')

# Experiment parameters
TRAINING_PERCENTAGES = np.arange(0.1, 1.1, 0.1)  # 10% to 100%
NUM_RUNS_PER_PERCENTAGE = 5  # For calculating standard deviation


# --- 3. Model Definition ---
class Perceptron():
    def __init__(self, task_name):
        self.task_name = task_name

        if task_name == "Face Detection":
            input_height = FACE_IMG_HEIGHT
            input_width = FACE_IMG_WIDTH
            self.weights = np.random.randint(-1, 1, size=(input_height, input_width))
            self.bias = 0
        elif task_name == "Digit Classification":
            input_height = DIGIT_IMG_HEIGHT
            input_width = DIGIT_IMG_WIDTH
            self.weights = [np.random.randint(-1, 1, size=(input_height, input_width)) for _ in range(10)]
            self.bias = [0 for _ in range(10)]

    def run(self, image):
        height, width = image.shape
        
        if self.task_name == "Face Detection":
            assert image.shape == self.weights.shape
            score = self.bias

            for r in range(height):
                for c in range(width):
                    score += self.weights[r][c] * image[r][c]

            if (score > 0):
                return 1
            else:
                return 0

        elif self.task_name == "Digit Classification":
            assert image.shape == self.weights[0].shape

            score = [self.bias[i] for i in range(10)]

            for i in range(10):
                for r in range(height):
                    for c in range(width):
                        score[i] += self.weights[i][r][c] * image[r][c]

            max = score[0]
            max_index = 0
            for i in range(10):
                if score[i] > max:
                    max = score[i]
                    max_index = i
            
            return max_index
    
    def update_weights(self, image, expected, result):
        height, width = image.shape

        if self.task_name == "Face Detection":
            if (expected == 0 and result == 1):                        
                self.bias -= 1
                for r in range(height):
                    for c in range(width):
                        if (image[r][c] == 1):
                            self.weights[r][c] -= image[r][c]
            elif (expected == 1 and result == 0):
                self.bias += 1
                for r in range(height):
                    for c in range(width):
                        if (image[r][c] == 1):
                            self.weights[r][c] += image[r][c]

        elif self.task_name == "Digit Classification":
            self.bias[expected] += 1
            self.bias[result] -= 1            
            for r in range(height):
                for c in range(width):
                    if (image[r][c] == 1):
                        self.weights[expected][r][c] += 1
                        self.weights[result][r][c] -= 1



# --- 4. Training Function ---
def train_model(task_name, model, train_dataset, indices, num_epochs):
    start_time = time.time()
    for epoch in range(num_epochs):
        for i in indices:
            image = train_dataset.image_data[i]
            label = train_dataset.label_data[i]

            res = model.run(image)

            if (res != label):
                model.update_weights(image, label, res)

    end_time = time.time()
    training_time = end_time - start_time
    # print(f'Finished Training. Total time: {training_time:.2f}s')
    return training_time




# --- 5. Evaluation Function ---
def evaluate_model(model, test_dataset):
    correct = 0
    total = 0

    for i in range(len(test_dataset)):
        image = test_dataset.image_data[i]
        label = test_dataset.label_data[i]

        res = model.run(image)

        if (res == label):
            correct += 1

        total += 1

    accuracy = 100 * correct / total
    error_rate = 1.0 - (correct / total)
    # print(f'Accuracy on test set: {accuracy:.2f} %')
    return error_rate


# --- 6. Experiment Loop ---
def run_experiment(task_name, train_dataset, test_dataset):
    print(f"\n--- Running Experiment for: {task_name} ---")
    results_time = {}
    results_error_mean = {}
    results_error_std = {}

    n_train_total = len(train_dataset)

    for percent in TRAINING_PERCENTAGES:
        n_train_subset = int(n_train_total * percent)
        print(f"\nTraining on {percent * 100:.0f}% ({n_train_subset} samples)")

        current_percent_times = []
        current_percent_errors = []

        for run in range(NUM_RUNS_PER_PERCENTAGE):
            print(f"  Run {run + 1}/{NUM_RUNS_PER_PERCENTAGE}")
            # Create model, loss, optimizer FROM SCRATCH for each run
            model = Perceptron(task_name)

            # Sample training indices for this run
            indices = np.random.choice(range(n_train_total), n_train_subset, replace=False)

            # Train
            train_time = train_model(task_name, model, train_dataset, indices, NUM_EPOCHS)

            # Evaluate
            test_error = evaluate_model(model, test_dataset)

            current_percent_times.append(train_time)
            current_percent_errors.append(test_error)
            print(f"    Run {run + 1}: Time={train_time:.2f}s, Error={test_error:.4f}")

            print(model.weights)

        # Calculate stats for this percentage
        avg_time = np.mean(current_percent_times)
        avg_error = np.mean(current_percent_errors)
        std_error = np.std(current_percent_errors)

        results_time[percent] = avg_time
        results_error_mean[percent] = avg_error
        results_error_std[percent] = std_error

        print(f"  Avg Time: {avg_time:.2f}s, Avg Error: {avg_error:.4f}, Std Dev Error: {std_error:.4f}")

    return results_time, results_error_mean, results_error_std


# --- Main Execution ---
if __name__ == '__main__':
    np.set_printoptions(linewidth=np.inf)
    np.set_printoptions(threshold=np.inf)
    
    # Determine the base directory (where the script is located)
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Load Data (handle potential errors during loading)
    try:
        print("Loading Digit Data...")
        digit_train_dataset = ImageLabelDataset(os.path.join(base_dir, 'digitdata/trainingimages'),
                                              os.path.join(base_dir, 'digitdata/traininglabels'), DIGIT_IMG_HEIGHT,
                                              DIGIT_IMG_WIDTH)
        digit_test_dataset = ImageLabelDataset(os.path.join(base_dir, 'digitdata/testimages'),
                                             os.path.join(base_dir, 'digitdata/testlabels'), DIGIT_IMG_HEIGHT,
                                             DIGIT_IMG_WIDTH)

        print("\nLoading Face Data...")
        # Adjust class label mapping if needed (e.g., labels are 0/1)
        face_train_dataset = ImageLabelDataset(os.path.join(base_dir, 'facedata/facedatatrain'),
                                             os.path.join(base_dir, 'facedata/facedatatrainlabels'), FACE_IMG_HEIGHT,
                                             FACE_IMG_WIDTH)
        face_test_dataset = ImageLabelDataset(os.path.join(base_dir, 'facedata/facedatatest'),
                                            os.path.join(base_dir, 'facedata/facedatatestlabels'), FACE_IMG_HEIGHT,
                                            FACE_IMG_WIDTH)
    except (FileNotFoundError, ValueError, AssertionError) as e:
        print(f"\nError loading data: {e}")
        print(f"Current script directory: {base_dir}")
        print(
            "Please ensure data.zip is downloaded, unzipped correctly so that 'digitdata' and 'facedata' folders are in the same directory as this script.")
        print("and the file paths and parsing logic in ImageLabelDataset are correct.")
        exit()


    # Run Experiments
    digit_times, digit_errors, digit_stds = run_experiment(
        "Digit Classification", digit_train_dataset, digit_test_dataset 
    )

    face_times, face_errors, face_stds = run_experiment(
        "Face Detection", face_train_dataset, face_test_dataset
    )

    # # --- Plotting Results ---
    # percentages = list(digit_times.keys())
    # percentages_100 = [p * 100 for p in percentages]  # For x-axis label

    # # Plot Training Time
    # plt.figure(figsize=(12, 5))
    # plt.subplot(1, 2, 1)
    # plt.plot(percentages_100, list(digit_times.values()), marker='o', label='Digits')
    # plt.plot(percentages_100, list(face_times.values()), marker='s', label='Faces')
    # plt.xlabel('Percentage of Training Data Used (%)')
    # plt.ylabel('Average Training Time (s)')
    # plt.title('Training Time vs. Training Data Size')
    # plt.legend()
    # plt.grid(True)

    # # Plot Prediction Error
    # plt.subplot(1, 2, 2)
    # plt.errorbar(percentages_100, list(digit_errors.values()), yerr=list(digit_stds.values()), fmt='-o', label='Digits',
    #              capsize=5)
    # plt.errorbar(percentages_100, list(face_errors.values()), yerr=list(face_stds.values()), fmt='-s', label='Faces',
    #              capsize=5)
    # plt.xlabel('Percentage of Training Data Used (%)')
    # plt.ylabel('Average Prediction Error Rate')
    # plt.title('Prediction Error vs. Training Data Size')
    # plt.legend()
    # plt.grid(True)

    # plt.tight_layout()
    # plt.savefig('perceptron_performance_curves.png')  # Save the plot
    # print("\nPerformance curves saved to perceptron_performance_curves.png")
    # plt.show()