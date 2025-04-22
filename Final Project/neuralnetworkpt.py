import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import numpy as np
import time
import os
import matplotlib.pyplot as plt # For plotting results

# --- 1. Constants and Hyperparameters ---
# TODO: Determine these from your data files!
DIGIT_IMG_HEIGHT = 28 # Example
DIGIT_IMG_WIDTH = 28  # Example
FACE_IMG_HEIGHT = 70  # Example
FACE_IMG_WIDTH = 60  # Example

# Calculate Input Size
INPUT_SIZE_DIGITS = DIGIT_IMG_HEIGHT * DIGIT_IMG_WIDTH
INPUT_SIZE_FACES = FACE_IMG_HEIGHT * FACE_IMG_WIDTH

# Model Hyperparameters (tune these)
HIDDEN1_SIZE = 128
HIDDEN2_SIZE = 64
LEARNING_RATE = 0.001
BATCH_SIZE = 64
NUM_EPOCHS = 20 # Adjust as needed

# Output sizes
NUM_CLASSES_DIGITS = 10
NUM_CLASSES_FACES = 2 # Using 2 output neurons and CrossEntropyLoss

# Data Paths (TODO: Update these after unzipping)
DATA_DIR = './data' # Assumes data.zip is unzipped into a 'data' directory
# Example paths - adjust based on actual file names/structure
DIGIT_TRAIN_IMAGES_PATH = os.path.join(DATA_DIR, 'digitdata/trainingimages')
DIGIT_TRAIN_LABELS_PATH = os.path.join(DATA_DIR, 'digitdata/traininglabels')
DIGIT_TEST_IMAGES_PATH  = os.path.join(DATA_DIR, 'digitdata/testimages')
DIGIT_TEST_LABELS_PATH  = os.path.join(DATA_DIR, 'digitdata/testlabels')

FACE_TRAIN_IMAGES_PATH = os.path.join(DATA_DIR, 'facedata/facedatatrain')
FACE_TRAIN_LABELS_PATH = os.path.join(DATA_DIR, 'facedata/facedatatrainlabels')
FACE_TEST_IMAGES_PATH  = os.path.join(DATA_DIR, 'facedata/facedatatest')
FACE_TEST_LABELS_PATH  = os.path.join(DATA_DIR, 'facedata/facedatatestlabels')

# Experiment parameters
TRAINING_PERCENTAGES = np.arange(0.1, 1.1, 0.1) # 10% to 100%
NUM_RUNS_PER_PERCENTAGE = 5 # For calculating standard deviation

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 2. Custom Dataset ---
class ImageLabelDataset(Dataset):
    def __init__(self, image_file, label_file, img_height, img_width):
        self.img_height = img_height
        self.img_width = img_width
        self.image_data = self._load_images(image_file)
        self.label_data = self._load_labels(label_file)
        assert len(self.image_data) == len(self.label_data), "Number of images and labels must match!"

    def _load_images(self, file_path):
        # TODO: Implement file parsing based on the actual data format
        # This is a placeholder - you NEED to adapt this based on how
        # images are stored in the files (e.g., one image per line,
        # fixed width characters, etc.)
        print(f"Loading images from: {file_path}")
        images = []
        with open(file_path, 'r') as f:
            lines = f.readlines()
            img = []
            for i, line in enumerate(lines):
                # Assuming each line is a row of the image
                # Modify this part extensively based on actual format!
                # Example for fixed width representation:
                line = line.rstrip('\n') # Remove trailing newline IMPORTANTLY
                if len(line) == self.img_width: # Check if line looks like image row data
                   img.extend([1.0 if c in ['#', '+'] else 0.0 for c in line]) # Example: Binary features
                # Detect image boundaries (e.g., after height lines)
                if len(img) == self.img_height * self.img_width:
                   images.append(np.array(img, dtype=np.float32))
                   img = [] # Reset for next image
                elif len(line) != self.img_width and len(img) > 0 :
                    print(f"Warning: Potential parsing issue at line {i+1} in {file_path}. Img buffer size: {len(img)}")
                    # Decide how to handle potentially incomplete images or parsing errors
                    img = [] # Discard potentially corrupt buffer

        print(f"Loaded {len(images)} images.")
        if not images:
             raise ValueError(f"No images loaded from {file_path}. Check parsing logic and file format.")
        return torch.tensor(np.array(images)) # Shape: (num_images, height * width)

    def _load_labels(self, file_path):
        # TODO: Implement label file parsing
        print(f"Loading labels from: {file_path}")
        with open(file_path, 'r') as f:
            # Assuming one label per line, convert to integer
            labels = [int(line.strip()) for line in f if line.strip()]
        print(f"Loaded {len(labels)} labels.")
        return torch.tensor(labels, dtype=torch.long) # Use torch.long for CrossEntropyLoss

    def __len__(self):
        return len(self.label_data)

    def __getitem__(self, idx):
        image = self.image_data[idx]
        label = self.label_data[idx]
        return image, label

# --- 3. Model Definition ---
class ThreeLayerNet(nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, num_classes):
        super(ThreeLayerNet, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden1_size)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(hidden1_size, hidden2_size)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(hidden2_size, num_classes)
        # No final activation here if using nn.CrossEntropyLoss
        # If using nn.NLLLoss, add nn.LogSoftmax()
        # If binary (1 output neuron) + nn.BCEWithLogitsLoss, no final activation needed
        # If binary (1 output neuron) + nn.BCELoss, add nn.Sigmoid()

    def forward(self, x):
        out = self.layer1(x)
        out = self.relu1(out)
        out = self.layer2(out)
        out = self.relu2(out)
        out = self.layer3(out)
        return out

# --- 4. Training Function ---
def train_model(model, train_loader, criterion, optimizer, num_epochs):
    start_time = time.time()
    model.train() # Set model to training mode
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}') # Optional: print epoch loss

    end_time = time.time()
    training_time = end_time - start_time
    # print(f'Finished Training. Total time: {training_time:.2f}s')
    return training_time


# --- 5. Evaluation Function ---
def evaluate_model(model, test_loader):
    model.eval() # Set model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad(): # No need to track gradients during evaluation
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            # Get predictions from the maximum value
            _, predicted = torch.max(outputs.data, 1)
            # For binary classification with 1 output neuron + sigmoid, you'd do:
            # predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    error_rate = 1.0 - (correct / total)
    # print(f'Accuracy on test set: {accuracy:.2f} %')
    return error_rate

# --- 6. Experiment Loop ---
def run_experiment(task_name, train_dataset, test_dataset, input_size, num_classes):
    print(f"\n--- Running Experiment for: {task_name} ---")
    results_time = {}
    results_error_mean = {}
    results_error_std = {}

    # Create test loader (used for all percentages)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    n_train_total = len(train_dataset)

    for percent in TRAINING_PERCENTAGES:
        n_train_subset = int(n_train_total * percent)
        print(f"\nTraining on {percent*100:.0f}% ({n_train_subset} samples)")

        current_percent_times = []
        current_percent_errors = []

        for run in range(NUM_RUNS_PER_PERCENTAGE):
            print(f"  Run {run+1}/{NUM_RUNS_PER_PERCENTAGE}")
            # Create model, loss, optimizer FROM SCRATCH for each run
            model = ThreeLayerNet(input_size, HIDDEN1_SIZE, HIDDEN2_SIZE, num_classes).to(device)
            criterion = nn.CrossEntropyLoss() # Adjust if needed for faces (e.g., BCEWithLogitsLoss)
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

            # Sample training indices for this run
            indices = np.random.choice(range(n_train_total), n_train_subset, replace=False)
            sampler = SubsetRandomSampler(indices)
            train_loader_subset = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, sampler=sampler)

            # Train
            train_time = train_model(model, train_loader_subset, criterion, optimizer, NUM_EPOCHS)

            # Evaluate
            test_error = evaluate_model(model, test_loader)

            current_percent_times.append(train_time)
            current_percent_errors.append(test_error)
            print(f"    Run {run+1}: Time={train_time:.2f}s, Error={test_error:.4f}")


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
    # Load Data (handle potential errors during loading)
    try:
        print("Loading Digit Data...")
        digit_train_dataset = ImageLabelDataset(DIGIT_TRAIN_IMAGES_PATH, DIGIT_TRAIN_LABELS_PATH, DIGIT_IMG_HEIGHT, DIGIT_IMG_WIDTH)
        digit_test_dataset = ImageLabelDataset(DIGIT_TEST_IMAGES_PATH, DIGIT_TEST_LABELS_PATH, DIGIT_IMG_HEIGHT, DIGIT_IMG_WIDTH)

        print("\nLoading Face Data...")
        # Adjust class label mapping if needed (e.g., labels are 0/1)
        face_train_dataset = ImageLabelDataset(FACE_TRAIN_IMAGES_PATH, FACE_TRAIN_LABELS_PATH, FACE_IMG_HEIGHT, FACE_IMG_WIDTH)
        face_test_dataset = ImageLabelDataset(FACE_TEST_IMAGES_PATH, FACE_TEST_LABELS_PATH, FACE_IMG_HEIGHT, FACE_IMG_WIDTH)
    except (FileNotFoundError, ValueError, AssertionError) as e:
        print(f"\nError loading data: {e}")
        print("Please ensure data.zip is downloaded, unzipped correctly into a 'data' directory,")
        print("and the file paths and parsing logic in ImageLabelDataset are correct.")
        exit()

    # Run Experiments
    digit_times, digit_errors, digit_stds = run_experiment(
        "Digit Classification", digit_train_dataset, digit_test_dataset, INPUT_SIZE_DIGITS, NUM_CLASSES_DIGITS
    )

    # Note: If face labels are 0/1, CrossEntropyLoss still works if your model has 2 outputs.
    # If you change the model to have 1 output + Sigmoid, use BCEWithLogitsLoss and adjust evaluation.
    face_times, face_errors, face_stds = run_experiment(
        "Face Detection", face_train_dataset, face_test_dataset, INPUT_SIZE_FACES, NUM_CLASSES_FACES
    )

    # --- Plotting Results ---
    percentages = list(digit_times.keys())
    percentages_100 = [p * 100 for p in percentages] # For x-axis label

    # Plot Training Time
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(percentages_100, list(digit_times.values()), marker='o', label='Digits')
    plt.plot(percentages_100, list(face_times.values()), marker='s', label='Faces')
    plt.xlabel('Percentage of Training Data Used (%)')
    plt.ylabel('Average Training Time (s)')
    plt.title('Training Time vs. Training Data Size')
    plt.legend()
    plt.grid(True)

    # Plot Prediction Error
    plt.subplot(1, 2, 2)
    plt.errorbar(percentages_100, list(digit_errors.values()), yerr=list(digit_stds.values()), fmt='-o', label='Digits', capsize=5)
    plt.errorbar(percentages_100, list(face_errors.values()), yerr=list(face_stds.values()), fmt='-s', label='Faces', capsize=5)
    plt.xlabel('Percentage of Training Data Used (%)')
    plt.ylabel('Average Prediction Error Rate')
    plt.title('Prediction Error vs. Training Data Size')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('pytorch_nn_performance_curves.png') # Save the plot
    print("\nPerformance curves saved to pytorch_nn_performance_curves.png")
    plt.show()

    # You would then use these plots and the printed results for your report.

