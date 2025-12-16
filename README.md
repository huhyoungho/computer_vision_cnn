# Computer_Vision_Cnn
This is a final project for computer vision class.

# End-to-End Deep Learning Pipeline for Medical Tissue Classification

## 📌 Project Overview
This project implements a comprehensive deep learning training pipeline to classify histological tissue images into four distinct categories: **Decidual tissue, Hemorrhage, Chorionic villi, and Trophoblastic tissue**.

The solution utilizes **Transfer Learning with DenseNet121**, incorporating a robust **Custom Dataset Class** with specific preprocessing algorithms (contour-based cropping) and an advanced training strategy (Cosine Annealing, AdamW) to ensure high performance and generalization.

---

## 📖 Theoretical Background & Pipeline Design
*This project strictly follows the standard Deep Learning Training Pipeline principles.*

### 1. Training Pipeline Concept
*   **Data Conversion:** Training a deep learning model requires converting raw data into a specific format processable by the model.
*   **Dimension Mismatch:** For example, while the model may require $512 \times 512$ inputs, raw collection data might be $1280 \times 720$. We need a mechanism to convert available data into the exact required format.
*   **Goal:** The goal of the Image Dataset component is to load images and their corresponding labels efficiently to feed them into the model.

### 2. Three Steps Implementation
The code is structured into three essential steps:

*   **Step 1: Import Libraries**
    *   Essential libraries (`torch`, `torchvision`, `cv2`, `numpy`) are imported to handle tensor operations and image processing.

*   **Step 2: Create the Custom Dataset Class**
    *   The main task is to return a `[input, label]` pair. We defined functions inside the class to preprocess data dynamically.
    *   **`__init__`**: Loads image files using `PIL`/`OS` libraries, stores file paths/labels, and defines transformations.
    *   **`__len__`**: Returns the total number of images (length of the dataset).
    *   **`__getitem__`**: Returns one training example. It reads the image from disk, applies specific **5-Step Preprocessing**, and converts labels to tensors.
        *   **Preprocessing Logic Implemented:**
            1.  Convert image to **Binary**.
            2.  Apply **Morphological Operations** (Dilation/Erosion) to remove noise.
            3.  Select the **Largest Contour** and calculate extreme points (Top, Bottom, Left, Right).
            4.  **Crop** the image using these extreme points to focus on the Region of Interest (ROI).
            5.  **Resize** the image to $224 \times 224$ using **Bicubic Interpolation**.

*   **Step 3: Instantiate Dataset and DataLoader**
    *   **DataLoader**: Wraps the dataset, enabling batching, shuffling, and parallel loading via multiprocessing.
    *   **Batch Processing**: Instead of processing one sample at a time (which is inefficient), we process data in batches (e.g., shape `[N, C, H, W]`).
    *   **Epoch vs. Iteration**: An Epoch is when the model has seen all training data once; an Iteration is when one batch is passed.

### 3. Model Definition
*   **Architecture**: Defined using `torch.nn.Module` by leveraging a pre-trained **DenseNet121** from `torchvision.models`.
*   **Transfer Learning**: To act as a powerful feature extractor, the backbone is used, and a new Fully Connected (FC) layer is fine-tuned for our 4 specific classes.
*   **Optimization**:
    *   **Loss Function**: `CrossEntropyLoss` (standard for multi-class classification).
    *   **Optimizer**: `AdamW` (Adam with Weight Decay) is selected to update model weights based on calculated gradients.

### 4. Training Loop & Evaluation
*   **Forward Pass**: Input data is passed through the model to generate predictions.
*   **Calculate Loss**: The difference between predictions and actual labels is computed.
*   **Backward Pass**: Backpropagation calculates gradients, and the optimizer updates the weights.
*   **Validation**: The model is evaluated on a separate Test set every epoch to monitor generalization and prevent overfitting.
*   **Saving**: The model state dictionary (`state_dict`) is saved whenever the F1-Score improves.

---

## 🛠️ Technical Improvements & Hyperparameters
To achieve an **A+ standard performance**, I introduced advanced tuning beyond the basic pipeline:

| Component | Setting | Reason |
| :--- | :--- | :--- |
| **Model** | **DenseNet121** | chosen for its efficient feature reuse and gradient flow. |
| **Transfer Strategy** | **Partial Unfreezing** | The last Dense Block (`denseblock4`) is unfrozen to allow the model to learn tissue-specific textures, while earlier layers remain frozen. |
| **Resize Method** | **Bicubic** | Strictly adhered to the requirement for high-quality downsampling. |
| **Optimizer** | **AdamW** | Superior handling of L2 regularization compared to standard Adam. |
| **Scheduler** | **CosineAnnealingLR** | Gradually decreases LR to help the model converge to the global minimum. |
| **Metrics** | **Weighted F1** | Used to accurately evaluate performance despite potential class imbalances. |
| **Augmentation** | **Strong** | Rotation, Color Jitter, and Flips are applied to improve robustness. |

---

## 🚀 How to Run

1.  **Environment Setup**
    Ensure the following dependencies are installed:
    ```bash
    pip install torch torchvision opencv-python numpy matplotlib scikit-learn tqdm pillow
    ```

2.  **Dataset Location**
    The code is configured to automatically detect the dataset at:
    `~/Desktop/computer_vision/computer_vision_cnn/POC_Dataset`

3.  **Execute Training**
    Run the main script:
    ```bash
    python cnn.py
    ```

---

## 📊 Evaluation Results
The pipeline outputs the Loss, Accuracy, Precision, Recall, and F1-Score for each epoch.

```text
📅 Epoch 20/20 | LR: 0.000007
   [Train] Loss: 0.0668 | Acc: 0.9588
   [Test ] Loss: 0.7976 | Acc: 0.8458 | Prec: 0.8516 | Rec: 0.8458 | F1: 0.8421

🎉 Final Best F1-Score: 0.8777 (Epoch 6/20)