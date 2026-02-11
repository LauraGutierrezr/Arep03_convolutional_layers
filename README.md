# Convolutional Neural Networks: Architecture Design & Experimental Analysis

This project implements **convolutional neural networks** to understand architectural design choices, inductive bias, and empirical performance trade-offs. Rather than treating neural networks as black boxes, this assignment focuses on intentional architectural decisions: kernel sizes, filter depths, stride, and padding — and their measurable impact on learning.

The concrete implementation in this repository is a small CNN for **CIFAR-10** built with TensorFlow/Keras in the notebook `convolutional_layers.ipynb`. It includes dataset exploration, a fully connected baseline, a CNN with 3×3 kernels, a controlled experiment on kernel size (3×3 vs 5×5), interpretation of results, and brief deployment notes for AWS SageMaker.

## Learning Objectives

By completing this assignment, students should be able to:

- **Understand** the role and mathematical intuition behind convolutional layers
- **Analyze** how architectural decisions (kernel size, depth, stride, padding) affect learning dynamics
- **Compare** convolutional layers with fully connected layers on image-based data
- **Design** architectures from scratch with explicit architectural reasoning (not copy-paste)
- **Perform** exploratory data analysis (EDA) suitable for neural network tasks
- **Conduct** controlled experiments isolating one architectural variable while keeping others fixed
- **Communicate** design trade-offs and empirical findings with clarity and rigor

## Getting Started

These instructions will help you set up the project locally for development and experimentation. Instructions for cloud execution using **AWS SageMaker** are included in the Deployment section.

### Prerequisites

You need the following software installed:

- Python 3.9 or later
- Jupyter Notebook or Jupyter Lab
- Git (for version control)

#### Main Python Libraries Used

```
numpy          # Numerical computation
matplotlib     # Visualization
jupyter        # Interactive notebook environment
tensorflow     # CNN models (Keras API)
```

In this lab we use TensorFlow/Keras to focus on **architectural design and experimental analysis**, rather than on low-level implementation of convolution and optimization in pure NumPy.

### Installation

1. **Clone or create the project directory:**

```bash
cd convolutional_layers
```

2. **Create and activate a virtual environment:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. **Install dependencies:**

```bash
pip install numpy pandas matplotlib jupyter
```

4. **Start Jupyter Notebook:**

```bash
jupyter notebook
```

5. **Open the main notebook:**

Open `convolutional_layers.ipynb` and run all cells sequentially.

## Dataset Selection & Justification

### Recommended Datasets

Students must choose one existing public dataset suitable for convolutional neural networks:

#### Suggested Sources:
- **TensorFlow Datasets** (`tensorflow_datasets` library)
- **PyTorch** (`torchvision.datasets`)
- **Kaggle** (non-competition datasets only)

#### Dataset Constraints:
- **Image-based**: 2D or 3D tensors (e.g., grayscale or RGB images)
- **Minimum 2 classes** for meaningful classification
- **Fits in memory** on a standard laptop or cloud notebook

#### Examples (Illustrative):
- **MNIST / Fashion-MNIST**: 28×28 grayscale; 10 classes; ~70k samples
- **CIFAR-10 / CIFAR-100**: 32×32 RGB; 10/100 classes; 50k training + 10k test
- **Medical Images** (X-ray, microscopy): Small subsets (~1-5k images)
- **Satellite / Land-use Images**: Aerial imagery classification

### Your Dataset Justification

**In your notebook, provide a brief section explaining:**

1. Why this dataset is appropriate for convolutional architectures
2. Dataset size and class distribution
3. Image dimensions and number of channels
4. Preprocessing requirements (normalization, resizing)
5. Examples of samples from each class (visualized)


### Images Notebook

Estructure proyect

<img width="264" height="232" alt="Captura" src="https://github.com/user-attachments/assets/a3175cd2-e432-4de8-b3e6-73c2a85f08e2" />


### This Lab: CIFAR-10

In this repository we use **CIFAR-10** as the dataset:

- 50,000 training images and 10,000 test images
- 10 classes of small natural images (e.g., airplane, dog, truck)
- Image shape 32×32 with 3 RGB channels
- Simple preprocessing: convert to `float32` and normalize pixel values to the range [0, 1]

This dataset is well suited for convolutional architectures because it consists of small images where **local patterns** (edges, corners, textures) are important for classification.

## Notebook Structure and Figures

The main analysis is in `convolutional_layers.ipynb` and generates several figures saved under the `images/` folder:

- `images/cifar10.png`: One example image per class in CIFAR-10. This gives an intuitive visual overview of what the model must learn to distinguish.
- `images/cifar10_dims.png`: Histograms of image width and height (all images are 32×32). This confirms the spatial resolution is fixed and suitable for a small CNN.
- `images/cifar10_preprocessing.png`: Shows an original image, its normalized version, and the pixel value distribution after normalization. This illustrates the basic preprocessing used before training.
- `images/baseline_performance.png`: Training and validation accuracy/loss curves for the fully connected (non-convolutional) baseline model.
- `images/cnn_performance.png`: Training and validation accuracy/loss curves for the CNN with 3×3 kernels, which typically outperforms the baseline.
- `images/kernel_size_comparison.png`: Validation accuracy and loss curves comparing two CNNs that differ only in the first kernel size (3×3 vs 5×5). This supports the controlled experiment on kernel size.

## Validation & Testing

### End-to-End Notebook Validation

1. **Execution**: Run all notebook cells without errors
2. **Loss curves**: Training and validation loss decrease monotonically (or plateau)
3. **Baseline performance**: Non-convolutional model achieves >random accuracy
4. **Convolutional improvement**: CNN should outperform baseline (or provide clear explanation if it doesn't)
5. **Experiment reproducibility**: Run experiments multiple times; report mean ± std dev

### Code Quality Checks

- **No high-level ML libraries**: Verify no TensorFlow, PyTorch, Keras, or scikit-learn imports (except for data loading if necessary)
- **NumPy vectorization**: Convolution, activation, and gradient operations use NumPy efficiently
- **Clear separation**: Distinct sections for data prep → model definition → training → evaluation
- **Inline comments**: Explain non-obvious operations, especially convolution logic

## Deployment to AWS SageMaker

### Steps:

1. **Prepare notebook**: Ensure all cells are executable and documented
2. **Upload to SageMaker Studio**: Create a SageMaker notebook instance
3. **Train in cloud**: Run training with cloud resources
4. **Save model**: Export weights and architecture parameters
5. **Create endpoint**: Deploy to SageMaker endpoint for real-time inference
6. **Test inference**: Document sample predictions on new data

### Example Inference Test:

```
Input: Random image from test set
Output: Class prediction + confidence probability
```

## Assignment Structure & Tasks

### Task 1: Dataset Exploration (EDA)

**Deliverable**: A concise EDA section in your notebook with visualizations.

**Include:**
- Dataset size (number of images and classes)
- Class distribution (balanced or imbalanced?)
- Image dimensions and channels (e.g., 28×28×1 for grayscale)
- Examples of samples per class (display images)
- Preprocessing applied (normalization, resizing, augmentation)

**Goal:** Understand data structure, not exhaustive statistics.

---

### Task 2: Baseline Model (Non-Convolutional)

**Deliverable**: A fully connected neural network baseline.

**Architecture:**
- Flatten image → Dense layers → Output
- No convolutions; simple MLP (multi-layer perceptron)

**Report:**
- Architecture diagram or description
- Number of parameters
- Training/validation accuracy and loss curves
- Observed limitations (e.g., overfitting, slow convergence, poor spatial understanding)

**Purpose:** Establish reference point for comparison.

---

### Task 3: Convolutional Architecture Design

**Deliverable**: A CNN designed from scratch with explicit justification.

**Design decisions to document:**
- Number of convolutional layers (1, 2, 3, etc.)
- Kernel sizes (3×3, 5×5, 7×7, etc.) and *why*
- Number of filters per layer and *why*
- Stride and padding choices and *why*
- Activation functions (ReLU, sigmoid, etc.)
- Pooling strategy (max pooling, avg pooling, none) and *why*
- Fully connected layers at the end

**Important:** Architecture should be simple but **intentional**. Do not copy from tutorials. Every choice must be justified.

**Example Justification:**
> "We use 3×3 kernels because they capture local features efficiently while keeping parameters manageable. Stride=1 preserves spatial information. Max pooling reduces dimensionality without losing important features."

---

### Task 4: Controlled Experiments

**Deliverable**: Systematic exploration of ONE architectural variable.

**Choose one aspect to experiment on:**
1. **Kernel size**: Compare 3×3 vs 5×5 vs 7×7 (keep filters, depth, stride fixed)
2. **Number of filters**: Compare 16, 32, 64 filters (keep kernel, depth, stride fixed)
3. **Depth**: Compare 1, 2, 3 convolutional layers (keep kernel, filters, stride fixed)
4. **Pooling**: Compare with/without pooling (keep kernel, filters, depth, stride fixed)
5. **Stride**: Compare stride=1 vs stride=2 (keep kernel, filters, depth, pooling fixed)

**Report:**
- **Quantitative results**: Accuracy, loss, training time for each variant
- **Qualitative observations**: How does the feature map change? Is the model faster/slower?
- **Trade-offs**: Performance vs. computational complexity vs. interpretability

**Visualization:**
- Loss curves for all variants on same plot
- Accuracy comparison (bar chart or table)
- Feature map visualizations (if applicable)

---

### Task 5: Interpretation & Architectural Reasoning

**Deliverable**: A reflective analysis section answering these questions:

1. **Why convolutional layers outperform (or don't) the baseline:**
   - Did the CNN improve over MLP? By how much?
   - What inductive biases does convolution introduce?
   - Is spatial locality captured better?

2. **What is the inductive bias of convolution?**
   - Weight sharing across spatial locations
   - Local receptive fields (only see neighbors, not entire image)
   - Translation equivariance (small shifts in input → small shifts in output)

3. **When would convolution NOT be appropriate?**
   - Fully connected (dense) graphs?
   - Time series without spatial structure?
   - High-dimensional, non-image data?

**Grading note:** This section is **weighted heavily** (20 points). Depth and clarity of reasoning matter more than perfect accuracy.

---

### Task 6: Deployment in SageMaker

**Deliverable**: Model trained and deployed in AWS SageMaker.

**Steps:**
1. Upload notebook to SageMaker Studio
2. Run training with cloud compute resources
3. Save trained weights and model architecture
4. Create a SageMaker endpoint for inference
5. Test with sample images; document predictions

**Document:**
# Estructure

The CNN model was trained in SageMaker using a PyTorch Estimator and a custom training script. The trained model was saved to S3 and automatically packaged by SageMaker. After successful training, the model was deployed to a real-time endpoint and tested with a CIFAR-10 formatted input. The endpoint returned a valid class prediction, confirming correct deployment.


<img width="335" height="233" alt="Captura" src="https://github.com/user-attachments/assets/ac0fe947-4d15-4e04-a1ef-9cd730f67467" />


# Creation endpoints successfully

<img width="1059" height="223" alt="Captura" src="https://github.com/user-attachments/assets/7962996d-52bc-45c7-9697-f14e31b9b707" />


<img width="668" height="557" alt="Captura" src="https://github.com/user-attachments/assets/9e820c4d-fbbe-4686-91ce-cf2e10c7eeef" />


<img width="689" height="721" alt="Captura" src="https://github.com/user-attachments/assets/95abe118-602e-44ab-974b-4a790fe2cc9e" />


---

## Deliverables Checklist

- [ ] **Git repository** (if applicable) with clean commit history
  - [ ] EDA section with visualizations
  - [ ] Baseline MLP implementation
  - [ ] CNN architecture with explicit design justification
  - [ ] Controlled experiment (4+ variants, same plot)
  - [ ] Interpretation and reasoning section
  - [ ] Clean, executable code with markdown explanations
- [ ] **README.md** (this file) with:
  - [ ] Problem and motivation
  - [ ] Dataset description and selection justification
  - [ ] Architecture diagrams (simple sketches acceptable)
  - [ ] Experimental results summary
  - [ ] Interpretation section

---

## Evaluation Criteria (100 points)

| Criterion | Points | Details |
|-----------|--------|---------|
| **Dataset Understanding & EDA** | 15 | Clear exploration, visualizations, preprocessing justification |
| **Baseline Model & Comparison** | 15 | Non-convolutional reference; clear performance report |
| **CNN Architecture Design & Justification** | 25 | Intentional design choices; explicit reasoning for all hyperparameters |
| **Experimental Rigor** | 25 | Controlled experiments; quantitative results; reproducibility |
| **Interpretation & Clarity** | 20 | Deep reasoning about inductive bias and design trade-offs |
| **Bonus: Visualization** | + | Learned filters, feature maps, or other insightful visualizations |

---

## Built With

* Python 3.9+ – Core programming language  
* NumPy – Numerical computation and convolution
* Pandas – Data manipulation and loading  
* Matplotlib – Data visualization and analysis
* Jupyter Notebook – Interactive development and documentation
* AWS SageMaker Studio – Cloud training and inference endpoint deployment

## Authors

* **Valentina Gutiérrez** – Initial implementation and analysis

## License

This project is for academic use only as part of a university machine learning course assignment.

## Acknowledgments

* Course instructors for guidance on neural network architecture and deep learning theory
* TensorFlow and PyTorch communities for dataset documentation and examples
* AWS Academy for cloud infrastructure and SageMaker access
