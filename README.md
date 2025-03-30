# SGD Optimizations: Adaptive Deep Learning Training

## Overview  
In this project, I used a combination of advanced techniques for deep learning model training optimization. These include adaptive learning rates, data normalization, advanced training techniques, and optimizer tuning, all to a synthetic noisy sine wave dataset. SGD and Adam optimizers are employed in training the model to maximize convergence and minimize loss in training to achieve efficient learning even in the presence of noisy data.

## Key Concepts Implemented  

### 1. Data Normalization  
Normalization is a crucial preprocessing step in machine learning. I scaled the dataset to ensure all features were proportionally adjusted, preventing the model from favoring features with larger values or different scales.

### 2. Activation Function: ReLU  
The ReLU activation function introduces non-linearity, allowing the model to learn complex relationships between inputs and outputs. This helps in identifying sophisticated patterns in the data.

### 3. Weight Initialization: Xavier Initialization  
To prevent exploding or vanishing gradients, I used Xavier Initialization. This method ensures weights are scaled appropriately, facilitating smoother gradient flow during backpropagation.

### 4. Loss Function: Mean Squared Error (MSE)  
For training, I used MSE to measure the difference between predicted and true values. This loss function is well-suited for regression tasks, where the goal is to minimize prediction errors.

### 5. Optimizers: SGD and Adam  
I implemented two different optimizers: Stochastic Gradient Descent (SGD) and Adam. Both optimizers are used to update the weights of the model during training.  

- **SGD (Stochastic Gradient Descent):** Updates weights using a fixed learning rate.

- **Adam:** Dynamically adjusts the learning rate, combining Momentum and RMSprop benefits for better convergence. 

### 6. Learning Rate Scheduling: Cyclic Learning Rate (CyclicLR)  
I used a Cyclic Learning Rate scheduler to dynamically adjust the learning rate during training. This helps the model escape local minima, improving training efficiency and generalization. 

## Training Pipeline  

### 1. Data Generation  
A synthetic noisy sine wave dataset was created to serve as input-output pairs for training, providing a controlled yet challenging regression task.

### 2. Model Training  
The model was trained using configurations from a `config.yaml` file, allowing flexible adjustments to hyperparameters like learning rate, batch size, and optimizer choice. 

### 3. Adaptive Learning Rates  
By combining Adam with CyclicLR, the model adapted its learning rate dynamically, leading to better convergence and avoiding training stagnation.

### 4. Evaluation  
Training progress was monitored by tracking loss and learning rate at each epoch. This allowed for adjustments to improve model performance.  
<p align="center">
  <img src="[https://github.com/ranzeet013/iBeer.ai/blob/main/projectLogo/iBeer.ai.png](https://github.com/ranzeet013/SGD-Optimizations/blob/main/results/loss_curve.png)" alt="Training Loss Image" width="300" />
</p>

## Results  
The trained model successfully fit the noisy sine wave function. Adaptive learning rates and proper initialization helped achieve stable convergence and low loss. 
## Conclusion  
This project demonstrated the effectiveness of techniques like Xavier initialization, adaptive learning rates, and advanced optimizers (Adam) in improving deep learning model training. These methods are widely applicable and can be scaled to more complex models and larger datasets.
