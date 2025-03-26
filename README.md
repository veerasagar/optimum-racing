# Optimum Racing: A Formula one race strategy predictor

This project implements a Deep Recurrent Q-Network (DRQN) for optimizing race strategy decisions, specifically focusing on pit stop management in a simulated racing environment. The agent learns when and which tire compound to change during a race to maximize performance based on lap times, tire degradation, and positional changes.

## DRQN Model and Training

### DRQN Architecture

- Consists of:
  - An LSTM layer to handle sequential input data.
  - Fully connected layers that process the hidden state output of the LSTM to produce Q-values for each available action.

### Replay Buffer

- Implemented using a deque to store experiences (state, action, reward, next state, done) for training.
- Supports randomized sampling during model updates, which improves training stability.

### Training Loop

- Involves:
  - Iteratively interacting with the environment by choosing actions based on an ε-greedy strategy.
  - Storing experiences in the replay buffer.
  - Sampling batches from the replay buffer to update the DRQN using mean squared error loss.
  - Updating a target network periodically to stabilize learning.

### Execution Flow

1. **Data Loading:**  
   - The system reads CSV files from specified folders representing different seasons.

2. **Environment Initialization:**  
   - For each CSV file, the environment is initialized, and a sequential state representation is built.

3. **Model Training:**  
   - The DRQN model is trained over several episodes per file.
   - After each training session on a file, the model is updated and used as a shared model for subsequent files.

4. **Model Saving:**  
   - Once training is complete across all files, the trained model is saved for future inference or further training.

### Dependencies

- Python 3.x
- NumPy
- Pandas
- PyTorch

## To run

```python
cd src
python dqrn_torch.py 
```
