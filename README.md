# Optimum Racing: A Formula one race strategy predictor

This project implements a Deep Recurrent Q-Network (DRQN) for optimizing race strategy decisions, specifically focusing on pit stop management in a simulated racing environment. The agent learns when and which tire compound to change during a race to maximize performance based on lap times, tire degradation, and positional changes.

### DRQN Architecture

- Consists of:
  - An LSTM layer to handle sequential input data.
  - Fully connected layers that process the hidden state output of the LSTM to produce Q-values for each available action.

### To run

```python
cd src
python dqrn_torch.py 
```
