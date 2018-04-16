# Stacked_Bidirectional_Unidirectional_LSTM
## Deep Bidirectional and Unidirectional LSTM Recurrent Neural Network for Network-wide Traffic Speed Prediction

## Prologue
Normally, we use RNN to characterize the forward dependency of time series data. While, bi-directional RNNs can capture both forward and backward dependencies in time series data. It has been shown that stacked (multi-layer) RNNs/LSTMs work better than one-layer RNN/LSTM in many NLP related applications. It is good to try a combination of bi-directional RNNs and uni-directional RNNs. We find that a neural network with multiple stacked bi-directional LSTMs followed by an uni-directiaonl LSTM works better.

## New Progress
We are designing several internal structures in the LSTM cell to overcome the missing values problem in time series data (replacing the masking layer in the following figure), and to make the model to be suitable for graph-structured data. 

The original model is implemented by Keras. A newly improved version will soon be implemented by PyTorch and released. 

### Environment
* Python 3.6.1
* Keras 2.1.5
* PyTorch 0.3.0

For more detailed information about the model, you can refer to our [paper](https://arxiv.org/abs/1801.02143), referenced at the bottom.

## Model Structure
![alt text](/Images/Architecture.png)

## Data 
To run the code, you need to download the data from the following link: https://drive.google.com/drive/folders/1Mw8tjiPD-wknFu6dY5NTw4tqOiu5X9rz?usp=sharing and put them in the right directory. The data contains two traffic networks in Seattle: a loop detector based freeway network and an INRIX data-based urban traffic network. The details about these netowrk is described in the reference paper.

Description of the datasets:
* `inrix_seattle_speed_matrix_2012`: INRIX Speed Matrix (can be read by Pandas)
* `speed_matrix_2015`: Loop Speed Matrix (can be read by Pandas)

## Cite
Please cite our paper if you use this code or data in your own work:
[Deep Bidirectional and Unidirectional LSTM Recurrent Neural Network for Network-wide Traffic Speed Prediction](https://arxiv.org/abs/1801.02143)
```
@article{cui2018deep,
  title={Deep Bidirectional and Unidirectional LSTM Recurrent Neural Network for Network-wide Traffic Speed Prediction},
  author={Cui, Zhiyong and Ke, Ruimin and Wang, Yinhai},
  journal={arXiv preprint arXiv:1801.02143},
  year={2018}
}
```

