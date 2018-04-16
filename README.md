# Stacked_Bidirectional_Unidirectional_LSTM
## Deep Bidirectional and Unidirectional LSTM Recurrent Neural Network for Network-wide Traffic Speed Prediction

Normally, we use RNN to characterize the forward dependency of time series data. While, bi-directional RNNs can capture both forward and backward dependencies in time series data. It has been shown that stacked (multi-layer) RNNs/LSTMs work better than one-layer RNN/LSTM in many NLP related applications. But few works investigated the combination of bi-directional RNNs and uni-directional RNNs. In our work, we find that the network structure with multiple stacked bi-directional LSTMs followed by an uni-directiaonl LSTM at the end works better.

We are also designing several internal structures in the LSTM cell to overcome the missing values problem in time series data and to make the model to be suitable for graph-structured data. 

The original model is implemented by Keras. A newly improved version will soon be implemented by PyTorch and released. 

For more detailed information about the model, you can refer to our [paper](https://arxiv.org/abs/1801.02143), referenced at the bottom.

### Model Structure
![alt text](/Images/Architecture.png)

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

