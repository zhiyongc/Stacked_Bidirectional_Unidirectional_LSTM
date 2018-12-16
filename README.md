# Deep Stacked Bidirectional and Unidirectional LSTM Recurrent Neural Network
#### For *Network-wide Traffic Speed Prediction*

## Prologue
Normally, we use RNN to characterize the forward dependency of time series data. While, bi-directional RNNs can capture both forward and backward dependencies in time series data. It has been shown that stacked (multi-layer) RNNs/LSTMs work better than one-layer RNN/LSTM in many NLP related applications. It is good to try a combination of bi-directional RNNs and uni-directional RNNs. We find that a neural network with multiple stacked bi-directional LSTMs followed by an uni-directiaonl LSTM works better.

## New Progress
We are designing several internal structures in the LSTM cell to overcome the missing values problem in time series data (replacing the masking layer in the following figure), and to make the model to be suitable for graph-structured data. 

The original model is implemented by Keras. A newly improved version implemented by PyTorch will soon be released. 

### Environment
* Python 3.6.1
* Keras 2.1.5
* PyTorch 0.3.0

For more detailed information about the model, you can refer to our [paper](https://arxiv.org/abs/1801.02143), referenced at the bottom.

## Model Structure
![alt text](/Images/Architecture.png)


## Data 
To run the code, you need to download the loop detector data from my GitHub link: https://github.com/zhiyongc/Seattle-Loop-Data. I'm sorry that the INRIX data can not be shared because of the confidentiality issues.



## Cite
Hope our work can benefit your. If you use this code or data in your own workPlease cite our paper:
[Deep Bidirectional and Unidirectional LSTM Recurrent Neural Network for Network-wide Traffic Speed Prediction](https://arxiv.org/abs/1801.02143)
```
@article{cui2018deep,
  title={Deep Bidirectional and Unidirectional LSTM Recurrent Neural Network for Network-wide Traffic Speed Prediction},
  author={Cui, Zhiyong and Ke, Ruimin and Wang, Yinhai},
  journal={arXiv preprint arXiv:1801.02143},
  year={2018}
}
```
or
```
@inproceedings{cui2016deep,
  title={Deep Stacked Bidirectional and Unidirectional LSTM Recurrent Neural Network for Network-wide Traffic Speed Prediction},
  author={Cui, Zhiyong and Ke, Ruimin and Wang, Yinhai},
  booktitle={6th International Workshop on Urban Computing (UrbComp 2017)},
  year={2016}
}
```

