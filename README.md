# Focus Rectification Deep Logistic Regression
*Single-Label Multi-Class Image Classification by Deep Logistic Regression. AAAI2019 (Oral)*. [[PaperDownload]](https://arxiv.org/abs/1811.08400)
 
There are the proposed Focus Rectification Logistic Regression losses in both [Tensorflow](https://www.tensorflow.org/) and [Pytorch](https://pytorch.org/) implementations. You may refer to our [Slides]() and [Poster](https://github.com/yanbeic/Deep-Association-Learning/blob/master/poster/bmvc18-poster.pdf) for a quick overview.


## How to use
Apply the provided loss functions with any Deep Networks. 

Some tips:
- The inputs are the logits (outputs of last layer) and the groundtruth label (single label or multi-label).
- Generally, the setting in training models are consistent to that for Deep Networks with Softmax Cross Entropy Loss.
- Compared with Softmax Cross Entropy loss, Logistic regression optimisation prefers a smaller learning rate.
- For some specific applications, a balance between Logistic Loss and Regularisation loss is recommended.



## Citation
Please refer to the following if this repository is useful for your research.
```
@article{dong2018single,
  title={Single-Label Multi-Class Image Classification by Deep Logistic Regression},
  author={Dong, Qi and Zhu, Xiatian and Gong, Shaogang},
  journal={AAAI},
  year={2019}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.


## Contact
Feel free to contact [me](http://www.eecs.qmul.ac.uk/~qd301/) for any question.

 
