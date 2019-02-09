# Single-Label Multi-Class Image Classification by Deep Logistic Regression
*Published on AAAI 2019 (Oral)*. [Paper](https://arxiv.org/abs/1811.08400), [Slides](http://www.eecs.qmul.ac.uk/~qd301/papers/LRSR_AAAI19_slides.pdf) and [Poster](http://www.eecs.qmul.ac.uk/~qd301/papers/LRSR_AAAI19_poster.pdf) for your reference.

## Abstract
The objective learning formulation is essential for the success of convolutional neural networks. In this work, we analyse thoroughly the standard learning objective functions for multi- class classification CNNs: softmax regression (SR) for single- label scenario and logistic regression (LR) for multi-label scenario. Our analyses lead to an inspiration of exploiting LR for single-label classification learning, and then the disclosing of the negative class distraction problem in LR. To address this problem, we develop two novel LR based objective functions that not only generalise the conventional LR but importantly turn out to be competitive alternatives to SR in single label classification. Extensive comparative evaluations demonstrate the model learning advantages of the proposed LR functions over the commonly adopted SR in single-label coarse-grained object categorisation and cross-class fine-grained person in- stance identification tasks. We also show the performance superiority of our method on clothing attribute classification in comparison to the vanilla LR function.

![lossvar](http://www.eecs.qmul.ac.uk/~qd301/papers/loss_var.png)


## How to use
Here are Focus Rectification Logistic Regression losses in both [Tensorflow](https://www.tensorflow.org/) and [Pytorch](https://pytorch.org/) implementations for your reference.
Apply the provided loss functions with any Deep Networks directly. 

Some tips:
- The inputs are the logits (outputs of last layer) and the groundtruth label (single label or multi-label).
- Generally, the setting in training models are consistent to that for Deep Networks with Softmax Cross Entropy Loss.
- Compared with Softmax Cross Entropy loss, Logistic regression optimisation prefers a smaller learning rate empirically.
- For some specific applications, a weighting for the balance between Logistic Loss and Regularisation loss is recommended.



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
Feel free to contact [Qi Dong](http://www.eecs.qmul.ac.uk/~qd301/) for any question. Cheers.

 
