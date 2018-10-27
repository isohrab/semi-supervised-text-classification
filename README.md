# Semi-supervised document classification

Using Autoencoders as an unsupervised method to extract the structure of sentences and use it in a Document Classification.

## Motivation

As described in this paper [Andrew M. Dai, Quoc V. Le](https://arxiv.org/pdf/1511.01432), with autoencoders, we can present data in a lower dimension which is called latent space. We use Wikipedia sentences for this step. The goal is that encoder converts input sentences to a latent space, and then decoder should reconstruct these sentences. So in another word, x=y.

After training autoencoder, we will use only encoder part in the classification step. The inputs are a set of sentences which belong to a document. We feed them to the encoder and use latent space to feed another RNN for classification.

## Relevant literature 

A short list of literature (articles/books/blog posts/...). We will
pick some of the listed papers for further class discussion.

- Unsupervised Feature Selection for Text Classification via Word Embedding by Weikang Rui, Jinwen Liu, and Yawei Jia. [PDF](http://ieeexplore.ieee.org/abstract/document/7509787/)
- Semi-supervised Sequence Learning by Andrew M. Dai, Quoc V. Le [PDF](https://arxiv.org/pdf/1511.01432)


## Available data, tools, resources
To compare our result, we created also a simple basic classifier model which does not include encoder part. Then we evaluated our model on two datasets and three different word embedding. The results are shown table below:


#### Basic model results
| dataset             |  Glove  |  Lexvec | FastText |
|----------           |:-------:|:-------:|:--------:|
| Large Movie review  | ----  | ----  | ----   |
| 20 newsgroups       | 73.35%  | 71.19%  | 72.65%   |

#### Model with Encoder part
| dataset             |  Glove  |  Lexvec | FastText |
|----------           |:-------:|:-------:|:--------:|
| Large Movie review  | ----  | ----  | ----   |
| 20 newsgroups       | 76.28%  | 73.84%  | 76.15%   |

 __Note__: The comparison is unfair. I used Glove 42B, Fasttext 16B and Lexvec 7B. [issue#1](https://github.com/isohrab/semi-supervised-text-classification/issues/1)

## Acknowledgement

Many thanks to [JayParks](https://github.com/JayParks). I'm not use only his seq2seq model, but also I learn a lot from his [code and comments](https://github.com/JayParks/tf-seq2seq/blob/master/seq2seq_model.py)
