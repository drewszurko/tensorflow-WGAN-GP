# TensorFlow 2.0 WGAN-GP 

TensorFlow 2.0 implementation of Improved Training of Wasserstein GANs [[1]](https://arxiv.org/abs/1704.00028). 
New/existing TensorFlow features found in this repository include eager execution, AutoGraph, Keras high-level API, and TensorFlow Datasets.


## Requirements
* [Python 3](https://www.python.org/)
* [Abseil](https://abseil.io/)
* [NumPy](http://www.numpy.org/)
* [TensorFlow >= 2.0](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf)
* [TensorFlow Datasets](https://www.tensorflow.org/datasets/)
* [tqdm](https://tqdm.github.io/)

## Datasets
* [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
* [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
* [TF Flowers](http://download.tensorflow.org/example_images/flower_photos.tgz)
* [Oxford Flowers 102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)
* [Oxford-IIIT pet](http://www.robots.ox.ac.uk/~vgg/data/pets/)

## Usage
**Install requirements**
```
$ pip install -r requirements.txt
```

**Train model**
```
$ python main.py -dataset celeb_a -batch_size 64 -image_size 64
```

## References
[1] [Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028) 
