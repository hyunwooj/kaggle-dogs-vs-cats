# kaggle-dogs-vs-cats
In [this competetion](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition), I am given images of dogs and cats and I need to predict the probability that an image is a dog.

## Take #1
I resized images to 28x28 since the resolutions of images are high and each image has a different resolution. A network consisted of 2 convolutional layers and 2 fully-connected layers. Mini-batch gradient descent is used to optimize. Loss function is a cross entropy.

The result of this network is very bad. It is even worse than predicting all images is 50% dog.

## Take #2
I increased the input size to 64*64 but it did not improved the result at all.

## Input Investigation
- Training
  - 25,000 images: 12,500 dogs and 12,500 cats
  - Max, mean, min of dimensions:
    - Width: 1050, 404, 42
    - Height: 768, 360, 32
- Test
  - 12,500 images
  - Max, mean, min of dimensions:
    - Width: 500, 404, 37
    - Height: 500, 360, 44
