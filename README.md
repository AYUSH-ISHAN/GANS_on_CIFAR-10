# APPLYING GAN ON CIFAR-10 IMAGE DATASET

# INTRODUCTION:

GAN consist of two networks:
<ul>
  <li>A discriminator D receive input from training data and generated data. Its job is to learn how to distinguish between these two inputs.</li>
  <li>A generator G generate samples from a random noise Z. Generator objective is to generate a sample that is as real as possible it could not be distinguished by        Discriminator.</li>
<br>
<img src = "https://github.com/AYUSH-ISHAN/GANS_on_CIFAR-10/blob/main/GANS.jpg" height = "360" width = "660"/>
  
# DATASET:

<h3><B>The CIFAR-10 dataset</B></h3>
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.<br>
<br>
<p>The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class.</p>
The classes are completely mutually exclusive. There is no overlap between automobiles and trucks. "Automobile" includes sedans, SUVs, things of that sort. "Truck" includes only big trucks. Neither includes pickup trucks.

# MODEL:
  
  <B>1. Descriminator Model:</B>
  
                          _________________________________________________________________
                          Layer (type)                 Output Shape              Param #
                          =================================================================
                          conv2d_1 (Conv2D)            (None, 32, 32, 64)        1792
                          _________________________________________________________________
                          leaky_re_lu_1 (LeakyReLU)    (None, 32, 32, 64)        0
                          _________________________________________________________________
                          conv2d_2 (Conv2D)            (None, 16, 16, 128)       73856
                          _________________________________________________________________
                          leaky_re_lu_2 (LeakyReLU)    (None, 16, 16, 128)       0
                          _________________________________________________________________
                          conv2d_3 (Conv2D)            (None, 8, 8, 128)         147584
                          _________________________________________________________________
                          leaky_re_lu_3 (LeakyReLU)    (None, 8, 8, 128)         0
                          _________________________________________________________________
                          conv2d_4 (Conv2D)            (None, 4, 4, 256)         295168
                          _________________________________________________________________
                          leaky_re_lu_4 (LeakyReLU)    (None, 4, 4, 256)         0
                          _________________________________________________________________
                          flatten_1 (Flatten)          (None, 4096)              0
                          _________________________________________________________________
                          dropout_1 (Dropout)          (None, 4096)              0
                          _________________________________________________________________
                          dense_1 (Dense)              (None, 1)                 4097
                          =================================================================
                          Total params: 522,497
                          Trainable params: 522,497
                          Non-trainable params: 0
                          _________________________________________________________________
  <B>2. Generator Model:</B><br>

  
                          _________________________________________________________________
                          Layer (type)                 Output Shape              Param #
                          =================================================================
                          dense_1 (Dense)              (None, 4096)              413696
                          _________________________________________________________________
                          leaky_re_lu_1 (LeakyReLU)    (None, 4096)              0
                          _________________________________________________________________
                          reshape_1 (Reshape)          (None, 4, 4, 256)         0
                          _________________________________________________________________
                          conv2d_transpose_1 (Conv2DTr (None, 8, 8, 128)         524416
                          _________________________________________________________________
                          leaky_re_lu_2 (LeakyReLU)    (None, 8, 8, 128)         0
                          _________________________________________________________________
                          conv2d_transpose_2 (Conv2DTr (None, 16, 16, 128)       262272
                          _________________________________________________________________
                          leaky_re_lu_3 (LeakyReLU)    (None, 16, 16, 128)       0
                          _________________________________________________________________
                          conv2d_transpose_3 (Conv2DTr (None, 32, 32, 128)       262272
                          _________________________________________________________________
                          leaky_re_lu_4 (LeakyReLU)    (None, 32, 32, 128)       0
                          _________________________________________________________________
                          conv2d_1 (Conv2D)            (None, 32, 32, 3)         3459
                          =================================================================
                          Total params: 1,466,115
                          Trainable params: 1,466,115
                          Non-trainable params: 0
                          _________________________________________________________________
  

# RESULTS:
  




