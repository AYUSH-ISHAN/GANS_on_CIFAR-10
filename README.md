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





