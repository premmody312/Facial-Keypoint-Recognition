# Facial-Keypoint-Recognition

Use image processing techniques and deep learning to recognize faces and facial keypoints, such as the location of the eyes and mouth on a face.

## Dependencies:

numpy pandas matplotlib opencv pytorch

## Installing dependencies:

> pip install numpy pandas matplotlib opencv-python | pip install torch===1.5.1 torchvision===0.6.1 -f https://download.pytorch.org/whl/torch_stable.html

## Screenshots:
![alt text](https://github.com/premmody312/Facial-Keypoint-Recognition/blob/master/images/haar_cascade_ex.png)
___
![alt text](https://github.com/premmody312/Facial-Keypoint-Recognition/blob/master/images/face_filter_ex.png)
___


## Algorithm:

### Loading and Visualizing the dataset

Facial keypoints (also called facial landmarks) are the small magenta dots shown on each of the faces in the image above. In each training and test image, there is a single face and 68 keypoints, with coordinates (x, y), for that face. These keypoints mark important areas of the face: the eyes, corners of the mouth, the nose, etc. These keypoints are relevant for a variety of tasks, such as face filters, emotion recognition, pose recognition, and so on. Here they are, numbered, and you can see that specific ranges of points match different portions of the face.

#### Load and Visualize Data
The first step in working with any dataset is to become familiar with your data; you'll need to load in the images of faces and their keypoints and visualize them! This set of image data has been extracted from the YouTube Faces Dataset, which includes videos of people in YouTube videos. These videos have been fed through some processing steps and turned into sets of image frames containing one face and the associated keypoints.

#### Training and Testing Data
This facial keypoints dataset consists of 5770 color images. All of these images are separated into either a training or a test set of data.

3462 of these images are training images, for you to use as you create a model to predict keypoints.
2308 are test images, which will be used to test the accuracy of your model.

#### Transforms
Now, the images above are not of the same size, and neural networks often expect images that are standardized; a fixed size, with a normalized range for color ranges and coordinates, and (for PyTorch) converted from numpy lists and arrays to Tensors.

Therefore, we will need to write some pre-processing code. Let's create four transforms:

Normalize: to convert a color image to grayscale values with a range of [0,1] and normalize the keypoints to be in a range of about [-1, 1]
Rescale: to rescale an image to a desired size.
RandomCrop: to crop an image randomly.
ToTensor: to convert numpy images to torch images.

### Defining the Convolutional Neural Network Architecture:

Net(
  (conv1): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1))
  (conv1_bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
  (conv2_bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (conv3): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1))
  (conv3_bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (conv4): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1))
  (conv4_bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (fc1): Linear(in_features=36864, out_features=4000, bias=True)
  (fc1_bn): BatchNorm1d(4000, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (fc2): Linear(in_features=4000, out_features=1000, bias=True)
  (fc2_bn): BatchNorm1d(1000, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (fc3): Linear(in_features=1000, out_features=136, bias=True)
  (dropout): Dropout(p=0.4)
)

### Face and Facial Keypoint detection:

After you've trained a neural network to detect facial keypoints, you can then apply this network to any image that includes faces. The neural network expects a Tensor of a certain size as input and, so, to detect any face, you'll first have to do some pre-processing.

Detect all the faces in an image using a face detector (we'll be using a Haar Cascade detector in this notebook).
Pre-process those face images so that they are grayscale, and transformed to a Tensor of the input size that your net expects. This step will be similar to the data_transform you created and applied in Notebook 2, whose job was tp rescale, normalize, and turn any iimage into a Tensor to be accepted as input to your CNN.
Use your trained model to detect facial keypoints on the image

### Keypoints Applications:

Using your trained facial keypoint detector, you can now do things like add filters to a person's face, automatically. In this optional notebook, you can play around with adding sunglasses to detected face's in an image by using the keypoints detected around a person's eyes.


## Conclusion:
Thus we gain knowledge in defining and using the CNN Architecture by referring to many research papers and were able to apply this knowledge to generate face filters like in Snapchat
