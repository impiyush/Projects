# Recognize US Dollar Bills

***A Flask app to detect the US dollar bill in the image.***

## Dataset

- Images of US Dollar bills
- Dataset created using images taken from my phone
- The dataset is in the [imagenet](http://image-net.org/about-overview) format, i.e., set of multiple folders, where each folder name is a class label and the folder itself contains images corresponding to that class
- There are a total of `165 images` in the dataset belonging to `6 classes`
- Data augmentation techniques were used while training, to supplement the data for training purposes
- Images were `resized to (224, 224)` and image pixel values `rescaled to be between [0,1]` before being used for training and validation
- `80%` of data was used for training and remaining `20%` used for validation

### Dataset Summary

| | |
|:-|:-:|
|Number of images | 165  |
|Number of classes   | 6   |
|Image image dimensions <br/>(used for training and validation)   | (224 x 224 x 3)  |
|Training split | 80% |
|Validation split | 20% |
|Classes | [1, 5, 10, 20, 50, 100]|
| | |

## Model Features

- Model trained using `TensorFlow v2.1.0`
- Uses `data augmentation` techniques to supplement the data
- Uses `Transfer Learning` with weights of pre-trained Convolutional Network (Resnet50 V2)
- Fine tuning the model by un-freezing and retraining the `top 70 layers of Resnet50`
- Inference using model parameters saved in `.h5` format

### Model Architecture

![Recognize US Bills - Model Architecture](images/model_arch.jpg)

### Model Performance

![Accuracy-Loss plots](images/accuracy_loss_plots.jpg)

## App Features

- Python Flask based web app
- Mobile-friendly UI
- Supports image drag-and-drop

## References

- [Building a simple Keras + deep learning REST API](https://blog.keras.io/building-a-simple-keras-deep-learning-rest-api.html)
- [Deploy Keras Model with Flask as Web App in 10 Minutes](https://github.com/mtobeiyf/keras-flask-deploy-webapp)
- [TensorFlow 2.0 Tutorial 01: Basic Image Classification](https://lambdalabs.com/blog/tensorflow-2-0-tutorial-01-image-classification-basics/)
- [Trasfer Learning with a pretrained ConvNet](https://www.tensorflow.org/tutorials/images/transfer_learning)
