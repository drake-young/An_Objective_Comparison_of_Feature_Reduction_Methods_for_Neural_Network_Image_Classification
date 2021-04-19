# An Objective Comparison of Feature Reduction Methods for Neural Network Image Classification

This repository contains the Jupyter Notebooks used for experimentation and results aggregation within the MS Project Report "An Objective Comparison of Feature Reduction Methods for Neural Network Image Classification" by Drake Young. Each notebook is fully documented for each cell of code used in the experiments explaining the code, its purpose, and the results. This readme explains which experiment the jupyter notebook performs.

All experiments conducted within the provided Jupyter notebooks are conducted through Anaconda using Python 3.7, and accelerated with the use of an NVIDIA RTX 2070 graphics card for a tensorflow-gpu backend of keras. Two different experiment datasets were used, both hosted publicly on Kaggle. We utilized the [Dogs & Cats Images](https://www.kaggle.com/chetankv/dogs-cats-images) for binary classification experimentation, and the [Intel Image Classification](https://www.kaggle.com/puneet6060/intel-image-classification) dataset for multi-class classification experiments. Please keep in mind that the purpose of this experiment is to show the direct impacts on performance when changing the preprocessing techniques used, not to achieve the highest performing model in all contexts. That is, we are more concerned with showing changes in performance across the experiments than across any specific experiment.

For a complete context into the problem, please see the project report (which will be uploaded to this repository after the graduate committee has made a decision on the status). This repository only contains the Jupyter Notebooks of the experiments, excluding the generated results files and the input datasets. The reasoning for this exclusion was due to the GitHub size limitations, as many of the traind models were 250MB-1.1GB in size. In order to access the entire project including dataset and results, please see the following Google Drive folder: [Google Drive Link](https://drive.google.com/drive/folders/1KPBMunLt7tYlkAyQJSSTqPbe3Vu2zZ9W?usp=sharing). 

There are two different colvolutional neural network model architectures which are experimented within the project. The first architecture is referred to as "Custom Model," which is a shallow CNN architecture with significantly fewer trainable parameters. The second model architecture examined is the VGG19 model architecutre. The model architecture diagrams are shown below.

## Model: Custom

The following image depicts the CNN model architecture diagram for the custom CNN model used in experimentations in the future files listed below. 

![Custom Model Architecture](custom_model.png?raw=true "Custom Model Architecture")

## Model: VGG19

The following image depicts the VGG19 model architecture for CNN experiments labeled as "VGG19" or "vgg19" within the experiment files. 

![VGG19 Model Architecture](VGG19_model.png?raw=true "VGG19 Model Architecture")

## File: LSH\_CatVDog\_CustomModel.ipynb

Using a Locality-Sensitive Hashing (LSH) approach to preprocessing the images prior to training the neural network, the custom convolutional neural network (CNN) architecture is trained and evaluated for experimentation within this notebook. The Dogs & Cats Images dataset is used for this experiment. Both the training history for loss and accuracy as well as the overhead time costs for inference, preprocessing, and training on the dataset.

## File: LSH\_CatVDog\_VGG19.ipynb

Using a Locality-Sensitive Hashing (LSH) approach to preprocessing the images prior to training the neural network, the VGG19 convolutional neural network (CNN) architecture is trained and evaluated for experimentation within this notebook. The Dogs & Cats Images dataset is used for this experiment. Both the training history for loss and accuracy as well as the overhead time costs for inference, preprocessing, and training on the dataset.

## File: LSH\_Intel\_CustomModel.ipynb

Using a Locality-Sensitive Hashing (LSH) approach to preprocessing the images prior to training the neural network, the custom convolutional neural network (CNN) architecture is trained and evaluated for experimentation within this notebook. The Intel Image Classification dataset is used in this experiment. Both the training history for loss and accuracy as well as the overhead time costs for inference, preprocessing, and training on the dataset.

## File: LSH\_Intel\_VGG19.ipynb

Using a Locality-Sensitive Hashing (LSH) approach to preprocessing the images prior to training the neural network, the VGG19 convolutional neural network (CNN) architecture is trained and evaluated for experimentation within this notebook. The Intel Image Classification dataset is used in this experiment. Both the training history for loss and accuracy as well as the overhead time costs for inference, preprocessing, and training on the dataset.

## File: NoPreprocessing\_CatVDog\_CustomModel.ipynb

Using no preprocessing on the dataset, this experiment acts as a baseline for this configuration of model and dataset. The custom convolutional neural network (CNN) architecture is trained and evaluated for experimentation within this notebook. The Dogs & Cats Images dataset is used for this experiment. Both the training history for loss and accuracy as well as the overhead time costs for inference, preprocessing, and training on the dataset.

## File: NoPreprocessing\_CatVDog\_VGG19.ipynb

Using no preprocessing on the dataset, this experiment acts as a baseline for this configuration of model and dataset. The VGG19 convolutional neural network (CNN) architecture is trained and evaluated for experimentation within this notebook. The Dogs & Cats Images dataset is used for this experiment. Both the training history for loss and accuracy as well as the overhead time costs for inference, preprocessing, and training on the dataset.

## File: NoPreprocessing\_Intel\_CustomModel.ipynb

Using no preprocessing on the dataset, this experiment acts as a baseline for this configuration of model and dataset. The custom convolutional neural network (CNN) architecture is trained and evaluated for experimentation within this notebook. The Intel Image Classification dataset is used for this experiment. Both the training history for loss and accuracy as well as the overhead time costs for inference, preprocessing, and training on the dataset.

## File: NoPreprocessing\_Intel\_VGG19.ipynb

Using no preprocessing on the dataset, this experiment acts as a baseline for this configuration of model and dataset. The VGG19 convolutional neural network (CNN) architecture is trained and evaluated for experimentation within this notebook. The Intel Image Classification dataset is used for this experiment. Both the training history for loss and accuracy as well as the overhead time costs for inference, preprocessing, and training on the dataset.

## File: PCA\_CatVDog\_CustomModel.ipynb

Using a Principal Component Analysis (PCA) approach to preprocessing the images prior to training the neural network, the custom convolutional neural network (CNN) architecture is trained and evaluated for experimentation within this notebook. The Dogs & Cats Images dataset is used for this experiment. Both the training history for loss and accuracy as well as the overhead time costs for inference, preprocessing, and training on the dataset.

## File: PCA\_CatVDog\_VGG19.ipynb

Using a Principal Component Analysis (PCA) approach to preprocessing the images prior to training the neural network, the VGG19 convolutional neural network (CNN) architecture is trained and evaluated for experimentation within this notebook. The Dogs & Cats Images dataset is used for this experiment. Both the training history for loss and accuracy as well as the overhead time costs for inference, preprocessing, and training on the dataset.

## File: PCA\_Intel\_CustomModel.ipynb

Using a Principal Component Analysis (PCA) approach to preprocessing the images prior to training the neural network, the custom convolutional neural network (CNN) architecture is trained and evaluated for experimentation within this notebook. The Intel Image Classification dataset is used for this experiment. Both the training history for loss and accuracy as well as the overhead time costs for inference, preprocessing, and training on the dataset.

## File: PCA\_Intel\_VGG19.ipynb

Using a Principal Component Analysis (PCA) approach to preprocessing the images prior to training the neural network, the VGG19 convolutional neural network (CNN) architecture is trained and evaluated for experimentation within this notebook. The Intel Image Classification dataset is used for this experiment. Both the training history for loss and accuracy as well as the overhead time costs for inference, preprocessing, and training on the dataset.

## File: requirements.txt

This text file contains a list of all python packages installed in the computation environment which the experiments were run on. This requirements text file can be used as input for a pip install command in order to install the necessary dependencies with the same versions as those used in the experimentations. 

## File: Results\_Aggregation.ipynb

This notebook aggregates the results produced through all 32 experiments conducted, generating the relevant data visualization, producing all of the graphs and tables used within the report with the exception of the model architecture diagrams. No experimentations are run within this notebook, only aggregation of the results logged by the other notebooks. If this notebook is run before all 32 experiments have completed, errors will be produced, as the necessary log files would not be present. 

## File: SmallGrayReg\_CatVDog\_CustomModel.ipynb

The preprocessing approach used in this experiment involves converting the image dataset to grayscale and resizing to dimensions less than 100x100 which are factors of the original image size, and finally, the pixel values are regularized by dividing their value by 255. The custom convolutional neural network (CNN) architecture is trained and evaluated for experimentation within this notebook. The Dogs & Cats Images dataset is used for this experiment. Both the training history for loss and accuracy as well as the overhead time costs for inference, preprocessing, and training on the dataset.

## File: SmallGrayReg\_CatVDog\_VGG19.ipynb

The preprocessing approach used in this experiment involves converting the image dataset to grayscale and resizing to dimensions less than 100x100 which are factors of the original image size, and finally, the pixel values are regularized by dividing their value by 255. The VGG19 convolutional neural network (CNN) architecture is trained and evaluated for experimentation within this notebook. The Dogs & Cats Images dataset is used for this experiment. Both the training history for loss and accuracy as well as the overhead time costs for inference, preprocessing, and training on the dataset.

## File: SmallGrayReg\_Intel\_CustomModel.ipynb

The preprocessing approach used in this experiment involves converting the image dataset to grayscale and resizing to dimensions less than 100x100 which are factors of the original image size, and finally, the pixel values are regularized by dividing their value by 255. The custom convolutional neural network (CNN) architecture is trained and evaluated for experimentation within this notebook. The Intel Image Classification dataset is used for this experiment. Both the training history for loss and accuracy as well as the overhead time costs for inference, preprocessing, and training on the dataset.

## File: SmallGrayReg\_Intel\_VGG19.ipynb

The preprocessing approach used in this experiment involves converting the image dataset to grayscale and resizing to dimensions less than 100x100 which are factors of the original image size, and finally, the pixel values are regularized by dividing their value by 255. The VGG19 convolutional neural network (CNN) architecture is trained and evaluated for experimentation within this notebook. The Intel Image Classification dataset is used for this experiment. Both the training history for loss and accuracy as well as the overhead time costs for inference, preprocessing, and training on the dataset.

## File: TUNING\_

There exist 16 different jupyter notebooks which begin with `TUNING_` followed by the name of one of the previously listed experiment notebooks. These experiments are near-identical to the counterparts with a shared name. The difference between these experiments and the previously listed is that these experiments conduct hyperparameter tuning on the learning rate in order to optimize validation accuracy within 10 epochs of training utilizing the kerastuner library for this. 

## Closing Remarks

This project is used as the experimental programs for my MS Term project to be presented to the graduate committee. If you intend to use the experiments provided, please consider citing the project report associated with these experiments, which will be provided after review by the graduate committee. Thank you.
