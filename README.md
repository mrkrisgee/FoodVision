# FoodVision

FoodVision builds upon the groundwork laid by the [FoodVisionMini](https://github.com/mrkrisgee/FoodVisionMini) experiment, which delved into the intricacies of Convolutional Neural Network (CNN) architectures. With a foundation established, this project ventures further by training a model on the *[Food101 Image Data Set](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)*. Derived from insights gained in FoodVisionMini, the architecture employed here not only trains on the Food101 dataset but also extends its capabilities to make predictions on its own data and custom images sourced from the web.

## Dataset Description

The Food101 dataset consists of 101,000 images of food divided into 101 categories, with 1,000 images per category. These images represent a wide variety of foods, spanning different cuisines and dishes. This diverse dataset serves as the foundation for training and evaluating the FoodVision model, enabling it to recognize and classify various types of food accurately.

## Technical Background

Building upon the insights garnered from FoodVisionMini, FoodVision continues its exploration of Convolutional Neural Network (CNN) architectures for image classification tasks. CNNs are particularly adept at analyzing visual data, making them well-suited for tasks such as food classification. Leveraging the power of deep learning, FoodVision aims to develop a robust model capable of accurately identifying various food items from images.

## Model Architecture

Based on the findings from the FoodVisionMini experiment, the **[EfficientNet-B2](https://pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_b2.html#efficientnet-b2)** model emerged as the standout performer, thus becoming the cornerstone of the subsequent FoodVision experiment. The EfficienNet-B2 model, along with pre-trained weights, is readily accesible from the PyTorch library. The architecture, including its layers and parameters, have been summarized [here](https://github.com/mrkrisgee/FoodVision/blob/main/model_architectures/effnetb2.png). 

## Detailed Methodology of steps performed in the code

### Data Preparation

1. **Download the *[Food101 Image Data Set](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)*.**
   - Obtain the original dataset.
       
### Preprocessing and Data Augmentation

1. **Prepare DataLoaders:**
   - Implement Train and Test DataLoaders to manage large datasets in batches.
   - Apply data augmentation techniques such as resizing and shuffling to increase the diversity of the training data.
   - Normalize the image data to standardize the input.
  
### Model Training

1. **Set Up Experiment Iterations:**
   - Configure the model architectures with pre-defined and adjusted classifier heads to alight with our experimental requirements.
   - Instantiate training model.
   - Define loss functions and optimizers.
   - Train the model.
   - Track experiments using TensorBoard.
   - Save the trained model.

### Evaluation

1. **Evaluate model:**

The models were evaluated based on accuracy and loss metrics. These metrics provide insights into the performance of the models during training and testing phases. In addition to further assess the performance of the trained model, the following steps are undertaken:   
   - Load the trained model: Retrieve the saved weights and architecture of the trained EfficientNet-B2 model.
   - Instantiate a new model: Re-create the model architecture and load the trained weights to ensure consistency.
   - Make predictions with the new model: Utilize the re-instantiated model to predict food categories for a set of images.
   - Plot random images from original dataset: Randomly select images from the original Food101 dataset and visualize them.
   - Evaluate prediction accuracy: Compare the model's predictions with the ground truth labels to determine the accuracy of its classifications.
   - Plot custom images obtained online: Acquire custom food images from online sources and feed them into the model for prediction. This allows for testing the model's performance on real-world, unseen data.
   - Evaluate model's capability on unseed data: Assess the model's ability to correctly classify food items from custom images, including those it has not encountered during training. This step evaluates the model's generalization ability.
   - Create a confusion matrix: Construct a confusion matrix to compare target and predicted labels. This matrix provides a visual representation of the model's performance across different classes, highlighting which classes the model struggles to classify accurately. This analysis aids in identifying areas                                    for model improvement and fine-tuning.

## Results

A few stat graphs visualized in TensorBoard:

### Accuracy
![Accuracy](https://github.com/mrkrisgee/FoodVision/blob/main/results/Accuracy.png)

### Loss
![Loss](https://github.com/mrkrisgee/FoodVision/blob/main/results/Loss.png)

### Metric Outputs and Total Training Times for Each Model:
![Metric Outputs](https://github.com/mrkrisgee/FoodVision/blob/main/results/results.png)

The results of FoodVision can be examined in the [`results`](https://github.com/mrkrisgee/FoodVision/tree/main/results) directory.

### Usage



## Future work

## References

