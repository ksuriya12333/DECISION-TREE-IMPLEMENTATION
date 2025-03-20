# DECISION-TREE-IMPLEMENTATION-TASK1
# SENTIMENT ANALYSIS WITH NLP-TASK2
# IMAGE CLASSIFICATION MODEL-TASK3
# RECOMMENDATION SYSTEM-TASK4
*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: KONDAKAMARLA SURIYA

*INTERN ID*: CT12PBR

*DOMAIN*: MACHINE LEARNING

*DURATION*: 8 WEEKS

*MENTOR*: NEELA SANTOSH KUMAR

*DESCRIPTION*
##
# Task 1: Decision Tree Classifier (CodTech_IT_Task1.ipynb)
This task involves training a Decision Tree Classifier, a machine-learning model used to classify data based on given features. The Iris dataset is used, which contains information about three types of flowers: Setosa, Versicolor, and Virginica. Each flower is described using four features:

Sepal length

Sepal width

Petal length

Petal width

Steps in the Notebook:
Loading the Dataset

The dataset is loaded into a Pandas DataFrame.

The target labels (flower species) are assigned numerical values.

Splitting the Data

The dataset is divided into training and testing sets using train_test_split().

This ensures that the model learns from one portion of the data and is tested on unseen data to check its accuracy.

Training a Decision Tree Model

A Decision Tree Classifier is initialized and trained using the training data.

The model works by splitting the data at different feature points to classify new samples.

Evaluating the Model

The trained model is tested on the test set, and its accuracy is measured.

A confusion matrix and classification report are generated to analyze model performance.

Visualizing the Decision Tree

The Decision Tree is plotted to show how the model makes decisions.

This helps in understanding which features play an important role in classification.

# Task 2: Sentiment Analysis using Logistic Regression (CodTech_IT_Task2.ipynb)
This task is about analyzing customer reviews to determine whether they are positive or negative using Logistic Regression. Sentiment analysis is widely used in applications like product reviews, movie reviews, and customer feedback analysis.

Steps in the Notebook:
Creating a Sample Dataset

A small dataset of customer reviews is created, where each review has a corresponding sentiment:

1 (positive)

0 (negative)

Preprocessing the Text Data

Since machine learning models cannot work directly with text, the reviews are converted into numerical form using TF-IDF (Term Frequency - Inverse Document Frequency).

TF-IDF helps identify important words in the text.

Training a Logistic Regression Model

The Logistic Regression model is trained on the dataset.

Logistic Regression is a classification algorithm that assigns probabilities to different classes (positive or negative).

Evaluating the Model

The model’s accuracy is calculated.

A confusion matrix and classification report are used to analyze how well the model classifies positive and negative reviews.

Visualizing the Results

A heatmap is plotted to visualize the confusion matrix, making it easier to understand the model's performance.

# Task 3: Image Classification using CNN (CodeIT_Tech_Task_3.ipynb)
This task focuses on image classification using Convolutional Neural Networks (CNNs). CNNs are widely used for image processing tasks such as face recognition, medical imaging, and self-driving cars.

Steps in the Notebook:
Loading the CIFAR-10 Dataset

The CIFAR-10 dataset is used, which contains 60,000 images belonging to 10 different categories such as airplanes, cars, cats, and dogs.

Preprocessing the Data

The pixel values of the images are normalized to improve training efficiency.

Class labels are defined so the model knows what each image represents.

Building a CNN Model

The CNN model consists of multiple layers:

Convolutional Layers: Extract features from the images.

Pooling Layers: Reduce the size of feature maps to improve efficiency.

Fully Connected Layers: Make predictions based on extracted features.

Training the Model

The model is compiled using the Adam optimizer and categorical cross-entropy loss function.

The training process involves feeding the images into the network, allowing it to learn patterns.

Evaluating the Model

The trained model is tested on unseen images to measure its accuracy.

Predictions are made on test images, and their actual and predicted labels are compared.

# Task 4: Movie Recommendation System using SVD (Cod_Tech_IT_task_4.ipynb)
This task involves building a movie recommendation system using Singular Value Decomposition (SVD), a popular technique in collaborative filtering-based recommendation systems. Netflix, Amazon, and YouTube use similar approaches to suggest content to users.

Steps in the Notebook:
Loading the MovieLens Dataset

The MovieLens 100K dataset is used, which contains 100,000 user ratings for movies.

Each entry includes:

User ID

Movie ID

Rating (1 to 5 stars)

Timestamp

Preparing the Data

The dataset is prepared for Surprise, a Python library for recommendation systems.

A Reader object is created to define the rating scale (1 to 5 stars).

Splitting the Data

The dataset is split into training and testing sets to evaluate the model’s performance.

Training the SVD Model

The SVD model is trained on the dataset.

SVD identifies patterns in user preferences by breaking down the user-item interaction matrix into smaller components.

Making Predictions

The trained model predicts ratings for movies the user hasn’t seen.

These predictions help recommend movies to users.

Evaluating the Model

The model is evaluated using:

RMSE (Root Mean Squared Error)

MAE (Mean Absolute Error)

These metrics help measure how accurate the recommendations are.

Conclusion
These tasks cover various areas of machine learning and data science:

# Task 1 teaches how to use Decision Trees for classification.

# Task 2 focuses on sentiment analysis using text classification.

# Task 3 explores image classification using deep learning (CNNs).

# Task 4 demonstrates recommendation systems for personalized suggestions.

Each task applies fundamental machine learning and deep learning techniques that are useful in real-world applications like customer review analysis, automated classification, and personalized content recommendations.

# OUTPUTS
https://github.com/ksuriya12333/DECISION-TREE-IMPLEMENTATION/issues/1#issue-2935220643

##

