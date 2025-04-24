# Contributors:
1. Madhusoodhan Tirunangur Girinarayanan (A20580122) - mtirunangurgirinaray@hawk.iit.edu
2. Mukund Sanjay Bharani (A20577945) - mbharani@hawk.iit.edu
3. Muthu Nageswaran Kalyani Narayanamoorthy (A20588118) - mkalyaninarayanamoor@hawk.iit.edu
4. Sabarish Raja Ramesh Raja (A20576363) - srameshraja@hawk.iit.edu

# <b> Introduction </b>
This project implements Gradient Boosting Classifier using decision tree as base learners. Logistic Regression is used as regression principles for binary classification. Also, sigmoid function is used for probability estimation.
Gradient Boosting Classifier belongs to ensemble learners. It combines multiple decision trees to create a strong predictive model. 
GB Classifier builds model Sequentially where new model tries to correct the errors made by previous models. This algorithm uses gradient descent to minimize the loss function by itertively improving predictions.

# <b>Working of Gradient Boosting Classifier</b>
1. Initialize the model
2. Calculate the residuls(error) between true value and current predictions.
3. Train a decision tree and fit it to the residuals. This tree will learn to correct the mistakes made by the current ensemble of predictions.
4. Add prediction of the decision tree with current prediction. Scale these updates by a learning rate to control the contribution of each tree and ensure gradual learning.
5. Repeat the process of calculating residuals, training decision trees, and updating predictions for a predefined number of iterations (or until convergence).
6. At the end, combine prediction of ll decision trees to produce the finaal output. For classification, a sigmoid function is used to convert the cumulative prediction into class probabilities.

# <b> Hyper-parameters used for optimizing the performance:
1. n_estimators => Total number of decision trees to be used in the ensemble model/
2. learning_rate => Controls the contribution of each tree.
3. max_depth => Limit the depth of each decision tree to prevent overfitting.
4. loss_function => calculate the residual errors.

## Running the Code
To run the code in this project using VS Code , follow these steps:

### 1. Set Up a Virtual Environment
Open the terminal in VS Code by navigating to View > Terminal or using the shortcut Ctrl + ` (backtick).
Create a virtual environment using Python's venv module:
```
python -m venv venv
```
Activate the virtual environment:
```
venv\Scripts\activate
```

### 2. Run the Tests
Navigate to the test_case directory and execute the test suite using pytest.
First, navigate to the tests folder.
Then the main test script using :
```
python GBTesting.py
```

## Test Output Visualizations
![image_alt](https://github.com/sabarishraja/Project-2-Gradient-Boosting-Classifier-/blob/main/Output%20images/Basic%20functionality.jpeg?raw=true)
![image_alt](https://github.com/sabarishraja/Project-2-Gradient-Boosting-Classifier-/blob/main/Output%20images/High%20learning%20rate.jpeg?raw=true)
![image_alt](https://github.com/sabarishraja/Project-2-Gradient-Boosting-Classifier-/blob/main/Output%20images/Imbalanced%20classess.jpeg?raw=true)
![image_alt](https://github.com/sabarishraja/Project-2-Gradient-Boosting-Classifier-/blob/main/Output%20images/Low%20learning%20rate.jpeg?raw=true)
![image_alt](https://github.com/sabarishraja/Project-2-Gradient-Boosting-Classifier-/blob/main/Output%20images/Shallow%20trees.jpeg?raw=true)
![image_alt](https://github.com/sabarishraja/Project-2-Gradient-Boosting-Classifier-/blob/main/Output%20images/non-linear%20boundary.jpeg?raw=true)

# Quesions:
