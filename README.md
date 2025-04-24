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
[!image_alt](https://github.com/sabarishraja/Project-2-Gradient-Boosting-Classifier-/blob/main/Working%20of%20gradient_boosting%20classifier.png?raw=true)
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
## 1. What does the model you have implemented do and when should it be used?
* The model implements a custom gradient boosting classifier for binary classification tasks.
* It uses decision trees as base learners to iteratively improve predictions.
* The model minimizes errors by focusing on residuals (differences between actual and predicted values).
* It predicts probabilities using the sigmoid function and converts them into binary class labels.
* The learning rate controls the contribution of each tree, ensuring gradual and stable learning.
* It also offers a tunable bias–variance trade-off: increasing n_estimators or max_depth reduces bias but may lead to overfitting, while decreasing learning_rate yields more conservative updates, often used in conjunction with more trees.
  
## 2. How did you test your model to determine if it is working reasonably correctly?

## 3. What parameters have you exposed to users of your implementation in order to tune performance? (Also perhaps provide some basic usage examples.)
1.	n_estimators (int, default=100)
o	Number of boosting rounds (i.e., number of trees to build).
o	More estimators can capture complex patterns but may lead to overfitting.
o	Result Impact
	Increase to better fit of the data, Having Lower Bias
	Decrease for faster training but may lead to underfitting
2.	learning_rate (float, default=0.1)
o	Shrinks the contribution of each tree.
o	Lower values require more trees but can improve generalization.
	Increase for faster convergence but might overfit quickly
	Decrease for slower training better generalization 
3.	max_depth (int, default=3)
o	Maximum depth of individual regression trees.
o	Controls the complexity of the weak learners.
	Increase leads to learning more complex relationships but might overfit 
	Decrease might improve the speed of training but might miss important patterns.

## 4. Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental?
* The implementation may struggle with very large datasets due to the inefficiency of custom decision tree splitting and recursion.
* It may have trouble with high-dimensional data because the best split calculation does not scale well with many features.
* Handling missing values is not implemented, so inputs with missing data could cause errors or incorrect splits.
* The model assumes binary classification and cannot handle multi-class problems without significant modifications.
