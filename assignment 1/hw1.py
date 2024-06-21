###### Your ID ######
# ID1: 208663534
# ID2: 206480402
#####################

# imports
import numpy as np
import pandas as pd


def preprocess(X, y):
    """
    Perform mean normalization on the features and true labels.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - X: The mean normalized inputs.
    - y: The mean normalized labels.
    """

    # Creating the variables that are part of the normalization.
    X_max = np.max(X, axis=0)
    X_min = np.min(X, axis=0)
    X_mean = np.mean(X, axis=0)

    y_max = np.max(y, axis=0)
    y_min = np.min(y, axis=0)
    y_mean = np.mean(y, axis=0)

    # Normalization of the features and true labels.
    X = (X - X_mean) / (X_max - X_min)
    y = (y - y_mean) / (y_max - y_min)

    return X, y


def apply_bias_trick(X):
    """
    Applies the bias trick to the input data.

    Input:
    - X: Input data (m instances over n features).

    Returns:
    - X: Input data with an additional column of ones in the
        zeroth position (m instances over n+1 features).
    """

    # Creating a unity vector with the size of X.
    m = len(X)
    ones_column = np.ones((m))

    # Connecting two vectors to one vector with 2 columns.
    X = np.c_[ones_column, X]

    return X


def compute_cost(X, y, theta):
    """
    Computes the average squared difference between an observation's actual and
    predicted values for linear regression.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: the parameters (weights) of the model being learned.

    Returns:
    - J: the cost associated with the current set of parameters (single number).
    """

    J = 0  # We use J for the cost.

    # Creating the variables for the equation.
    m = len(X)
    h = X.dot(theta)

    # Calculate the value of the equation.
    J =  np.sum((h - y) ** 2) / (2 * m)

    return J


def gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of the model using gradient descent using
    the training set. Gradient descent is an optimization algorithm
    used to minimize some (loss) function by iteratively moving in
    the direction of steepest descent as defined by the negative of
    the gradient. We use gradient descent to update the parameters
    (weights) of our model.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """

    theta = theta.copy()  # optional: theta outside the function will not change
    J_history = []  # Use a python list to save the cost value in every iteration

    m = len(X)

    # A loop that runs a formula that updates theta so that we have the lowest cost.
    for i in range(num_iters):
        h = X.dot(theta)
        error = h - y
        theta = theta - (alpha / m) * np.dot(X.T, error)
        J_history.append(compute_cost(X, y, theta))

    return theta, J_history


def compute_pinv(X, y):
    """
    Compute the optimal values of the parameters using the pseudoinverse
    approach as you saw in class using the training set.

    #########################################
    #### Note: DO NOT USE np.linalg.pinv ####
    #########################################

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - pinv_theta: The optimal parameters of your model.
    """

    pinv_theta = []

    # Calculation according to the formula we saw in the lecture.
    X_transpose = X.T
    mult1 = np.dot(X_transpose, X)
    reverse = np.linalg.pinv(mult1)
    mult2 = np.dot(reverse, X_transpose)
    pinv_theta = np.dot(mult2, y)

    return pinv_theta


def efficient_gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of your model using the training set, but stop
    the learning process once the improvement of the loss value is smaller
    than 1e-8. This function is very similar to the gradient descent
    function you already implemented.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """

    theta = theta.copy()  # optional: theta outside the function will not change
    J_history = []  # Use a python list to save the cost value in every iteration

    m = len(X)

    # We calculate the cost function until we reach that the cost value is less than 1e-8.
    for i in range(num_iters):
        h = X.dot(theta)
        error = h - y
        theta = theta - (alpha / m) * np.dot(X.T, error)
        J_history.append(compute_cost(X, y, theta))
        if i > 0 and J_history[i - 1] - J_history[i] < 1e-8:
            break

    return theta, J_history


def find_best_alpha(X_train, y_train, X_val, y_val, iterations):
    """
    Iterate over the provided values of alpha and train a model using
    the training dataset. maintain a python dictionary with alpha as the
    key and the loss on the validation set as the value.

    You should use the efficient version of gradient descent for this part.

    Input:
    - X_train, y_train, X_val, y_val: the training and validation data
    - iterations: maximum number of iterations

    Returns:
    - alpha_dict: A python dictionary - {alpha_value : validation_loss}
    """

    alphas = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 2, 3]
    alpha_dict = {}  # {alpha_value: validation_loss}

    # A loop that goes through all the alphas and puts the alpha with its cost value in a dictionary.
    for alpha in alphas:
        theta = np.ones(X_train.shape[1])
        theta = efficient_gradient_descent(X_train, y_train, theta, alpha, iterations)[0]
        alpha_dict[alpha] = compute_cost(X_val, y_val, theta)

    return alpha_dict


def forward_feature_selection(X_train, y_train, X_val, y_val, best_alpha, iterations):
    """
    Forward feature selection is a greedy, iterative algorithm used to
    select the most relevant features for a predictive model. The objective
    of this algorithm is to improve the model's performance by identifying
    and using only the most relevant features, potentially reducing overfitting,
    improving accuracy, and reducing computational cost.

    You should use the efficient version of gradient descent for this part.

    Input:
    - X_train, y_train, X_val, y_val: the input data without bias trick
    - best_alpha: the best learning rate previously obtained
    - iterations: maximum number of iterations for gradient descent

    Returns:
    - selected_features: A list of selected top 5 feature indices
    """
    selected_features = []

    # Implementing the forward feature selection by the guidelines.
    for i in range(5):
        np.random.seed(42)
        theta_random = np.random.random(size=i + 2)
        fun_cost = []
        for j in range(X_train.shape[1]):
            if j not in selected_features:

                # Adding temporarily the feature.
                selected_features.append(j)

                # Modifying the training and validation data according to the selected features.
                X_train_selcted = apply_bias_trick(X_train[:, selected_features])
                X_val_selcted = apply_bias_trick(X_val[:, selected_features])

                # Calculating theta and cost function.
                theta, cost = efficient_gradient_descent(X_train_selcted, y_train, theta_random, best_alpha, iterations)
                cost_value = compute_cost(X_val_selcted, y_val, theta)

                # Adding feature with its cost value and removing the feature from selected_features.
                fun_cost.append((j, cost_value))
                selected_features.pop()

        # Finding the feature whose cost value is the lowest.
        best_feature = min(fun_cost, key=lambda x: x[1])[0]
        selected_features.append(best_feature)

    return selected_features


def create_square_features(df):
    """
    Create square features for the input data.

    Input:
    - df: Input data (m instances over n features) as a dataframe.

    Returns:
    - df_poly: The input data with polynomial features added as a dataframe
               with appropriate feature names
    """

    df_poly = df.copy()
    lst = []

    # Go through all the features in the data.
    for i, col in enumerate(df_poly.columns):
        for new_col in df_poly.columns[i:]:

            # Checking which feature we are in now the same feature or different feature.
            if col != new_col:
                feature = col + '*' + new_col
            else:
                feature = col + '^2'

            # Calculating the value of the new feature and adding the new feature's name and value to the list.
            new_col_list = df_poly[col] * df_poly[new_col]
            new_col_list.name = feature
            lst.append(new_col_list)

    # Attaching the new frame with the list of the new features and their values.
    df_poly = pd.concat([df_poly] + lst, axis=1)

    return df_poly