import numpy as np
np.random.seed(42)
import matplotlib.pyplot as plt
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)
max_iters = 2000
step_size = 0.0001


# given an input vector z, return a vector of the outputs of a logistic
# function applied to each input value
# z is a n x 1 vector
# logit_z is a n x 1 vector where logit_z[i] is the result of applying 
# the logistic function to z[i]
def logistic(z):
    logit_z = 1 / (1 + np.exp(-z))
    return logit_z

# given an input data matrix X, label vector y, and weight vector w
# compute the negative log likelihood of a logistic regression model
# using w on the data defined by X and y
# X is a n x d matrix of examples where each row corresponds to a single d-dimensional example
# y is a n x 1 vector representing the tumor labels of the examples in X
# w is a d x 1 weight vector
# nll is the value of the negative log likelihood 
def calculateNegativeLogLikelihood(X, y, w):
    nll = -np.sum(y*np.log(logistic(X@w) + 0.0000001) + (1-y)*np.log(1-logistic(X@w) + 0.0000001))
    return nll

# given an input data matrix X, tumor label vector y, maximum number of iterations max_iters, 
# and step size step_size. run max_iters of gradient descent with a step size of step_size to
# optimize a weight vector that minimize negative log likelihood on the data defined by X and y
# X is a n x d matrix of examples where each row corresponds to a single d-dimensional example
# y is a n x 1 vector representing the tumor labels of the examples in X
# max_iters is the maximum number of gradient descent iterations
# step_size is the step size for gradient descent
# w is the d x 1 weight vector at the end of training
# losses is a list of negative log likelihood values for each iteration
def trainLogistic(X, y, max_iters=max_iters, step_size=step_size):
    # Initialize weights vector with zeros
    w = np.zeros((X.shape[1], 1))
    
    # keep track of losses for plotting
    losses = np.array([calculateNegativeLogLikelihood(X,y,w)])
    
    # take up to max_iters steps of gradient descent
    for i in range(max_iters):
        # compute the gradient over the dataset and store in w_grad
        w_grad = X.T@(logistic(X@w) - y)
        
        # This make sure the gradient is the right shape
        #assert(w_grad.shape == (X.shape[1],1))
        
        # take the update step in gradient descent
        w = w - step_size*w_grad
        
        # calculate the negative log likelihood with the new weight vector
        # and store it for plotting later
        losses = np.append(losses, calculateNegativeLogLikelihood(X,y,w))
    return w, losses

# given an input data matrix X, add a column of ones to the left-hand side
# X is a n x d matrix of examples where each row corresponds to a single d-dimensional example
# aug_X is a n x (d+1) matrix of examples where each row corresponds to a single d-dimensional example 
# where the first column is all ones
def dummyAugment(X):
    aug_X = np.hstack((np.ones((len(X),1)),X))
    return aug_X

# given a matrix X with size n x d, and y vector with size n x 1, perform k fold cross validation
def kFoldCrossVal(X, y, k):
    fold_size = int(np.ceil(len(X)/k))
    rand_inds = np.random.permutation(len(X))
    X = X[rand_inds]
    y = y[rand_inds]
    accuracy = np.array([])
    inds = np.arange(len(X))
    for i in range(k):
        start = min(len(X), fold_size*i)
        end = min(len(X), fold_size*(i+1))
        test_idx = np.arange(start, end)
        train_idx = np.concatenate([np.arange(0,start), np.arange(end, len(X))])
        if len(test_idx) < 2:
            break
        
        X_fold_test = X[test_idx]
        y_fold_test = y[test_idx]
        
        X_fold_train = X[train_idx]
        y_fold_train = y[train_idx]
        
        w, losses = trainLogistic(X_fold_train, y_fold_train)
        accuracy = np.append(accuracy,np.mean((X_fold_test@w >=0) == y_fold_test))
    return np.mean(accuracy), np.std(accuracy)

# load data from csv file
def loadData():
    train = np.genfromtxt('train_cancer.csv', delimiter=',')
    test = np.genfromtxt('test_cancer_pub.csv', delimiter=',')
    # store eight attributes data to x_train
    x_train = train[:,:-1]
    # store the tumor label in y_train
    y_train = train[:,-1]
    # transform to n x 1 vector
    y_train = y_train[:, np.newaxis]
    # store eight attributes data to x_test
    x_test = test
    return x_train, y_train, x_test


def main():
    # load the data
    logging.info("Loading data")
    x_train, y_train, x_test = loadData()
    logging.info("\n-----------------------------------------\n")
    
    # fit a logistic regression model on train and plot its losses
    logging.info("Training logistic regression model (No Bias Term)")
    w, losses = trainLogistic(x_train, y_train)
    y_pred_train = x_train@w >= 0
    
    logging.info(f"Learning weight vector: {([np.round(a,4)[0] for a in w])}")
    logging.info(f"Train accuracy: {np.mean(y_pred_train == y_train)*100:.4f}")
    logging.info("\n-----------------------------------------\n")
    
    x_train_bias = dummyAugment(x_train)
    # fit a logistic regression model on train and plot its losses
    logging.info("Training logistic regression model (Add Bias Term)")
    w, bias_losses = trainLogistic(x_train_bias, y_train)
    y_pred_train = x_train_bias@w >=0
    
    logging.info(f"Learning weight vector: {([np.round(a,4)[0] for a in w])}")
    logging.info(f"Train accuracy: {np.mean(y_pred_train == y_train)*100:.4f}")
    
    plt.figure(figsize=(16,9))
    plt.plot(range(len(losses)), losses, label="No Bias Term Added")
    plt.plot(range(len(bias_losses)), bias_losses, label="Bias Term Added")
    plt.title("Logistic Regression Training Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Negative Log Likelihood")
    plt.legend()
    plt.show()
    
    logging.info("\n-----------------------------------------\n")
    logging.info("Running cross-fold validation for bias case:")
    
    # perform k-fold cross validation
    for k in [2,3,4,5,10,20,50]:
        cv_acc, cv_std = kFoldCrossVal(x_train_bias, y_train, k)
        logging.info(f"{k}-fold Cross Validation Accuracy -- Mean (stdev): {cv_acc*100:.4} ({cv_std*100:.4})")
        
    x_test_bias = dummyAugment(x_test)
    y_pred_test = x_test_bias@w >= 0
    y_pred_test.astype(int)
    test_out = np.concatenate((np.expand_dims(np.array(range(len(y_pred_test)),dtype=np.int64), axis=1), y_pred_test), axis=1)
    header = np.array([["id", "type"]])
    test_out = np.concatenate((header, test_out))
    np.savetxt('test_predicted.csv', test_out, fmt='%s', delimiter=',')


if __name__ == "__main__":
    main()