import numpy as np


def softmax(logits, y):
    """
    Computes the loss and gradient for softmax classification.

    Args:
    - logits: A numpy array of shape (N, C)
    - y: A numpy array of shape (N,). y represents the labels corresponding to
    logits, where y[i] is the label of logits[i], and the value of y have a 
    range of 0 <= y[i] < C

    Returns (as a tuple):
    - loss: Loss scalar
    - dlogits: Loss gradient with respect to logits
    """
    loss, dlogits = None, None
    """
    TODO: Compute the softmax loss and its gradient using no explicit loops
    Store the loss in loss and the gradient in dW. If you are not careful
    here, it is easy to run into numeric instability. Don't forget the
    regularization!
    """
    ###########################################################################
    #                           BEGIN OF YOUR CODE                            #
    ###########################################################################
    reg = -np.max(logits)
    #normalization
    probs = np.exp(logits+reg) / np.sum(np.exp(logits+reg), axis=1, keepdims=True)
    correct_log_probs = -np.log(probs[range(probs.shape[0]),y])
    loss = np.sum(correct_log_probs)/ logits.shape[0]
    dlogits = probs
    dlogits[range(logits.shape[0]),y] -= 1
    dlogits /= logits.shape[0]
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
    return loss, dlogits
