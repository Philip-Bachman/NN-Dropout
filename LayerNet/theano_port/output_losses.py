import numpy as np
import theano
import theano.tensor as T

class LogisticRegression(object):
    """Multi-class Logistic Regression loss dangler."""

    def __init__(self, linear_layer):
        """Dangle a logistic regression from the given linear layer.

        The given linear layer should be a HiddenLayer (or subclass) object,
        for HiddenLayer as defined in LayerNet.py."""
        self.input_layer = linear_layer

    def loss_func(self, y):
        """Return the multiclass logistic regression loss for y.

        The class labels in y are assumed to be in correspondence with the
        set of column indices for self.input_layer.linear_output.
        """
        p_y_given_x = T.nnet.softmax(self.input_layer.linear_output)
        loss = -T.mean(T.log(p_y_given_x)[T.arange(y.shape[0]),y])
        return loss

    def errors(self, y):
        """Compute the number of wrong predictions by self.input_layer.

        Predicted class labels are computed as the indices of the columns of
        self.input_layer.linear_output which are maximal. Wrong predictions are
        those for which max indices do not match their corresponding y values.
        """
        # Compute class memberships predicted by self.input_layer
        y_pred = T.argmax(self.input_layer.linear_output, axis=1)
        errs = 0
        # check if y has same dimension of y_pred
        if y.ndim != y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', y_pred.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            errs = T.sum(T.neq(y_pred, y))
        else:
            raise NotImplementedError()
        return errs


class MCL2Hinge(object):
    """Multi-class one-vs-all L2 hinge loss dangler."""

    def __init__(self, linear_layer):
        """Dangle a squred hinge lossfrom the given linear layer.

        The given linear layer should be a HiddenLayer (or subclass) object,
        for HiddenLayer as defined in LayerNet.py."""
        self.input_layer = linear_layer

    def loss_func(self, y):
        """Return the multiclass squared hinge loss for y.

        The class labels in y are assumed to be in correspondence with the
        set of column indices for self.input_layer.linear_output.
        """
        y_hat = self.input_layer.linear_output
        margin_pos = T.maximum(0.0, (1.0 - y_hat))
        margin_neg = T.maximum(0.0, (1.0 + y_hat))
        obs_idx = T.arange(y.shape[0])
        loss_pos = T.sum(margin_pos[obs_idx,y]**2.0)
        loss_neg = T.sum(margin_neg**2.0) - T.sum(margin_neg[obs_idx,y]**2.0)
        loss = (loss_pos + loss_neg) / y.shape[0]
        return loss

    def errors(self, y):
        """Compute the number of wrong predictions by self.input_layer.

        Predicted class labels are computed as the indices of the columns of
        self.input_layer.linear_output which are maximal. Wrong predictions are
        those for which max indices do not match their corresponding y values.
        """
        # Compute class memberships predicted by self.input_layer
        y_pred = T.argmax(self.input_layer.linear_output, axis=1)
        errs = 0
        # check if y has same dimension of y_pred
        if y.ndim != y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', y_pred.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            errs = T.sum(T.neq(y_pred, y))
        else:
            raise NotImplementedError()
        return errs

