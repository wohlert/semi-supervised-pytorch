import torch


def generate_label(batch_size, label, nlabels=2):
    """
    Generates a `torch.Tensor` of size batch_size x n_labels of
    the given label.

    Example: generate_label(2, 1, 3) #=> torch.Tensor([[0, 1, 0],
                                                       [0, 1, 0]])
    :param batch_size: number of labels
    :param label: label to generate
    :param nlabels: number of total labels
    """
    labels = (torch.ones(batch_size, 1) * label).type(torch.LongTensor)
    y = torch.zeros((batch_size, nlabels))
    y.scatter_(1, labels, 1)
    return y.type(torch.LongTensor)


def onehot(k):
    """
    Converts a number to its one-hot or 1-of-k representation
    vector.
    :param k: (int) length of vector
    :return: onehot function
    """
    def hot_vector(label):
        y = torch.LongTensor(k)
        y.zero_()
        y[label] = 1
        return y
    return hot_vector