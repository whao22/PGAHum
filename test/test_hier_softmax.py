def hierarchical_softmax(x):
    def softmax(x):
        return torch.nn.functional.softmax(x, dim=-1)

    def sigmoid(x):
        return torch.sigmoid(x)

    n_batch, n_point, n_dim = x.shape
    x = x.flatten(0,1)

    prob_all = torch.ones(n_batch * n_point, 24, device=x.device)

    prob_all[:, [1, 2, 3]] = prob_all[:, [0]] * sigmoid(x[:, [0]]) * softmax(x[:, [1, 2, 3]])
    prob_all[:, [0]] = prob_all[:, [0]] * (1 - sigmoid(x[:, [0]]))

    prob_all[:, [4, 5, 6]] = prob_all[:, [1, 2, 3]] * (sigmoid(x[:, [4, 5, 6]]))
    prob_all[:, [1, 2, 3]] = prob_all[:, [1, 2, 3]] * (1 - sigmoid(x[:, [4, 5, 6]]))

    prob_all[:, [7, 8, 9]] = prob_all[:, [4, 5, 6]] * (sigmoid(x[:, [7, 8, 9]]))
    prob_all[:, [4, 5, 6]] = prob_all[:, [4, 5, 6]] * (1 - sigmoid(x[:, [7, 8, 9]]))

    prob_all[:, [10, 11]] = prob_all[:, [7, 8]] * (sigmoid(x[:, [10, 11]]))
    prob_all[:, [7, 8]] = prob_all[:, [7, 8]] * (1 - sigmoid(x[:, [10, 11]]))

    prob_all[:, [12, 13, 14]] = prob_all[:, [9]] * sigmoid(x[:, [24]]) * softmax(x[:, [12, 13, 14]])
    prob_all[:, [9]] = prob_all[:, [9]] * (1 - sigmoid(x[:, [24]]))

    prob_all[:, [15]] = prob_all[:, [12]] * (sigmoid(x[:, [15]]))
    prob_all[:, [12]] = prob_all[:, [12]] * (1 - sigmoid(x[:, [15]]))

    prob_all[:, [16, 17]] = prob_all[:, [13, 14]] * (sigmoid(x[:, [16, 17]]))
    prob_all[:, [13, 14]] = prob_all[:, [13, 14]] * (1 - sigmoid(x[:, [16, 17]]))

    prob_all[:, [18, 19]] = prob_all[:, [16, 17]] * (sigmoid(x[:, [18, 19]]))
    prob_all[:, [16, 17]] = prob_all[:, [16, 17]] * (1 - sigmoid(x[:, [18, 19]]))

    prob_all[:, [20, 21]] = prob_all[:, [18, 19]] * (sigmoid(x[:, [20, 21]]))
    prob_all[:, [18, 19]] = prob_all[:, [18, 19]] * (1 - sigmoid(x[:, [20, 21]]))

    prob_all[:, [22, 23]] = prob_all[:, [20, 21]] * (sigmoid(x[:, [22, 23]]))
    prob_all[:, [20, 21]] = prob_all[:, [20, 21]] * (1 - sigmoid(x[:, [22, 23]]))

    prob_all = prob_all.reshape(n_batch, n_point, prob_all.shape[-1])
    return prob_all