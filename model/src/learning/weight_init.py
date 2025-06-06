from torch import nn
from torch.nn import init


def weight_init(m, spread=1.0):
    """
    Initializes a model's parameters.
    Credits to: https://gist.github.com/jeasinema

    Usage:
        model = Model()
        model.apply(weight_init)
    """
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data, mean=0, std=spread)
        if m.bias is not None:
            init.normal_(m.bias.data, mean=0, std=spread)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data, gain=spread)
        if m.bias is not None:
            init.normal_(m.bias.data, mean=0, std=spread)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data, gain=spread)
        if m.bias is not None:
            init.normal_(m.bias.data, mean=0, std=spread)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data, mean=0, std=spread)
        if m.bias is not None:
            init.normal_(m.bias.data, mean=0, std=spread)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data, gain=spread)
        if m.bias is not None:
            init.normal_(m.bias.data, mean=0, std=spread)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data, gain=spread)
        if m.bias is not None:
            init.normal_(m.bias.data, mean=0, std=spread)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=0, std=spread)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=0, std=spread)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=0, std=spread)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data, gain=spread)
        try:
            init.normal_(m.bias.data, mean=0, std=spread)
        except AttributeError:
            pass
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data, mean=0, std=spread)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data, mean=0, std=spread)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data, mean=0, std=spread)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data, mean=0, std=spread)
