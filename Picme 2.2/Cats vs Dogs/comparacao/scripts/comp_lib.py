import torch, torch.nn as nn
def lado(N, p, convs):
    for _ in range(5):
        for _ in range(convs): N = N + 2*p - 4
        N = N // 2
    return N
def rede(H, W, padding, convs=1):
    b = []; c = 3
    for _ in range(5):
        for _ in range(convs):
            b += [nn.Conv2d(c, 32, 5, 1, padding), nn.ReLU()]; c = 32
        b += [nn.MaxPool2d(2, 2), nn.Dropout(.2)]
    h, w = lado(H, padding, convs), lado(W, padding, convs)
    return nn.Sequential(*b, nn.Flatten(), nn.Linear(32*h*w, 1024), nn.ReLU(), nn.Linear(1024, 32), nn.ReLU(), nn.Linear(32, 2)), (h, w)
# altura, largura, padding, convoluções por bloco
CFG = {"referencia": (94, 100, 1, 1), "padding2": (94, 100, 2, 1), "camadas10": (94, 100, 2, 2), "imagem224": (224, 224, 1, 1)}
