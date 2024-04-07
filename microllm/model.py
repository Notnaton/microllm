import tinygrad

class Model(tinygrad.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = tinygrad.nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x)