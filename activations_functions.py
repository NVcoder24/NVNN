import math

def Perception(x:float) -> float:
    return 1 if x > 0 else 0

def Sigmoid(x:float) -> float:
    return 1 / (1 + (1 / math.exp(x)))

def Tanh(x:float) -> float:
    return math.tanh(x)

def ReLU(x:float) -> float:
    return x if x > 0 else 0

def LeakyReLU(x:float) -> float:
    return 0.1 * x if x < 0 else x

def ELU(x:float) -> float:
    return math.exp(x) - 1 if x < 0 else x

def Softplus(x:float) -> float:
    return math.log(1 + math.exp(x))