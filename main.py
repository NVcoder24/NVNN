import math
import activations_functions

def result_cost(got, expected):
    return sum([ (i[0] - i[1]) ** 2 for i in zip(got, expected)])

def solve_nn_fast(_input, layers, weights):
    activations = _input
    for i in range(len(weights)):
        new_activations = []
        for n in range(len(weights[i])):
            a = []
            for j in weights[i][n]:
                a.append(activations[j[0]] * j[1])
            a.append(layers[i][0][n])
            new_activations.append(layers[i][1](sum(a)))
        activations = new_activations
    return activations


print(
    solve_nn_fast(
        [0.8, 0.5], # Inputs
        [ # Layers
            [[1.5, 0.7, 0.1], activations_functions.Sigmoid], # Layer #1 (hidden, 3 neurons, ReLU)
            [[1], activations_functions.Sigmoid],             # Layer #2 (output, 1 neuron, Sigmoid)
        ],
        [ # Weights
            [ # Layer #1
                [ # Neuron #0
                    [0, 0.2], # Activation from input neuron #0, weight
                    [1, 0.3], # Activation from input neuron #1, weight
                ],
                [ # Neuron #1
                    [0, 0.3],  # Activation from input neuron #0, weight
                    [1, -1,2], # Activation from input neuron #1, weight
                ],
                [ # Neuron #2
                    [0, 1,5],  # Activation from input neuron #0, weight
                    [1, -0.7], # Activation from input neuron #1, weight
                ]
            ],
            [ # Layer #2
                [ # Neuron #1
                    [0, 0.6], # Activation from layer #1 neuron #0, weight
                    [1, 0.1], # Activation from layer #1 neuron #1, weight
                    [2, -2],  # Activation from layer #1 neuron #2, weight
                ]
            ]
        ]
    )
)