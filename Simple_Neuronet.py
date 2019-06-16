import numpy as np

# Inputs Values

Negative = 0.0
Positive1 = 1.0
Positive2 = 1.0


# Activation Function

def Heaviside_foo(x):
    if x >= 0.5:
        return 1
    else:
        return 0


# Prediction Function

def predict(Negative, Positive1, Positive2):
    inputs = np.array([Negative, Positive1, Positive2])

    weights_input_to_hidden_1 = [0.25, 0.25, 0]
    weights_input_to_hidden_2 = [0.5, -0.4, 0.9]

    weights_input_to_hidden = np.array([weights_input_to_hidden_1, weights_input_to_hidden_2])

    weights_hidden_to_output = np.array([-1, 1])

    hidden_input = np.dot(weights_input_to_hidden, inputs)
    print("Скрытый входной слой: " + str(hidden_input))

    hidden_output = np.array([Heaviside_foo(x) for x in hidden_input])
    print("Скрытый выходной слой: " + str(hidden_output))

    output = np.dot(weights_hidden_to_output, hidden_output)
    print("Выход: " + str(output))

    return Heaviside_foo(output) == 1


print("Результат: " + str(predict(Negative, Positive1, Positive2)))