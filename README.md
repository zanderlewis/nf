# NeuralFlow

NeuralFlow is a simple neural network library written in C. It includes basic functionality for creating, training, and using neural networks.

## Features

- Basic feedforward neural network implementation
- Training with backpropagation
- Simple data loading and prediction

## Usage

To use NeuralFlow in your project, follow these steps:

1. Include the header file in your project:
```c
#include "nf.h"
```

2. Create a neural network instance:
```c
NeuralNetwork *nn = create_nn(input_size, hidden_size, output_size);
```

3. Train the neural network:
```c
train_nn(nn, positive_samples, negative_samples, num_samples);
```

4. Make predictions:
```c
float prediction = predict(nn, input);
```

5. Save and load the neural network:
```c
save_nn(nn, "nn.bin");
NeuralNetwork *loaded_nn = load_nn("nn.bin");
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
