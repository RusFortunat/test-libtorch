// Copyright 2020-present pytorch-cpp Authors
#include <torch/torch.h>
#include <iostream>
#include <iomanip>
#include "linear_regression.h"

int main() {
    std::cout << "Linear Regression\n\n";

    // Device
    auto cuda_available = torch::cuda::is_available();
    torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
    std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

    // Hyper parameters
    int batch_size = 5;
    int input_size = batch_size;
    int hidden_size = 5;
    int output_size = batch_size;
    const size_t num_epochs = 10;
    const double learning_rate = 0.001;

    // Sample dataset
    auto x_train = torch::randint(0, 10, {1, batch_size},
                                  torch::TensorOptions(torch::kFloat).device(device));

    auto y_train = torch::randint(0, 10, {1, batch_size},
                                  torch::TensorOptions(torch::kFloat).device(device));

    // Linear regression model
    LinearReg model(input_size, hidden_size, output_size);
    model->to(device);

    // Optimizer
    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(learning_rate));

    // Set floating point output precision
    std::cout << std::fixed << std::setprecision(4);

    std::cout << "Training...\n";

    // Train the model
    for (size_t epoch = 0; epoch != num_epochs; ++epoch) {
        // Forward pass
        //auto output = model->forward(x_train);
        std::cout << "model parameters " << endl<< model->parameters() << std::endl;
        auto loss = model->compute_loss(x_train, y_train, output_size);
        //std::cout << "loss " << endl << loss << endl;
        //auto loss = torch::nn::functional::mse_loss(output, y_train);

        // Backward pass and optimize
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        if ((epoch + 1) % 5 == 0) {
            std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs <<
                "], Loss: " << loss.item<double>() << "\n";
        }
    }

    std::cout << "Training finished!\n";
}