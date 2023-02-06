#pragma once

#include <torch/torch.h>
using namespace std;

class LinearRegImpl : public torch::nn::Module {
    public:
        LinearRegImpl(int input_dims, int hidden_size, int n_actions);
        
        torch::Tensor forwardActor(torch::Tensor state);
        torch::Tensor forwardCritic(torch::Tensor state);
        torch::Tensor compute_loss(torch::Tensor state, torch::Tensor returns, int output_size);

    private:
        torch::nn::Linear pi1;
        torch::nn::Linear v1;
        torch::nn::Linear pi;
        torch::nn::Linear v;

};

TORCH_MODULE(LinearReg);

