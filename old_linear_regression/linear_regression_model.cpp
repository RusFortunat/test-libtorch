#include "linear_regression.h"
#include <torch/torch.h>
#include <random>
#include <math.h>
#include <algorithm>
namespace F = torch::nn::functional;
using namespace std;

random_device rd{};
mt19937 RNG{ rd() };

LinearRegImpl::LinearRegImpl(int input_dims, int hidden_size, int n_actions)
    : pi1(input_dims, hidden_size), pi(hidden_size, n_actions),
        v1(input_dims, hidden_size), v(hidden_size, 1){
    register_module("pi1", pi1);
    register_module("pi", pi);
    register_module("v1", v1);
    register_module("v", v);

}

torch::Tensor LinearRegImpl::forwardActor(torch::Tensor state) {
    auto output_actor = F::relu(pi1->forward(state));
    return pi->forward(output_actor);
}

torch::Tensor LinearRegImpl::forwardCritic(torch::Tensor state) {
    auto output_critic = F::relu(v1->forward(state));
    return v->forward(output_critic);
}

torch::Tensor LinearRegImpl::compute_loss(torch::Tensor state, torch::Tensor returns, int output_size){
    auto Pi = forwardActor(state);
    auto V = forwardCritic(state);
    uniform_int_distribution<int> dice(0, output_size-1); // a dice from 0 to 1
    auto probs = F::softmax(Pi, F::SoftmaxFuncOptions(1)); 
    double k = dice(RNG);
    auto actor_loss = log(probs[0][k]);
    cout << "actor loss" << endl << actor_loss << endl;

    return actor_loss;
}