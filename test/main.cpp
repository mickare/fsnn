
#include <cstdlib>
#include <iostream>
#include <cmath>
#include "neural.h"

using namespace neuralnetwork;
using namespace matrix_math;


constexpr float softsign(const float v) {
	return v / (1 + std::abs(v));
}

constexpr float softsign_gradient(const float v) {
	const float a = (1 + std::abs(v));
	return  1 / (a * a);
}

const float distance_sqrd = 0.2 * 0.2;

float expected(const std::array<float, 2>& input) {
	const float dist = (input[0] * input[0]) + (input[1] * input[1]);
	return (dist < distance_sqrd) ? 0 : 1;
}

template<size_t N, size_t... Ns>
void print(network<N, Ns...>& net) {}

template<size_t M, size_t N, size_t... Ns>
void print(network<M, N, Ns...>& net) {

	std::cout << "Weights:\n" << net.layer.weights;
	std::cout << "Bias:\n" << net.layer.bias;

	std::cout << std::endl;

	if (sizeof...(Ns) > 0) {
		network<N, Ns...>& base = net;
		print(base);
	}

}

template<size_t M, size_t N, size_t... Ns>
void print(network<M, N, Ns...>& net, network<M, N, Ns...>& trainer) {
	std::cout << "Net:" << std::endl;
	print(net);
	std::cout << "Trainer:" << std::endl;
	print(trainer);
}

int main(int argc, char** argv) {

	std::cout << "Start" << std::endl;

	network<5, 5> net(&softsign);
	init_network(net);

	network<5, 5> trainer(&softsign_gradient);
	init_network_trainer(trainer);

	training_config config;
	config.rate_learning = 0.1;
	config.rate_inertia = 0.01;

	matrix<float, 5, 1> in;

	std::cout << "Start" << std::endl;
	print(net, trainer);


	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dis(0, 1);

	const auto random = [&](const float v) { return dis(gen); };


	for (size_t i = 0; i < 100000; ++i) {
		in.apply(random);

		train(config, trainer, net, in, in);
	}

	std::cout << "End" << std::endl;
	print(net, trainer);

	for (size_t i = 0; i < 2; ++i) {
		in.apply(random);

		std::cout << "Test:\n";
		std::cout << "IN:\n" << in;
		std::cout << "OUT\n" << forward(net, in);


	}

	return 0;
}