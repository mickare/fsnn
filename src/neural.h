
#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <typeinfo>
#include <functional>
#include <random>
#include "helper_matrix.h"

using namespace matrix_math;

namespace neuralnetwork {

/*
 * neural network configuration and activation functions
 */

template <typename T>
struct network_config {
	typedef T type;
}

template <typename T>
constexpr T softsign_apply(const T v) {
	return v / (1 + std::abs(v));
}

template <typename T>
constexpr T softsign_gradient(const T v) {
	const T a = (1 + std::abs(v));
	return  1 / (a * a);
}

template <typename T>
struct activation_softsign :: network_config<T> { // Default
	static constexpr auto apply = softsign_apply<T>;
	static constexpr auto gradient = softsign_gradient<T>;
}


template <typename T>
constexpr T linear_apply(const T v) {
	return v;
}

template <typename T>
constexpr T linear_gradient(const T v) {
	return 1;
}

template <typename T>
struct activation_linear :: network_config<T> {
	static constexpr auto apply = linear_apply<T>;
	static constexpr auto gradient = linear_gradient<T>;
}

/*
 * The following is a variadic template neural network structure.
 * The synapse layers are recursive defined. That means a network<2,1,2> extends network<1,2>.
 */

template <typename T, size_t IN, size_t OUT>
struct layer {
	matrix<T, OUT, IN> weights;
	matrix<T, OUT, 1> bias;
};

template <template<typename> typename Config, typename T, size_t N, size_t ... Ns>
struct network {};
template <template<typename> typename Config = activation_softsign<float>, typename T, size_t M, size_t N, size_t ... Ns>
struct network<Config<T>, M, N, Ns...> : network<Config<T>, N, Ns...> {
	layer<T, M, N> layer; // tail
};

template <size_t, class>
struct elem_type_holder;

template <typename Config, size_t M, size_t N, size_t ... Ns>
struct elem_type_holder<0, network<Config, M, N, Ns...>> {
	typedef layer<T, M, N> layer;
	typedef matrix<T, M, 1> input;
	typedef matrix<T, N, 1> output;
};

template <size_t k, typename Config, typename T, size_t M, size_t N, size_t ... Ns>
struct elem_type_holder<k, network<T, M, N, Ns...>> {
	typedef typename elem_type_holder < k - 1, network<Config, T, N, Ns... >>::layer layer;
	typedef typename elem_type_holder < k - 1, network<Config, T, N, Ns... >>::input input;
	typedef typename elem_type_holder < k - 1, network<Config, T, N, Ns... >>::output output;
};

// Const getter missing

// Non-const getter

template <size_t k, typename Config, typename T, size_t M, size_t N, size_t... Ns>
typename std::enable_if <k == 0,
         typename elem_type_holder<0, network<Config, T, M, N, Ns...>>::layer& >::type
get_layer(network<Config, T, M, N, Ns...>& n) {
	return n.layer;
}

template <size_t k, typename Config, typename T, size_t M, size_t N, size_t ... Ns>
typename std::enable_if < k != 0,
         typename elem_type_holder<k, network<Config, T, M, N, Ns...>>::layer& >::type
get_layer(network<Config, T, M, N, Ns...>& n) {
	network<Config, T, N, Ns...>& base = n;
	return get_layer < k - 1 > (base);
}


// variadic template stuff end

template <typename Ramdom, typename T, size_t IN, size_t OUT>
void randomize(layer<T, IN, OUT>& layer, const Ramdom& random) {
	layer.weights.apply(random);
	layer.bias.apply(random);
}

template <typename Ramdom, typename Config, typename T, size_t N, size_t ... Ns>
void init_network_internal(network<Config, T, N, Ns...>& net, const Ramdom& random) {}

template <typename Ramdom, typename Config, typename T, size_t M, size_t N, size_t ... Ns>
void init_network_internal(network<Config, T, M, N, Ns...>& net, const Ramdom& random) {
	randomize(net.layer, random);
	if (sizeof...(Ns) > 0) {
		network<Config, T, N, Ns...>& base = net;
		init_network_internal(base, random);
	}
}

template <typename Ramdom, typename Config, typename T, size_t M, size_t N, size_t ... Ns>
void init_network(network<Config, T, M, N, Ns...>& net, const Ramdom& random) {
	init_network_internal(net, random);
}

template <typename Config, typename T, size_t M, size_t N, size_t ... Ns>
void init_network(network<Config, T, M, N, Ns...>& net) {

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dis(0, 1);

	const auto random = [&]() { return static_cast<T>(dis(gen)); };

	init_network(net, random);
}



template <size_t k, typename Config, typename T, size_t N, size_t ... Ns>
typename std::enable_if <k == 0,
         matrix<T, N, 1>>::type
forward_internal(const network< Config, T, N, Ns...>& net, const matrix<T, N, 1>& input) {
	return input;
}

template <size_t k, typename Config, typename T, size_t M, size_t N, size_t ... Ns>
typename std::enable_if < k != 0,
         typename elem_type_holder<sizeof...(Ns), network< Config, T, M, N, Ns...>>::output >::type
forward_internal(const network< Config, T, M, N, Ns...>& net, const matrix<T, M, 1>& input) {

	matrix<T, N, 1> out = net.layer.weights * input;
	for (size_t i = 0; i < N; ++i) {
		out[i][0] += net.layer.bias[i][0];
		out[i][0] = net.activation(out[i][0]);
	}

	const network<Config, T, N, Ns...>& base = net;
	return forward_internal < k - 1 > (base, out);
}

template <typename Config, typename T, size_t M, size_t N, size_t ... Ns>
typename elem_type_holder<sizeof...(Ns), network< Config, T, M, N, Ns...>>::output
forward(const network< Config, T, M, N, Ns...>& net, const matrix<T, M, 1>& input) {
	return forward_internal < sizeof...(Ns) + 1 > (net, input);
}

template <typename T>
struct training_config {
	T rate_learning = 0.03;
	T rate_inertia = 0.01;
};


template <typename Config, typename T, size_t N, size_t ... Ns>
void init_network_trainer(network<Config, T, N, Ns...>& trainer) {}

template <typename Config, typename T, size_t M, size_t N, size_t ... Ns>
void init_network_trainer(network<Config, T, M, N, Ns...>& trainer) {
	trainer.layer.weights.fill(0);
	trainer.layer.bias.fill(0);
	if (sizeof...(Ns) > 0) {
		network<Config, T, N, Ns...>& base = trainer;
		init_network_trainer(base);
	}
}



/**
 * Training for output layer.
 * @Returns error
 */
template <typename T, size_t M, size_t ... Ns>
typename std::enable_if < sizeof...(Ns) == 0, matrix<T, M, 1> >::type
train_internal(
    const training_config<T>& config,				// config for training (e.g. rate)
    network<T, M, Ns...>& trainer,			// trainer temporary structure for delta weights
    network<T, M, Ns...>& net,					// network to be trained
    const matrix<T, M, 1>& layer_input,	// input of layer M
    const matrix<T, M, 1>& layer_output,	// output of layer M
    const matrix<T, M, 1>& expected
) {

	matrix<T, M, 1> error_next;

	// output layer
	for (size_t m = 0; m < M; ++m) {
		error_next[m][0] = trainer.activation(layer_input[m][0]) * (expected[m][0] - layer_output[m][0]);
	}

	return error_next;

}

/**
 * Training for hidden layer.
 * @Returns error
 */
template <typename T, size_t M, size_t N, size_t ... Ns>
matrix<T, M, 1 >
train_internal(
    const training_config<T>& config,				// config for training (e.g. rate)
    network<T, M, N, Ns...>& trainer,			// trainer temporary structure for delta weights
    network<T, M, N, Ns...>& net,					// network to be trained
    const matrix<T, M, 1>& layer_input,	// input of layer M
    const matrix<T, M, 1>& layer_output,	// output of layer M
    const typename elem_type_holder<sizeof...(Ns), network<T, M, N, Ns...>>::output& expected
) {



	/* *******************************
	 * Calculate next layer input and output
	 * *******************************/

	//layer<M,N>  ==> weights: matrix<N,M>
	matrix<T, N, 1> layer_next_input = net.layer.weights * layer_output;
	for (size_t i = 0; i < N; ++i) {
		layer_next_input[i][0] += net.layer.bias[i][0];
	}
	matrix<T, N, 1> layer_next_output;
	for (size_t i = 0; i < N; ++i) {
		layer_next_output[i][0] = net.activation(layer_next_input[i][0]);
	}

	network<T, N, Ns...>& base = net;
	network<T, N, Ns...>& base_trainer = trainer;
	const matrix<T, N, 1> error_next = train_internal(config, base_trainer, base, layer_next_input, layer_next_output, expected);

	/* *******************************
	 * Calculate weight deltas...
	 * *******************************/

	// w(new) = delta_w + (rate_b * w(old))
	// with: delta_w = rate_a * err_next * output

	trainer.layer.weights *= config.rate_inertia; // new delta weights after calculation
	for (size_t n = 0; n < N; ++n) {
		for (size_t m = 0; m < M; ++m) {
			trainer.layer.weights[n][m] += config.rate_learning * error_next[n][0] * layer_output[m][0];
		}
	}

	// Don't forget the bias...
	trainer.layer.bias *= config.rate_inertia;
	for (size_t n = 0; n < N; ++n) {
		trainer.layer.bias[n][0] += config.rate_learning * error_next[n][0] * 1;
	}

	/* *******************************
	 * Calculate first layer error and return it...
	 * *******************************/

	matrix<T, M, 1> error;
	for (size_t m = 0; m < M; ++m) {
		T temp = 0;
		for (size_t n = 0; n < N; ++n) {
			temp += error_next[n][0] * net.layer.weights[n][m];
		}
		error[m][0] = trainer.activation(layer_input[m][0]) * temp; // activation in `trainer` is the gradient of the activation in the network `net`
	}


	/* *******************************
	 * Apply weight deltas...
	 * *******************************/
	net.layer.weights += trainer.layer.weights;
	net.layer.bias += trainer.layer.bias;

	return error;
}

template <typename T, size_t M, size_t N, size_t ... Ns>
matrix<T, M, 1> train(
    const training_config<T>& config,				 // config for training (e.g. rate)
    network<T, M, N, Ns...>& trainer,			 // trainer temporary structure for delta weights
    network<T, M, N, Ns...>& net,				 // network to be trained
    const matrix<T, M, 1>& layer_input,	 // input of layer M
    const typename elem_type_holder<sizeof...(Ns), network<T, M, N, Ns...>>::output& expected
) {
	// k is `sizeof...(Ns) + 1` because there are n*synapse layer but n+1 layers
	return train_internal(config, trainer, net, layer_input, layer_input, expected);

}

}

#endif /* NEURALNETWORK_H */