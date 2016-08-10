
#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <typeinfo>
#include <functional>
#include <random>
#include "helper_matrix.h"

#define NEURALNETWORK_CHECKS

using namespace matrix_math;

namespace neuralnetwork {

/*
 * The following is a variadic template neural network structure.
 * The synapse layers are recursive defined. That means a network<2,1,2> extends network<1,2>.
 */

template <size_t IN, size_t OUT>
struct layer {
	matrix<float, OUT, IN> weights;
	matrix<float, OUT, 1> bias;
};

template <size_t N, size_t ... Ns> struct network {
	network(std::function<float(float)> _activation) : activation(_activation) {}
	std::function<float(float)> activation;
};
template <size_t M, size_t N, size_t ... Ns>
struct network<M, N, Ns...> : network<N, Ns...>
{
	network(std::function<float(float)> _activation) : network<N, Ns...>(_activation) {}

	layer<M, N> layer; // tail
};

template <size_t, class> struct elem_type_holder;

template <size_t M, size_t N, size_t ... Ns>
struct elem_type_holder<0, network<M, N, Ns...>> {
	typedef layer<M, N> layer;
	typedef matrix<float, M, 1> input;
	typedef matrix<float, N, 1> output;
};

template <size_t k, size_t M, size_t N, size_t ... Ns>
struct elem_type_holder<k, network<M, N, Ns...>> {
	typedef typename elem_type_holder < k - 1, network<N, Ns... >>::layer layer;
	typedef typename elem_type_holder < k - 1, network<N, Ns... >>::input input;
	typedef typename elem_type_holder < k - 1, network<N, Ns... >>::output output;
};

// Const

/*

template <size_t k, size_t N, size_t... Ns>
typename std::enable_if <k == 0,
         typename elem_type_holder<0, network<N, Ns...>>::layer const& >::type
get_layer(const network<N, Ns...>& n) {
	return n.layer;
}

template <size_t k, size_t M, size_t N, size_t ... Ns>
typename std::enable_if < k != 0,
         typename elem_type_holder<k, network<M, N, Ns...>>::layer const& >::type
get_layer(const network<M, N, Ns...>& n) {
	const network<N, Ns...>& base = n;
	return get_layer < k - 1 > (base);
}
*/

// Non-const

template <size_t k, size_t M, size_t N, size_t... Ns>
typename std::enable_if <k == 0,
         typename elem_type_holder<0, network<M, N, Ns...>>::layer& >::type
get_layer(network<M, N, Ns...>& n) {
	return n.layer;
}

template <size_t k, size_t M, size_t N, size_t ... Ns>
typename std::enable_if < k != 0,
         typename elem_type_holder<k, network<M, N, Ns...>>::layer& >::type
get_layer(network<M, N, Ns...>& n) {
	network<N, Ns...>& base = n;
	return get_layer < k - 1 > (base);
}


// variadic template stuff end

template <typename F, size_t IN, size_t OUT>
void randomize(layer<IN, OUT>& layer, const F& func) {
	layer.weights.apply(func);
	layer.bias.apply(func);
}

template <typename F, size_t N, size_t ... Ns>
void init_network_internal(network<N, Ns...>& net, const F& func) {}

template <typename F, size_t M, size_t N, size_t ... Ns>
void init_network_internal(network<M, N, Ns...>& net, const F& func) {
	randomize(net.layer, func);
	if (sizeof...(Ns) > 0) {
		network<N, Ns...>& base = net;
		init_network_internal(base, func);
	}
}

template <size_t M, size_t N, size_t ... Ns>
void init_network(network<M, N, Ns...>& net) {

	std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1);

	const auto func = [&](const auto value) { return dis(gen); };
    init_network_internal(net, func);

}



template <size_t k, size_t N, size_t ... Ns>
typename std::enable_if <k == 0,
         matrix<float, N, 1>>::type
forward_internal(const network<N, Ns...>& net, const matrix<float, N, 1>& input) {
	return input;
}

template <size_t k, size_t M, size_t N, size_t ... Ns>
typename std::enable_if < k != 0,
         typename elem_type_holder<sizeof...(Ns), network<M, N, Ns...>>::output >::type
forward_internal(const network<M, N, Ns...>& net, const matrix<float, M, 1>& input) {

	matrix<float, N, 1> out = net.layer.weights * input;
	for (size_t i = 0; i < N; ++i) {
		out[i][0] += net.layer.bias[i][0];
		out[i][0] = net.activation(out[i][0]);
	}

	const network<N, Ns...>& base = net;
	return forward_internal < k - 1 > (base, out);
}

template <size_t M, size_t N, size_t ... Ns>
typename elem_type_holder<sizeof...(Ns), network<M, N, Ns...>>::output
forward(const network<M, N, Ns...>& net, const matrix<float, M, 1>& input) {
	return forward_internal < sizeof...(Ns) + 1 > (net, input);
}

struct training_config {
	float rate_learning = 0.03;
	float rate_inertia = 0.01;
};


template <size_t N, size_t ... Ns>
void init_network_trainer(network<N, Ns...>& trainer) {}

template <size_t M, size_t N, size_t ... Ns>
void init_network_trainer(network<M, N, Ns...>& trainer) {
	trainer.layer.weights.fill(0);
	trainer.layer.bias.fill(0);
	if (sizeof...(Ns) > 0) {
		network<N, Ns...>& base = trainer;
		init_network_trainer(base);
	}
}



/**
 * Training for output layer.
 * @Returns error
 */
template <size_t M, size_t ... Ns>
typename std::enable_if < sizeof...(Ns) == 0, matrix<float, M, 1> >::type
train_internal(
    const training_config& config,				// config for training (e.g. rate)
    network<M, Ns...>& trainer,			// trainer temporary structure for delta weights
    network<M, Ns...>& net,					// network to be trained
    const matrix<float, M, 1>& layer_input,	// input of layer M
    const matrix<float, M, 1>& layer_output,	// output of layer M
    const matrix<float, M, 1>& expected
) {

	matrix<float, M, 1> error_next;

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
template <size_t M, size_t N, size_t ... Ns>
matrix<float, M, 1 >
train_internal(
    const training_config& config,				// config for training (e.g. rate)
    network<M, N, Ns...>& trainer,			// trainer temporary structure for delta weights
    network<M, N, Ns...>& net,					// network to be trained
    const matrix<float, M, 1>& layer_input,	// input of layer M
    const matrix<float, M, 1>& layer_output,	// output of layer M
    const typename elem_type_holder<sizeof...(Ns), network<M, N, Ns...>>::output& expected
) {



	/* *******************************
	 * Calculate next layer input and output
	 * *******************************/

	//layer<M,N>  ==> weights: matrix<N,M>
	matrix<float, N, 1> layer_next_input = net.layer.weights * layer_output;
	for (size_t i = 0; i < N; ++i) {
		layer_next_input[i][0] += net.layer.bias[i][0];
	}
	matrix<float, N, 1> layer_next_output;
	for (size_t i = 0; i < N; ++i) {
		layer_next_output[i][0] = net.activation(layer_next_input[i][0]);
	}

	network<N, Ns...>& base = net;
	network<N, Ns...>& base_trainer = trainer;
	const matrix<float, N, 1> error_next = train_internal(config, base_trainer, base, layer_next_input, layer_next_output, expected);

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

	matrix<float, M, 1> error;
	for (size_t m = 0; m < M; ++m) {
		float temp = 0;
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

template <size_t M, size_t N, size_t ... Ns>
matrix<float, M, 1> train(
    const training_config& config,				 // config for training (e.g. rate)
    network<M, N, Ns...>& trainer,			 // trainer temporary structure for delta weights
    network<M, N, Ns...>& net,				 // network to be trained
    const matrix<float, M, 1>& layer_input,	 // input of layer M
    const typename elem_type_holder<sizeof...(Ns), network<M, N, Ns...>>::output& expected
) {
	// k is `sizeof...(Ns) + 1` because there are n*synapse layer but n+1 layers
	return train_internal(config, trainer, net, layer_input, layer_input, expected);

}

}

#endif /* NEURALNETWORK_H */