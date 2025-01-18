#include "Network.hpp"

float map(float min1, float max1, float min2, float max2, float x) {
	return (x - min1) / (max1 - min1) * (max2 - min2) + min2;
}

float clamp(float min, float max, float x) {
	return std::max(min, std::min(max, x));
}

//The first layer is only created for interoperability with calulations with data, representing the input layer
Network::Network(std::vector<int> layers) {
	//Symbolical input layer
	network.push_back(std::vector<Neuron>());
	for (int i = 1; i < layers.size(); i++) {
		network.push_back(std::vector<Neuron>());
		for (int j = 0; j < layers[i];j++) {
			network[i].push_back(Neuron{ 0,std::vector<float>() });
			for (int k = 0; k < layers[i - 1]; k++) {
				network[i][j].weights.push_back(0);
			}
		}
	}
}

float Network::relu(float x) {
	return std::max(x, 0.1f * x);
}

float Network::relu_derivative(float x) {
	if (x >= 0) {
		return 1;
	}
	else {
		return 0.1;
	}
}

float Network::logistic(float x) {
	return 1 / (1 + std::exp(-x));
}

float Network::logistic_derivative(float x) {
	return logistic(x) * (1  - logistic(x));
}

float Network::activate(float x) {
	return logistic(x);
}

float Network::activate_derivative(float x) {
	return logistic_derivative(x);
}



Network_state Network::construct_network_state() {
	Network_state network_state{ std::vector<std::vector<float>>() };
	network_state.network_state.push_back(std::vector<float>());
	for (int i = 0; i < network[1][0].weights.size();i++) {
		network_state[0].push_back(NULL);
	}
	for (int i = 1; i < network.size(); i++) {
		network_state.network_state.push_back(std::vector<float>());
		for (int j = 0; j < network[i].size();j++) {
			network_state[i].push_back(NULL);
		}
	}
	//std::cout << network_state[1].size();
	return network_state;
}

Network_gradient Network::construct_network_gradient() {
	Network_gradient gradient{
		std::vector<std::vector<float>>(network.size(),std::vector<float>()),
		std::vector<std::vector<std::vector<float>>>(network.size(),std::vector<std::vector<float>>())
	};
	for (int n_layer = 0; n_layer < network.size(); n_layer++) {
		gradient.biases[n_layer] = std::vector<float>(network[n_layer].size(), 0);
		gradient.weights[n_layer] = std::vector<std::vector<float>>(network[n_layer].size(), std::vector<float>());

		for (int n_neuron = 0; n_neuron < network[n_layer].size(); n_neuron++) {
			gradient.weights[n_layer][n_neuron] = std::vector<float>(network[n_layer][n_neuron].size());
		}
	}
	return gradient;
}

Network_state Network::execute(std::vector<float> input_layer) {
	Network_state eval = construct_network_state();
	eval[0] = input_layer;
	for (int n_layer = 1; n_layer < network.size(); n_layer++) {
		for (int n_neuron = 0; n_neuron < network[n_layer].size(); n_neuron++) {
			float sum = network[n_layer][n_neuron].bias;
			for (int n_connection = 0; n_connection < network[n_layer][n_neuron].weights.size(); n_connection++) {
				sum += network[n_layer][n_neuron][n_connection] * activate(eval[n_layer - 1][n_connection]);
			}
			eval[n_layer][n_neuron] = sum;
		}
	}
	return eval;
}

Network_state Network::calculate_errors(std::vector<float> expected_output, Network_state eval) {
	Network_state error = construct_network_state();
	for (int n_neuron = 0; n_neuron < network[network.size() - 1].size();n_neuron++) {
		error[network.size() - 1][n_neuron] = eval[eval.size() - 1][n_neuron] - expected_output[n_neuron];
	}
	for (int n_layer = network.size() - 2; n_layer >= 0; n_layer--) {
		for (int n_neuron = 0; n_neuron < network[n_layer].size();n_neuron++) {
			for (int n_forward_connection = 0; n_forward_connection < network[n_layer + 1].size();n_forward_connection++) {
				error[n_layer][n_neuron] += activate_derivative(eval[n_layer][n_neuron]) *
					error[n_layer + 1][n_forward_connection] * network[n_layer + 1][n_forward_connection][n_neuron];
			}

		}
	}
	return error;
}


Network_gradient Network::calculate_gradients(Network_state eval, Network_state error) {
	Network_gradient gradient = construct_network_gradient();
	for (int n_layer = 1; n_layer < network.size(); n_layer++) {
		for (int n_neuron = 0; n_neuron < network[n_layer].size();n_neuron++) {
			gradient.biases[n_layer][n_neuron] = error[n_layer][n_neuron];
			for (int n_connection = 0; n_connection < network[n_layer][n_neuron].size();n_connection++) {
				gradient.weights[n_layer][n_neuron][n_connection] =
					eval[n_layer - 1][n_connection] * error[n_layer][n_neuron];
			}
		}
	}
	return gradient;
}

void Network::apply_gradient(float learning_rate, Network_gradient network_gradient) {
	for (int n_layer = 1; n_layer < network_gradient.weights.size(); n_layer++) {
		for (int n_neuron = 0; n_neuron < network_gradient.weights[n_layer].size();n_neuron++) {
			network[n_layer][n_neuron].bias += learning_rate * network_gradient.biases[n_layer][n_neuron];
			for (int n_connection = 0; n_connection < network_gradient.weights[n_layer][n_neuron].size(); n_connection++) {
				network[n_layer][n_neuron][n_connection] -= learning_rate * network_gradient.weights[n_layer][n_neuron][n_connection];
			}
		}
	}
}

void Network::randomize_network() {
	for (int n_layer = 1; n_layer < network.size(); n_layer++) {
		for (int n_neuron = 0; n_neuron < network[n_layer].size(); n_neuron++) {
			network[n_layer][n_neuron].bias = map(0, 200, -1, 1, rand() % 200);
			for (int n_connection = 0; n_connection < network[n_layer][n_neuron].weights.size(); n_connection++) {
				network[n_layer][n_neuron][n_connection] = map(0, 200, -1, 1, rand() % 200);
			}
		}
	}
}

int Network::get_output_max(Network_state network_state) {
	float max = -INFINITY;
	int max_id = 0;
	for (int i = 0;i < network_state[network_state.size() - 1].size();i++) {
		float v = network_state[network_state.size() - 1][i];
		if (v > max) {
			max_id = i;
			max = v;
		}
	}
	return max_id;
}
