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

Network_state Network::construct_network_state() {
	Network_state network_state{ std::vector<std::vector<float>> ()};
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

Network_state Network::execute(std::vector<float> input_layer) {
	Network_state network_state = construct_network_state();
	network_state[0] = input_layer;
	for (int n_layer = 1; n_layer < network.size(); n_layer++) {
		for (int n_neuron = 0; n_neuron < network[n_layer].size(); n_neuron++) {
			for (int n_connection = 0; n_connection < network[n_layer][n_neuron].weights.size(); n_connection++) {
				network_state[n_layer][n_neuron] += relu(network[n_layer][n_neuron].bias
					+ network[n_layer][n_neuron][n_connection] * network_state[n_layer - 1][n_connection]);
			}
		}
	}
	return network_state;
}

Network_state Network::calculate_errors(std::vector<float> expected_output, Network_state network_state) {
	Network_state error_state = construct_network_state();
	for (int n_neuron = 0; n_neuron < network_state[network_state.size() - 1].size();n_neuron++) {
		error_state[network_state.size() - 1][n_neuron] = expected_output[n_neuron] - network_state[network_state.size() - 1][n_neuron];
	}
	for (int n_layer = network_state.size() - 2;n_layer >= 0;n_layer--) {
		for (int n_neuron = 0; n_neuron < network_state[n_layer].size();n_neuron++) {
			float error = 0;
			for (int n_forward_connection = 0; n_forward_connection < network_state[n_layer+1].size();n_forward_connection++) {
				//Mind that in 'network' layer 0 is the first layer after the input layer
				error += error_state[n_layer + 1][n_forward_connection] * network[n_layer + 1][n_forward_connection][n_neuron];
			}
		}
	}
	return error_state;
}


Network_gradient Network::calculate_gradients(Network_state error_state) {
	Network_gradient network_gradient{ std::vector<std::vector<float>>(), std::vector<std::vector<std::vector<float>>>() };
	network_gradient.biases.push_back(std::vector<float>());
	network_gradient.weights.push_back(std::vector<std::vector<float>>());
	for (int n_layer = 1; n_layer < error_state.size(); n_layer++) {
		network_gradient.biases.push_back(std::vector<float>());
		network_gradient.weights.push_back(std::vector<std::vector<float>>());
		for (int n_neuron = 0; n_neuron < error_state[n_layer].size();n_neuron++) {
			network_gradient.weights[n_layer].push_back(std::vector<float>());

			network_gradient.biases[n_layer].push_back(error_state[n_layer][n_neuron]);
			for (int n_connection = 0; n_connection < network[n_layer][n_neuron].weights.size();n_connection++) {
				network_gradient.weights[n_layer][n_neuron].push_back(network[n_layer][n_neuron][n_connection] * error_state[n_layer - 1][n_connection]);
			}
		}
	}
	return network_gradient;
}

void Network::apply_gradient(float learning_rate, Network_gradient network_gradient) {
	for (int n_layer = 1;n_layer < network_gradient.weights.size();n_layer++) {
		for (int n_neuron = 0; n_neuron < network_gradient.weights[n_layer].size();n_neuron++) {
			network[n_layer][n_neuron].bias += learning_rate * network_gradient.biases[n_layer][n_neuron];
			for (int n_connection = 0; n_connection < network_gradient.weights[n_layer][n_neuron].size(); n_connection++) {
				network[n_layer][n_neuron][n_connection] += learning_rate * network_gradient.weights[n_layer][n_neuron][n_connection];
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