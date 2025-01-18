#include <iostream>
#include <vector>
#include <cmath>


/*
* Feed_forward_neural_network
*/

struct Neuron {
	float bias;
	std::vector<float> weights;
	float& operator[](int index) {
		return weights[index];
	}
	int size() {
		return weights.size();
	}

};

//Layer 0 is the the input layer
struct Network_state {
public:
	std::vector<std::vector<float>> network_state;
	std::vector<float>& operator[](int index) {
		return network_state[index];
	}
	int size() {
		return network_state.size();
	}
};
struct Network_gradient {
	std::vector<std::vector<float>> biases;
	std::vector<std::vector<std::vector<float>>> weights;

	//Supports adding extra layers
	// Useful for building the first gradient from an empty  gradient

	void operator+=(const Network_gradient& r) {
		for (int n_layer = 0; n_layer < r.weights.size(); n_layer++) {
			if (n_layer >= weights.size()) {
				biases.push_back(std::vector<float>(r.weights[n_layer].size()));
				weights.push_back(std::vector<std::vector<float>>());
			}
			for (int n_neuron = 0; n_neuron < r.weights[n_layer].size(); n_neuron++) {
				if (n_neuron >= weights[n_layer].size()) {
					weights[n_layer].push_back(std::vector<float>(r.weights[n_layer][n_neuron].size()));
				}
				biases[n_layer][n_neuron] += r.biases[n_layer][n_neuron];
				for (int n_connection = 0; n_connection < r.weights[n_layer][n_neuron].size();n_connection++) {
					weights[n_layer][n_neuron][n_connection] += r.weights[n_layer][n_neuron][n_connection];
				}
			}
		}
	}
};



class Network {
public:
	std::vector < std::vector<Neuron>> network;
	Network(std::vector<int> layers);
	float relu(float x);
	float relu_derivative(float x);
	float logistic(float x);
	float logistic_derivative(float x);
	float activate(float x);
	float activate_derivative(float x);
	Network_state construct_network_state();
	Network_gradient construct_network_gradient();
	Network_state execute(std::vector<float> input_layer);
	Network_state calculate_errors(std::vector<float> expected_output, Network_state network_state);
	Network_gradient calculate_gradients(Network_state eval_state, Network_state error_state);
	void apply_gradient(float learning_rate, Network_gradient network_gradient);
	void randomize_network();
	int get_output_max(Network_state network_state);
};