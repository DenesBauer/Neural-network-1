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
};



class Network {
public:
	std::vector < std::vector<Neuron>> network;
	Network(std::vector<int> layers);
	float relu(float x);
	float sigmoid(float x);
	Network_state construct_network_state();
	Network_state execute(std::vector<float> input_layer);
	Network_state calculate_errors(std::vector<float> expected_output, Network_state network_state);
	Network_gradient calculate_gradients(Network_state network_state);
	void apply_gradient(float learning_rate, Network_gradient network_gradient);
	void randomize_network();
};