#include <iostream>
#include <cmath>
#include <fstream>

#include <raylib.h>

#include "Network.hpp"

int main() {
	srand(time(0));
	std::vector<int> layers = { 2, 4,5, 100,200,7, 2 };
	std::vector<float> expected_output = { 0.3, 1.16 };
	Network network(layers);
	network.randomize_network();
	Network_state result; 
	for (int iteration = 0; iteration < 10000; iteration++) {
		result = network.execute({ 2,3 });
		std::cout << std::endl << "Iteration " << iteration << std::endl;
		std::cout << "Output neuron 0: " << result[layers.size() - 1][0] << std::endl;
		std::cout << "Output neuron 1: " << result[layers.size() - 1][1] << std::endl;
		Network_state error = network.calculate_errors(expected_output, result);
		Network_gradient gradient = network.calculate_gradients(error);
		network.apply_gradient(0.3, gradient);
	}
	result = network.execute({ 0.2,1.3 });


}