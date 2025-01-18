#include <iostream>
#include <cmath>
#include <fstream>

#include <raylib.h>

#include "Network.hpp"

int main() {
	srand(time(0));
	std::vector<int> layers = { 2,5, 2 };
	std::vector<float> expected_output = { 0.3, 0.7 };
	std::vector<float> expected_output2 = { 1.0, 0.4 };
	Network network(layers);
	network.randomize_network();

	for (int iteration = 0; iteration < 400; iteration++) {
		Network_state eval;
		Network_state error;
		Network_gradient batch_gradient = network.construct_network_gradient();
		std::cout << std::endl << "Iteration " << iteration << std::endl;
		eval = network.execute({ 0.6,1 });
		std::cout << "1 - Output neuron 0 - 0.3: " << eval[layers.size() - 1][0] << std::endl;
		std::cout << "1 - Output neuron 1 - 0.7: " << eval[layers.size() - 1][1] << std::endl;
		error = network.calculate_errors(expected_output, eval);
		batch_gradient += network.calculate_gradients(eval, error);
		
		eval = network.execute({ 0,0 });
		std::cout << "2 - Output neuron 0 - 1.0: " << eval[layers.size() - 1][0] << std::endl;
		std::cout << "2 - Output neuron 1 - 0.4: " << eval[layers.size() - 1][1] << std::endl;
		error = network.calculate_errors(expected_output2, eval);
		batch_gradient += network.calculate_gradients(eval, error);
		
		network.apply_gradient(0.01, batch_gradient);
	}
	//result = network.execute({ 0.2,1.3 });


}