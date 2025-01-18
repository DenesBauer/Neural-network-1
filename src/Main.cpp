#include <iostream>
#include <cmath>
#include <fstream>

#include <raylib.h>

#include "Network.hpp"

int main() {
	srand(time(0));
	std::vector<int> layers = { 2,10,20, 2 };
	std::vector<float> expected_output = { 0, 1 };
	std::vector<float> expected_output2 = { 1, 0 };
	Network network(layers);
	network.randomize_network(0.01);

	int a = 23;
	int b = 95;

	//a ^= b;
	//b ^= a;
	//a ^= b;

	a ^= b ^= a ^= b;

	std::cout << "Cross-entropy: " << cross_entropy(0.8, 0.3) << std::endl;
	std::vector<float> d = softmax({ 0.2,.5,.8,.6 });
	std::cout << "Softmax at id 0: " << softmax({ 0.2,.5,.8,.6 })[0] << std::endl;

	for (int iteration = 0; iteration < 400; iteration++) {
		Network_state eval;
		Network_state error;
		Network_gradient batch_gradient = network.construct_network_gradient();
		std::cout << std::endl << "Iteration " << iteration << std::endl;

		eval = network.execute({ 0.6,1 });
		std::cout << "1 - Output neuron 0 - 0.0: " << eval[layers.size() - 1][0] << std::endl;
		std::cout << "1 - Output neuron 1 - 1.0: " << eval[layers.size() - 1][1] << std::endl;
		error = network.calculate_errors(expected_output, eval);
		batch_gradient += network.calculate_gradients(eval, error);

		eval = network.execute({ 0,0 });
		std::cout << "2 - Output neuron 0 - 1.0: " << eval[layers.size() - 1][0] << std::endl;
		std::cout << "2 - Output neuron 1 - 0.0: " << eval[layers.size() - 1][1] << std::endl;
		error = network.calculate_errors(expected_output2, eval);
		batch_gradient += network.calculate_gradients(eval, error);

		bool sanity = network.check_network_state(eval);
		std::cout << "Network sanity: " << sanity << std::endl;


		network.apply_gradient(10, batch_gradient);
	}
	//result = network.execute({ 0.2,1.3 });


}