#include <iostream>
#include <cmath>
#include <fstream>

#include <raylib.h>

#include "Network.hpp"

int main() {
	srand(time(0));
	std::vector<int> layers = { 2,10,8, 2 };
	std::vector<float> expected_output1 = { 0, 1 };
	std::vector<float> expected_output2 = { 1, 0 };
	Network network(layers);
	network.randomize_network(0.2);

	int a = 23;
	int b = 95;

	//a ^= b;
	//b ^= a;
	//a ^= b;

	a ^= b ^= a ^= b;

	std::cout << "Cross-entropy: " << network.cross_entropy_binary(0.8, 0.3) << std::endl;
	std::vector<float> d = network.softmax({ 0.2,.5,.8,.6 });
	std::cout << "Softmax at id 0: " << network.softmax({ 0.2,.5,.8,.6 })[0] << std::endl;


	std::vector<float> losses;
	float min_learning_rate = 0.000001;
	float steps_per_magnitude = 8;
	float max_learning_rate = 100;
	float learning_rate = min_learning_rate;

	int steps = steps_per_magnitude * (std::log10(max_learning_rate) - std::log10(min_learning_rate));

	int iterations = 100;
	for (int n_step = 0; n_step <= steps; n_step++) {
		learning_rate = std::pow(10, std::log10(min_learning_rate) +
			(std::log10(max_learning_rate) - std::log10(min_learning_rate)) * n_step / steps);
		std::cout << "Step: " << n_step << "/" << steps << std::endl;
		std::cout << "Learning rate: " << learning_rate << std::endl;
		float loss_total;
		for (int n_iteration = 0; n_iteration < iterations; n_iteration++) {
			int print_mod = 100;
			Network_state eval;
			Network_state error;
			Network_gradient batch_gradient = network.construct_network_gradient();
			if ((n_iteration + 1) % print_mod == 0) {
				std::cout << std::endl << "Iteration: " << n_iteration << std::endl;

			}

			eval = network.execute({ 0.6,1 });
			if ((n_iteration+1) % print_mod == 0) {

				std::cout << "1 - Output neuron 0 - 0.0: " << eval[layers.size() - 1][0] << std::endl;
				std::cout << "1 - Output neuron 1 - 1.0: " << eval[layers.size() - 1][1] << std::endl;
			}
			error = network.calculate_errors(expected_output1, eval);
			batch_gradient += network.calculate_gradients(eval, error);

			float loss1 = network.cross_entropy(expected_output1, eval[network.network.size() - 1]);


			eval = network.execute({ 0,0 });
			if ((n_iteration + 1) % print_mod == 0) {
				std::cout << "2 - Output neuron 0 - 1.0: " << eval[layers.size() - 1][0] << std::endl;
				std::cout << "2 - Output neuron 1 - 0.0: " << eval[layers.size() - 1][1] << std::endl;
			}
			error = network.calculate_errors(expected_output2, eval);
			batch_gradient += network.calculate_gradients(eval, error);

			bool sanity = network.check_network_state(eval);
			if ((n_iteration + 1) % print_mod == 0) {
				std::cout << "Network sanity: " << sanity << std::endl;
			}

			float loss2 = network.cross_entropy(expected_output2, eval[network.network.size() - 1]);
			loss_total = (loss1 + loss2) / 2;

			network.apply_gradient(learning_rate, batch_gradient);
		}
		std::cout << "Total loss: " << loss_total << std::endl << std::endl;
	}


	//result = network.execute({ 0.2,1.3 });


}