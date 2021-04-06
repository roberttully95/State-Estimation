#include <iostream>

#include "LinearStateSpaceModel.h"

#include <Eigen/Core>

int main()
{
	// Define the initial state of the system
	const int n = 2;
	const int q = 1;
	const int p = 1;
	float states[n] = { 1, 2 };
	float input[p] = { 0.6f };
	float state_mtx[n * n] = { 0, 2, -1, -3 };
	float input_mtx[n * p] = { 5, 0 };
	float output_mtx[q * n] = { 1, 0 };
	float feedforward_gain[q * p] = { 0 };

	// Convert input to eigen datatypes
	Eigen::Matrix<float, n, 1> x(states);
	Eigen::Matrix<float, p, 1> u(input);
	Eigen::Matrix<float, n, n, Eigen::RowMajor> A(state_mtx);
	Eigen::Matrix<float, n, p> B(input_mtx);
	Eigen::Matrix<float, q, n> C(output_mtx);
	Eigen::Matrix<float, q, p> D(feedforward_gain);

	// Create state space model
	auto ss = LinearStateSpaceModel<float, n, q, p>(A, B, C, D);
	ss.set_state(x);
	ss.set_input(u);

	// Iterate for 10 loops
	for (int i = 0; i < 10; i++)
	{
		// Calculate
		ss.propogate();

		// Output
		std::cout << "State Estimate @ t = " << (i + 1) << ": \n" << ss.x << std::endl;
		std::cout << "Output Estimate @ t = " << (i + 1) << ": " << ss.y << std::endl;
	}
}