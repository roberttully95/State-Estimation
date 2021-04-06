#pragma once

#include <Eigen/Core>

/**
 * @brief Defines the state space model.
 * @tparam T The type of data contained in the model.
 * @tparam n The number of states in the system.
 * @tparam q The number of inputs to the system.
 * @tparam p The number of outputs of the system.
*/
template <typename T, int n, int q, int p>
class LinearStateSpaceModel
{
public:
	/**
	 * @brief Default constructor. Initializes everything to zero.
	*/
	LinearStateSpaceModel()
	{
		this->A.setZero();
		this->x.setZero();
		this->B.setZero();
		this->u.setZero();
		this->y.setZero();
		this->C.setZero();
		this->D.setZero();
	}

	/**
	 * @brief Constructor with the state-space matrices provided.
	 * @param A The state transition matrix.
	 * @param B The control matrix.
	 * @param C The output matrix.
	 * @param D The feed-forward matrix.
	*/
	LinearStateSpaceModel(const Eigen::Matrix<T, n, n>& A, const Eigen::Matrix<T, n, p>& B, const Eigen::Matrix<T, q, n>& C, const Eigen::Matrix<T, q, p>& D)
	{
		this->A = A;
		this->x.setZero();
		this->B = B;
		this->u.setZero();
		this->y.setZero();
		this->C = C;
		this->D = D;
	}

	~LinearStateSpaceModel() = default;

	// Define the dimesions of the system
	Eigen::Matrix<T, n, n> A;
	Eigen::Matrix<T, n, 1> x;
	Eigen::Matrix<T, n, p> B;
	Eigen::Matrix<T, p, 1> u;
	Eigen::Matrix<T, q, 1> y;
	Eigen::Matrix<T, q, n> C;
	Eigen::Matrix<T, q, p> D;

	/**
	 * @brief Defines the state-space matrices.
	 * @param A The state transition matrix.
	 * @param B The control matrix.
	 * @param C The output matrix.
	 * @param D The feed-forward matrix.
	*/
	void set_state_matrices(const Eigen::Matrix<T, n, n>& A, const Eigen::Matrix<T, n, p>& B, const Eigen::Matrix<T, q, n>& C, const Eigen::Matrix<T, q, p>& D)
	{
		this->A = A;
		this->B = B;
		this->C = C;
		this->D = D;
	}

	/**
	 * @brief Overrides the current state of the system.
	 * @param x The n x 1 vector containing the state of the system.
	*/
	void set_state(const Eigen::Matrix<T, n, 1>& x)
	{
		this->x = x;
	}

	/**
	 * @brief Overrides the current input of the system.
	 * @param u The 1 x p vector containing the state of the system.
	*/
	void set_input(const Eigen::Matrix<T, 1, p>& u)
	{
		this->u = u;
	}

	/**
	 * @brief Propogates the system based on the current state and updates the state variables and output.
	*/
	void propogate()
	{
		auto x_dot = A * x + B * u;
		y = C * x + D * u;
		x = x_dot;
	}
};
