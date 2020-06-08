#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <chrono>

using namespace std;
using namespace std::chrono;
using namespace Eigen;

class NeuralNetwork{
	private:
		vector<MatrixXd> weights;

	public:
		explicit NeuralNetwork(int params[], int size);
		MatrixXd createMatrix(int N, int M);
		MatrixXd forward(MatrixXd input);
		void printNetwork();	
};

void NeuralNetwork::printNetwork(){
	for(auto const element: weights){
		cout << element << endl;
	}
}

NeuralNetwork::NeuralNetwork(int params[], int size){
	for(int i=0; i<size-1;i++){
		weights.push_back(createMatrix(params[i], params[i+1]));
	}
}

MatrixXd NeuralNetwork::createMatrix(int N, int M){
	return MatrixXd::Random(N,M);
}

MatrixXd NeuralNetwork::forward(MatrixXd input){
	for(auto const element: weights){
		input = input*element;	
	}
	return input;
}


int main(){
	int inputSize = 3;
	int* params = new int[inputSize]{2, 50, 1};
	NeuralNetwork nn = NeuralNetwork(params, inputSize);

	MatrixXd input = MatrixXd::Random(1, 2);

	//running the timing test
	auto start = high_resolution_clock::now();
	for(int i=0;i<100000;i++){
		nn.forward(input);
	}
	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop-start);
	cout << "Time: " << duration.count()/1000000.0 << endl;
}
