#include <iostream>
#include <vector>
#include <math.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

class NeuralNetwork{
	private:
		double** weights;
		int getSingleIndex(int row, int col, int numberOfColumnsInArray);
		int num_weights;
		int* weights_size;

	public:
	        explicit NeuralNetwork(int params[], int size);
		double* createMatrix(int N, int M);
		void printNetwork();
		double* forward(int input[]);
};

int NeuralNetwork::getSingleIndex(int row, int col, int numberOfColumnsInArray){
	return (row*numberOfColumnsInArray) + col;
}

NeuralNetwork::NeuralNetwork(int params[], int size){
	weights = new double*[size];
	num_weights = size-1;
	weights_size = params;

	for(int i=0; i<size-1; i++){
		weights[i] = createMatrix(params[i], params[i+1]);
	}
}

double* NeuralNetwork::createMatrix(int N, int M){
	double* w = new double[N*M];

	for(int i=0;i<N;i++){
		for(int j=0;j<M;j++){
			w[getSingleIndex(i, j, M)] = ((double) rand() / (RAND_MAX));
		}
	}

	return w;
}

void NeuralNetwork::printNetwork(){
	for(int i=0;i<num_weights;i++){
		cout<< "Weight " << i << endl;
		for(int j=0;j<weights_size[i];j++){
			for(int k=0;k<weights_size[i+1];k++){
			cout<< weights[i][getSingleIndex(j, k, weights_size[i+1])] << " ";	
			}
			cout << endl;
		}
	}
}

double* NeuralNetwork::forward(int input[]){
	//multiply input with the weights using dot product
}

int main(){
	int* params = new int[3]{2, 2, 1};
	NeuralNetwork nn = NeuralNetwork(params, 3);
	nn.printNetwork();
}
