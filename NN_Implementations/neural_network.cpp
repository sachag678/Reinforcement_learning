#include <iostream>
#include <vector>
#include <math.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

class NeuralNetwork{
	private:
		double** weights;
		int num_weights;
		int* weights_size;
		int getSingleIndex(int row, int col, int numberOfColumnsInArray);
		int getLayerSize(int firstIndex, int secondIndex);

	public:
	        explicit NeuralNetwork(int params[], int size);
		double* createMatrix(int N, int M);
		void printNetwork();
		double* forward(double input[]);
};

int NeuralNetwork::getLayerSize(int firstIndex, int secondIndex){
	if(firstIndex>=num_weights){
		return weights_size[firstIndex];
	}else{
		return weights_size[firstIndex]*weights_size[secondIndex];
	}
}

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
			w[getSingleIndex(i, j, M)] = (i+1)*(j+1);//((double) rand() / (RAND_MAX));
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

double* NeuralNetwork::forward(double input[]){
	//multiply input with the weights using dot product
	double* output = new double[weights_size[num_weights]];
	for(int k=0;k<num_weights;k++){
		double* inner = new double[getLayerSize(k+1, k+2)];
		for(int i=0;i<weights_size[k+1];i++){
			double sum = 0;
			for(int j=0;j<weights_size[k];j++){
				sum = sum + input[j]*weights[k][getSingleIndex(i, j, weights_size[k+1])];
			}
			inner[i] = sum;
		}
		if(k+1==num_weights){
			output = inner;
		}else{
			input = inner;
		}
	}

	return output;
}



int main(){
	int* params = new int[3]{2, 2, 1};
	NeuralNetwork nn = NeuralNetwork(params, 3);
	nn.printNetwork();

	double* input = new double[2]{2, 1};
	double* output = nn.forward(input);
	
	cout << "Output from Network for input: {2, 1}"<< endl;
	cout << output[0] << endl;
}
