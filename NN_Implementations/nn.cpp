#include <iostream>
#include <chrono>
#include <vector>

using namespace std;
using namespace std::chrono;

class Net {

	public: 
		vector<double*> weights;
		int inputSize;
		int outputSize;
		int hidddenSize;
		int numHiddenLayers;
	
	double initializeWeights(int inputSize, int outputSize, int hiddenSize, int numHiddenLayers){
		inputSize = inputSize;
		outputSize = outputSize;
		hiddenSize = hiddenSize;
		numHiddenLayers = numHiddenLayers;

		double* w1 = new double[inputSize][hiddenSize];
		double* wfinal = new double[hiddenSize][outputSize];

		for(int i=0;i<inputSize;++i){
			for(int j=0;j<hiddenSize;++j){
				w1[i][j] = 1;
			}
		}

		vector[0] = w1;

		int count = 1;
		for(int i =0;i<numHiddenLayers;++i){
			double * w = new double[hiddenSize][hiddenSize];

			for(int i=0;i<inputSize;++i){
				for(int j=0;j<hiddenSize;++j){
					w[i][j] = 1;
				}
			}
			vector[count] = w;
			count = count + 1;
		}
		
		for(int i=0;i<hiddenSize;++i){
			for(int j=0;j<hiddenSize;++j){
				wfinal[i][j] = 1;
			}
		}

		vector[count] = wfinal;

	}
	
	double forward(double input []){
		double h1[10];
		
		for(int i=0;i<10;++i){
			double sum = 0;
			for(int j=0;j<10;++j){
				sum= sum + input[i]*vector[0][i][j];
			}
			//Relu
			if(sum<0){
				sum = 0;
			}	
			h1[i] = sum;
		}
		
		double output = 0;
		for(int i=0;i<10;++i){
			output = output + h1[i]*vector[vector.length-1][i];
		}
		return output;
	}

};
//0.0048 time to beat.
int main(){
	Net net;

	net.initializeWeights();
	
	double input [10] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

    	auto start = high_resolution_clock::now();
	for(int i=0; i<1000;++i){
		net.forward(input);
	}

    	auto stop = high_resolution_clock::now();
    	auto duration = duration_cast<microseconds>(stop - start);
	
    	std::cout <<"iterative_factorial : " << duration.count() << std::endl;

	return 0;
}
