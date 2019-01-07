#include <iostream>
#include <chrono>
#include <map>
#include <Eigen/Dense>
#include <Eigen/Eigen>
#include <math.h>

using namespace std;
using namespace std::chrono;
using namespace Eigen; 

class Net {

	public: 
		MatrixXd w1 = MatrixXd::Ones(10, 10);
		MatrixXd w2 = MatrixXd::Ones(10, 1);
	
	double relu(double val){
		if(val< 0){
			return 0;
		}else{
			return val;
		}
	}

	double tanh(double val){
		return tanh(val);	
	}
	
	MatrixXd forward(MatrixXd input){
		MatrixXd h1 = w1*input;
		
		for(int i=0;i<h1.rows();++i){
			for(int j=0;j<h1.cols();++j){
				h1(i,j) = relu(h1(i, j));
				}
			}

		MatrixXd h2 = w2.transpose()*h1;
		return h2;
	}

};

class FasterNet {

	public:
		double w1 [10][10];
		double w2 [10][1];
	
	void initialize(){
		cout << sizeof(w1.length) << endl;
		
		for(int i=0;i<10;++i){
			for(int j=0;j<10;++j){
				w1[i][j] = 1;
			}
			w2[i][0] = 1;
		}
	}

	double forward(double input []){
		double h1 [10];
		for(int i=0;i<10;++i){
			double sum = 0;
			for(int j=0;j<10;++j){
				sum = sum + input[j]*w1[i][j];
			}
			if(sum<0){
				sum = 0;
			}
			h1[i] = sum;
		}
		
		double h2 = 0;
		for(int j=0; j<10;++j){
			h2 = h2 + h1[j]*w2[j][0];
		}
		return h2;
	}
};

//0.0048 time to beat.
int main(){
	Net net;
	MatrixXd input = MatrixXd::Ones(10, 1);
	
    	auto start = high_resolution_clock::now();
	for(int i=0; i<1000;++i){
		net.forward(input);
	}

    	auto stop = high_resolution_clock::now();
    	auto duration = duration_cast<microseconds>(stop - start);
	
    	std::cout <<"iterative_factorial : " << duration.count() << std::endl;

	FasterNet fnet;
	fnet.initialize();

	double input2 [] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

    	start = high_resolution_clock::now();
	for(int i=0; i<1000;++i){
		fnet.forward(input2);
	}

    	stop = high_resolution_clock::now();
    	duration = duration_cast<microseconds>(stop - start);
	
    	std::cout <<"iterative_factorial : " << duration.count() << std::endl;

	return 0;
}
