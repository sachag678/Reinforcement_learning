#include <iostream>
#include <math.h>

class Tensor2D{
	public:
		double* data;
		int rowSize;
		int colSize;

		Tensor2D(int rSize, int cSize, bool randomInit = true){
			rowSize = rSize;
			colSize = cSize;
			data = new double[rowSize*colSize];

			if(randomInit){
				for(int i=0;i<rowSize*colSize;i++){
					data[i] = (double) rand() / (RAND_MAX);
				}
			}
		}

		Tensor2D(const Tensor2D& other){
			rowSize = other.rowSize;
			colSize = other.colSize;
			data = other.data;
		}

		Tensor2D& operator=(const Tensor2D& other){
			std::cout << "inside assignment constructor" << std::endl;
			return *this;
		}

		Tensor2D(int rSize, int cSize, double* array){
			rowSize = rSize;
			colSize = cSize;
			data = array;
		}

		Tensor2D operator[](int index){
			if(rowSize!=1){
				Tensor2D result(1, colSize, false);
				for(int j = 0; j<colSize; j++){
					result.data[j] = data[index*colSize + j];
				}
				return result;
			}else{
				Tensor2D result(1, 1, false);
				result.data[index] = data[index];
				return result;
			}
		}

		Tensor2D mmul(Tensor2D& other){
			if(colSize != other.rowSize){
				throw std::invalid_argument( "ERROR: Inner col and row size do not match.");
			}

			Tensor2D result(rowSize, other.colSize, false);

			for(int i = 0; i < rowSize; i++){
				for(int j = 0; j < other.colSize; j++){
					double sum = 0;
					for(int k = 0; k < colSize; k ++){
						sum += data[i*colSize + k]*other.data[k*other.colSize + j];
					}
					std::cout << sum << std::endl;
					result.data[i*colSize + j] = sum;
				}
			}
			return result;
		}

		void Print(){
			std::cout << "shape(" << rowSize << ", " << colSize << ")" <<std::endl;
			std::cout << "[";
			for(int i=0;i<rowSize;i++){
				std::cout << "[";
				for(int j=0;j<colSize;j++){
					std::cout <<data[i*colSize + j];
					if(j<colSize-1){
						std::cout << ", ";
					}else{
						if(i<rowSize-1){
							std::cout << "],";
						}else{
							std::cout << "]";
						}
					}
				}
				if(i==rowSize-1){
					std::cout << "]" <<std::endl; 
				}else{
					std::cout << std::endl;
				}
			}
		}

		void shape(){
			std::cout << "shape(" << rowSize << ", " << colSize << ")" <<std::endl;
		}
		
};

void test(){

	Tensor2D t(2, 4);

	t.Print();

	Tensor2D t2(4, 1);

	t2.Print();

	t.mmul(t2).Print();

	Tensor2D t5 = t[0];
	std::cout << "after return" << std::endl;
	std::cout << t5.data[0] << std::endl;
	std::cout << t5.data[1] << std::endl;
	std::cout << t5.data[2] << std::endl;
	std::cout << t5.data[3] << std::endl;

	/*double* input = new double[6];
	input[0] = 1;
	input[1] = 2;
	input[2] = 4;
	input[3] = 3;
	input[4] = 6;
	input[5] = 5;
*/
	//Tensor2D t3(1, 6, input);

	//Tensor2D t4(1, 5, false);
}

void ff(){

	Tensor2D in(1, 50);
	Tensor2D l1(50, 1024);
	Tensor2D l2(1024, 256);
	Tensor2D l3(256, 10);

	in.mmul(l1).mmul(l2).mmul(l3).Print();
}

int main(){
	test();
}
