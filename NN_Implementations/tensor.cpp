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

		Tensor2D operator[](int index){
			if(rowSize!=1){
				Tensor2D result(1, colSize);
				for(int j = 0; j<colSize; j++){
					result.data[j] = data[index*colSize + j];
				}
				return result;
			}else{
				Tensor2D result(1, 1);
				result.data[index] = data[index];
				return result;
			}
		}

		Tensor2D mmul(Tensor2D& other){
			if(colSize != other.rowSize){
				throw std::invalid_argument( "ERROR: Inner col and row size do not match.");
			}

			Tensor2D result(rowSize, other.colSize);

			for(int i = 0; i < rowSize; i++){
				for(int j = 0; j < other.colSize; j++){
					double sum = 0;
					for(int k = 0; k < colSize; k ++){
						sum += data[i*colSize + k]*other.data[k*other.colSize + j];
					}
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

	Tensor2D t2(4, 1);

	t.Print();
	t2.Print();

	t.mmul(t2).Print();

	t[0][0].Print();
}

int main(){
	Tensor2D in(1, 50);
	Tensor2D l1(50, 1024);
	Tensor2D l2(1024, 256);
	Tensor2D l3(256, 10);

	in.mmul(l1).mmul(l2).mmul(l3).Print();
}
