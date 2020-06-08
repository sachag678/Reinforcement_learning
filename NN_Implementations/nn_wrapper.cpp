#include <boost/python.hpp>
using namespace boost::python

class FasterNet {

	public:
		double w1 [2][2];
		double w2 [2][1];
		double h1 [2];
		double h2;
		double alpha = 0.001;
	
	void initialize(){
		
		for(int i=0;i<2;++i){
			for(int j=0;j<2;++j){
				w1[i][j] = ((double) rand() / (RAND_MAX));
			}
			w2[i][0] = ((double) rand() / (RAND_MAX));
		}
	}

	double forward(double in []){
		for(int i=0;i<2;++i){
			double sum = 0;
			for(int j=0;j<2;++j){
				sum = sum + in[j]*w1[i][j];
			}
			if(sum<0){
				sum = 0;
			}
			h1[i] = sum;
		}
		
		h2 = 0;
		for(int j=0; j<2;++j){
			h2 = h2 + h1[j]*w2[j][0];
		}
		return h2;
	}

	void fit(double in[], double y){
		h2 = forward(in);
		double h2_delta = y - h2;
		
		double h1_delta[2];
		for(int i =0;i<2;++i){
			if(h2_delta*w2[i][0]>0){
				h1_delta[i] = h2_delta*w2[i][0];
			}else{
				h1_delta[i] = 0;
			}
		}

		//update weights
		for(int i=0; i<2;++i){
			for(int j=0; j<2;++j){
				w1[i][j] = w1[i][j] - alpha*in[i]*h1_delta[j];
			}
			w2[i][0]=w2[i][0] - alpha*h1[i]*h2_delta;
		}
	}
};

BOOST_PYTHON_MODULE(fnet)
{

	class <FasterNet>("FasterNet")
		.def("initialize", &FasterNet::initialize)
		.def("forward", &FasterNet::forward)
		.def("fit", &FasterNet::fit)
};
