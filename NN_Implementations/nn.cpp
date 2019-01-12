#include <iostream>
#include <chrono>
#include <map>
#include <math.h>

using namespace std;
using namespace std::chrono;

class FasterNet {

	public:
		double w1 [2][2];
		double w2 [2][1];
		double h1 [2];
		double h2;
		double alpha = 1;
	
	void initialize(){
		
		for(int i=0;i<2;++i){
			for(int j=0;j<2;++j){
				w1[i][j] = 1;
			}
			w2[i][0] = 1;
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

struct Pos{
	double row;
	double col;
};

Pos move(Pos pos, int move){
	if(move==0){
		return Pos{pos.row, pos.col+1};
	}
	if(move==1){
		return Pos{pos.row, pos.col-1};
	}
	if(move==2){
		return Pos{pos.row+1, pos.col};
	}
	if(move==3){
		return Pos{pos.row-1, pos.col};
	}
}

int select_action(Pos pos, FasterNet fnet){
		double max;
		double qval = 0;
		int move;

		if(pos.row+1 < 5){
			max = fnet.forward(new double [2]{pos.row+1, pos.col});
			move = 2;
		}
		if(pos.row-1 >= 0){
			qval = fnet.forward(new double [2]{pos.row-1, pos.col});
			if(max < qval){
				max = qval;
				move = 3;
			}
		}
		if(pos.col-1 >= 0){
			qval = fnet.forward(new double [2]{pos.row, pos.col-1});
			if(max < qval){
				max = qval;
				move = 1;
			}
		}
		if(pos.col+1 < 5){
			qval = fnet.forward(new double [2]{pos.row, pos.col+1});
			if(max < qval){
				max = qval;
				move = 0;
			}
		}
	return move;
}

void update(FasterNet fnet, Pos pos){

}

//keras: 2800 microseconds
//fasterNet: 206 microseconds
//net: 2000 microseconds
//fortran: 850 microseconds- seem to be fluctuating between 1200-170 us
int main(){

	FasterNet fnet;
	fnet.initialize();

	Pos pos = Pos{1, 3};
	
	while (1) 
	{	
		cout << pos.row <<", " <<pos.col << endl;
		pos = move(pos, select_action(pos, fnet));
		if(pos.row==4 & pos.col==4){
			cout << pos.row <<", " <<pos.col << endl;
			break;
		}
		update(fnet, pos);
	}

	for(int i=0;i<5;i++){
		for(int j=0;j<5;j++){
			cout << fnet.forward(new double [2]{double(i), double(j)}) << "  ";
		}
		cout << endl;
	}
	
	
	return 0;
}
