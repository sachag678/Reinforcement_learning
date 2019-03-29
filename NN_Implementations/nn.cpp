#include <iostream>
#include <vector>
#include <math.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

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

int select_action(Pos pos, FasterNet fnet, int eps){
		double max;
		double qval = 0;
		int move;
		if(((double) rand() / (RAND_MAX))>eps){
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
		}else{	
			int actionCount = 0;
			std::vector<int> availableActions;	
			if(pos.row+1 < 5){
				actionCount ++;
				availableActions.push_back(2);
			}
			if(pos.row-1 >= 0){
				actionCount ++;
				availableActions.push_back(3);
			}
			if(pos.col-1 >= 0){
				actionCount ++;
				availableActions.push_back(1);
			}
			if(pos.col+1 < 5){
				actionCount ++;
				availableActions.push_back(0);
			}
			//choose actions that are available randomyl.
			int index = rand() % actionCount;
			move = availableActions[index];
		}
	return move;
}

int reward(Pos pos){
	if(pos.row==4 & pos.col==4){
		return 10;
	}
	return -1;
}

FasterNet update(FasterNet fnet, Pos pos, Pos new_pos){
	fnet.fit(new double [2]{pos.row, pos.col}, reward(new_pos) + fnet.forward(new double[2]{new_pos.row, new_pos.col}));
	return fnet;
}

void value_fcn(FasterNet fnet){
	for(int i=0;i<5;i++){
		for(int j=0;j<5;j++){
			cout << fnet.forward(new double [2]{double(i), double(j)}) << "  ";
		}
		cout << endl;
	}
	cout << "----------------------------------------------------" << endl;
}

//keras: 2800 microseconds
//fasterNet: 206 microseconds
//net: 2000 microseconds
//fortran: 850 microseconds- seem to be fluctuating between 1200-170 us
int main(){

	auto start = high_resolution_clock::now();
	FasterNet fnet;
	fnet.initialize();
	
	//value_fcn(fnet);
	int finalRow = 4;
	int finalCol = 4;


	for(int i=0;i<10;i++){
		Pos pos = Pos{0, 0};
		Pos new_pos;
		int numSteps = 0;
		while (1){ 
			new_pos = move(pos, select_action(pos, fnet, 0.3));
			if(new_pos.row==finalRow & new_pos.col==finalCol){
				break;
			}
			fnet = update(fnet, pos, new_pos);
			pos = new_pos;
			numSteps++;
		}
		//cout << "Number of Steps: "<< numSteps + 1 << endl;
		//value_fcn(fnet);
	}

	cout << "Training complete!" << endl;
	Pos pos = Pos{0, 0};
	Pos new_pos;
	while (1){
		//cout << pos.row << ", " << pos.col << endl;	
		new_pos = move(pos, select_action(pos, fnet, 0));
		if(new_pos.row==finalRow & new_pos.col==finalCol){
			//cout << new_pos.row << ", " << new_pos.col << endl;	
			break;
		}
		fnet = update(fnet, pos, new_pos);
		pos = new_pos;
	}
	cout << "Testing complete!" << endl;
	
	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop-start);
	cout << "Time: " << duration.count() << endl;
	return 0;
}
