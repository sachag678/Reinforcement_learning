#include <iostream>
#include <Eigen/Dense>
#include <vector>

using namespace std;
using Eigen::MatrixXd;

int main()
{	
	MatrixXd in = MatrixXd::Random(2,2);
	MatrixXd *ptrIn;

	ptrIn = &in;

	cout << ptrIn << endl;
}
