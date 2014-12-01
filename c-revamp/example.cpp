#include <iostream>
#include <armadillo>

using namespace std;
using namespace arma;

sp_mat test_add(sp_mat A, sp_mat B) {
    return A + B;
}
sp_mat test_mult(sp_mat A, sp_mat B) {
    return A % B;
}

int main(int argc, char** argv) {
  sp_mat A = sp_mat(4,5);
  sp_mat B = sp_mat(4,5);
  sp_mat C = sp_mat(4,4);
  
  A(3,1) = 5;
  B(2,1) = 3;
  
  C = A * B.t();
  C.print();
  //cout << A*B.t() << endl;
  sp_mat Bt = B.t();
  C = A * Bt;
  C.print();
  
  C = dot(A.row(3), B.row(2));
  C.print();
  double c = C(0,0);
  printf("%f\n", c);
 /*
  for (int i=0; i < 4; i++) { 
      for (int j=0; j < 4; j++) {
          printf("(%d,%d):\n", i,j);
          C.print();
          //C = A.row(i) * Bt.col(j);
          //C.print();
      }
  }*/

  //A(1)* (B.t())(1);
  C = sp_mat(4,5);
  A(1,1) = 3;
  B(1,1) = 12;
  for (double i = 0; i < 1e7; i++) {
    //C = test_add(A, B);
    C = test_mult(A, B);
  }
  
  return 0;
}
