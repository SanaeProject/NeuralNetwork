#include "matrixtest.hpp"
#include "nntest.hpp"
#include "matrixbenchmark.hpp"

int main(){
    run_matrix_tests();
    run_benchmarks();
    run_nntest();
    
    return 0;
}