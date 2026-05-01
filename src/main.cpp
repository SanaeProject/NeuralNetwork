#include "matrixtest.hpp"
#include "nntest.hpp"
#include "matrixbenchmark.hpp"
#include "include/matrix/bit/bitarray.hpp"

int main(){
    // run_matrix_tests();
    // run_benchmarks();
    // run_nntest();

    try{
        BitArray<> bits(11, true);
        for(size_t i = 0; i < bits.size(); ++i){
            bits.set(i, 0);
            std::cout << bits << std::endl;
        }

        bits.emplace_back(1);
        std::cout << bits << std::endl;
    }catch(std::exception e){
        std::cout << e.what() << std::endl;
    }
    
    return 0;
}