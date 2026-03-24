#include <cstddef>
#include <execution>
#include "./nnlearncomplete.hpp"

void run_matrix_tests();
void run_layertest();

int main() {
    // run_matrix_tests();
    // run_layertest();
    const size_t num_runs = 10;

    // 学習率とバッチサイズの組み合わせで学習を実行
    for(uint32_t batch_size : {1, 10, 20, 50, 100}) {
        std::cout << "バッチサイズ: " << batch_size << std::endl;

        for(float lr : {0.1f, 0.3f, 0.5f}) {
            std::cout << "    学習率: " << lr << std::endl;
            size_t iterations = 0;
            for(int i = 0; i < num_runs; i++) {
                iterations += run_nnlearn(lr, batch_size);
            }
            std::cout << "    平均学習回数: " << iterations / num_runs << std::endl;
        }
    }

    return 0;
}