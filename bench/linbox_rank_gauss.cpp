#include "linbox/algorithms/gauss.h"
#include "linbox/solutions/rank.h"

#include <givaro/modular.h>
#include <iostream>

#define STOR_T SparseMatrixFormat::SparseSeq

using namespace LinBox;

int main (int argc, char **argv)
{
    typedef Givaro::Modular<uint32_t> Field;
    typedef SparseMatrix<Field, STOR_T> Blackbox;

    /* Params */
    static integer q = 42013U;
    std::string matfile = "data/sms.matrix";

    Field F(q);

    MatrixStream<Field> ms(F, std::cin); // read matrix on standard input
    Blackbox A(ms);
    size_t m, n;
    ms.getDimensions(m, n);

    /* Elements retournés */
    unsigned long rank;

    Method::Elimination ME;
    LinBox::rank(rank, A, ME);

    /* Output */
    std::cout << "====== RESULT =====" << std::endl << std::endl;
    std::cout << "Rank: " << rank << std::endl;
    return 0;
}

