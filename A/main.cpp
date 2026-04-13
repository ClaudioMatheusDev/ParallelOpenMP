#include <iostream>


int main(int argc, char** argv) {
    // Dimensões: A é 4x3 e B é 3x5 ? C será 4x5
    const int nl = 4;  // linhas de A e C
    const int nk = 3;  // colunas de A = linhas de B
    const int nc = 5;  // colunas de B e C

    double A[nl][nk], B[nk][nc], C[nl][nc];

    // Inicializa A: A[i][k] = i + k + 1
    for (int i = 0; i < nl; i++)
        for (int k = 0; k < nk; k++)
            A[i][k] = i + k + 1;

    // Inicializa B: B[k][j] = k + j + 1
    for (int k = 0; k < nk; k++)
        for (int j = 0; j < nc; j++)
            B[k][j] = k + j + 1;

    for (int i = 0; i < nl; i++)
        for (int j = 0; j < nc; j++)
            C[i][j] = 0;

    // ----------------------------------------
    // VERSÃO SERIAL (sem paralelismo)
    // ----------------------------------------
    std::cout << "--- Versão Serial ---" << std::endl;

    int i, j, k; // declaradas ANTES do parallel (serão usadas com private)

    for (i = 0; i < nl; i++) {
        for (j = 0; j < nc; j++) {
            C[i][j] = 0;
            for (k = 0; k < nk; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    std::cout << "Matriz C (serial):" << std::endl;
    for (i = 0; i < nl; i++) {
        for (j = 0; j < nc; j++)
            std::cout << C[i][j] << "\t";
        std::cout << std::endl;
    }

    // Reseta C
    for (i = 0; i < nl; i++)
        for (j = 0; j < nc; j++) C[i][j] = 0;

    // ----------------------------------------
    // VERSÃO PARALELA (com private)
    // ----------------------------------------
    std::cout << "\n--- Versão Paralela (com cláusula private) ---" << std::endl;

    // ? private(i, j, k) ? cada thread tem sua PRÓPRIA cópia de i, j, k ?
    // ? o #pragma omp for divide as iterações do for mais externo (linhas) ?
    #pragma omp parallel private(i, j, k)
    {
        #pragma omp for
        for (i = 0; i < nl; i++) {
            for (j = 0; j < nc; j++) {
                C[i][j] = 0;
                for (k = 0; k < nk; k++) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
    }

    std::cout << "Matriz C (paralelo):" << std::endl;
    for (i = 0; i < nl; i++) {
        for (j = 0; j < nc; j++)
            std::cout << C[i][j] << "\t";
        std::cout << std::endl;
    }

    std::cout << "\n(Os resultados serial e paralelo devem ser idênticos)" << std::endl;

    return 0;
	return 0;
}
