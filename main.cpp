titulo("[EX03] SOMA DE MATRIZES 1000x1000 � 4 Opcoes do Professor");

    // Usando nl=nc=100 para n�o travar demos; para prova use 1000
    const int nl = 100, nc = 100;

    // Aloca��o din�mica para evitar stack overflow com matrizes grandes
    float** A = new float*[nl];
    float** B = new float*[nl];
    float** C = new float*[nl];
    for (int i = 0; i < nl; i++) {
        A[i] = new float[nc];
        B[i] = new float[nc];
        C[i] = new float[nc];
    }

    // Alimenta conforme o enunciado do professor: A[i][j]=i, B[i][j]=j
    for (int i = 0; i < nl; i++)
        for (int j = 0; j < nc; j++) {
            A[i][j] = (float)i;
            B[i][j] = (float)j;
        }

    // -- OP��O 1: Paraleliza o for EXTERNO (linhas) -- RECOMENDADO --
    sep("Opcao 1 � for externo (linhas) � RECOMENDADO");
    // Distribui linhas entre as threads. O for interno percorre s� as colunas,
    // sem overhead de sincroniza��o dentro de cada linha.
    #pragma omp parallel for num_threads(4)
    for (int i = 0; i < nl; i++) {        // paralelo: cada thread pega algumas linhas
        for (int j = 0; j < nc; j++) {    // serial dentro da thread
            C[i][j] = A[i][j] + B[i][j];
        }
    }
    printf("  C[5][3] = %.0f  (esperado: 5+3=8)\n", C[5][3]);
    printf("  C[10][7]= %.0f  (esperado: 10+7=17)\n", C[10][7]);

    // -- OP��O 2: Paraleliza o for INTERNO (colunas) -- N�O RECOMENDADO --
    sep("Opcao 2 � for interno (colunas) � NAO RECOMENDADO");
    // Para cada uma das nl linhas, cria e destr�i um time de threads.
    // Overhead de fork/join nl vezes ? muito custoso!
    for (int i = 0; i < nl; i++) {        // serial externo
        #pragma omp parallel for num_threads(4)
        for (int j = 0; j < nc; j++) {    // paralelo interno (nl fork/joins!)
            C[i][j] = A[i][j] + B[i][j];
        }
    }
    printf("  C[5][3] = %.0f  (correto mas LENTO � %d fork/joins)\n", C[5][3], nl);
    printf("  MOTIVO: para cada linha, OpenMP cria e destroi o time de threads.\n");

    // -- OP��O 3: Paraleliza AMBOS os fors -- N�O RECOMENDADO --
    sep("Opcao 3 � ambos os fors � NAO RECOMENDADO");
    // O for externo distribui linhas para as threads.
    // O for interno, dentro de uma regi�o j� paralela, tenta criar MAIS threads.
    // Isso gera overhead de cria��o/sincroniza��o que supera o ganho.
    #pragma omp parallel for num_threads(4)
    for (int i = 0; i < nl; i++) {
        #pragma omp parallel for num_threads(2)  // paralelo ANINHADO � alto custo
        for (int j = 0; j < nc; j++) {
            C[i][j] = A[i][j] + B[i][j];
        }
    }
    printf("  C[5][3] = %.0f  (correto mas MAIS LENTO � criacao de threads aninhadas)\n", C[5][3]);
    printf("  MOTIVO: overhead de criar threads internas > ganho de paralelismo.\n");

    // -- OP��O 4: collapse(2) -- SOLU��O IDEAL --
    sep("Opcao 4 � collapse(2) � SOLUCAO IDEAL do professor");
    // collapse(2) transforma os 2 fors aninhados em 1 �nico loop de nl*nc itera��es.
    // Garante distribui��o uniforme SEM overhead adicional.
    // REQUISITO: os fors devem ser perfeitamente aninhados (sem c�digo entre eles).
    #pragma omp parallel for collapse(2) num_threads(4)
    for (int i = 0; i < nl; i++) {
        for (int j = 0; j < nc; j++) {   // collapse(2): trata como loop �nico de nl*nc
            C[i][j] = A[i][j] + B[i][j];
        }
    }
    printf("  C[5][3]   = %.0f  (esperado: 8)\n", C[5][3]);
    printf("  C[99][99] = %.0f  (esperado: 198)\n", C[99][99]);
    printf("  MOTIVO: distribui nl*nc=%d iteracoes igualmente � zero overhead extra.\n", nl*nc);

    // Libera mem�ria
    for (int i = 0; i < nl; i++) { delete[] A[i]; delete[] B[i]; delete[] C[i]; }
    delete[] A; delete[] B; delete[] C;
