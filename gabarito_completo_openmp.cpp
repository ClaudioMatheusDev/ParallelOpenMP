/*
================================================================================
  GABARITO COMPLETO — TÓPICOS AVANÇADOS EM COMPUTAÇÃO
  Prof. Dr. André Mendes Garcia — Programação Paralela com OpenMP
  
  GUIA DEFINITIVO DE ESTUDOS PARA A PROVA P1
  
  Compilar (Linux/Mac):  g++ -fopenmp -O2 -o gabarito gabarito_completo_openmp.cpp -lm
  Compilar (Windows):    g++ -fopenmp -O2 -o gabarito gabarito_completo_openmp.cpp
  Executar:              ./gabarito
================================================================================

  ÍNDICE COMPLETO DE EXERCÍCIOS:
  ─────────────────────────────────────────────────────────────────
  /// EXERCÍCIOS DO PDF (obrigatórios) ///
  [EX01] Soma de Vetores — Método Manual (limites por thread + correção última thread)
  [EX02] Soma de Vetores — Método Implícito (#pragma omp parallel for)
  [EX03] Soma de Matrizes 1000x1000 — 4 Opções do Professor (com análise)
  [EX04] Multiplicação de Matrizes — private(i,j,k) declarados FORA do parallel
  [EX05] Produto Escalar — Race condition (demonstração visual), critical, paux
  [EX06] Produto Escalar — Cláusula reduction (solução ideal do professor)
  [EX07] Cálculo de Pi — Versão com critical (exercício 1 do professor)
  [EX08] Cálculo de Pi — Versão com reduction + comparação de tempo
  [EX09] Fatoração LU — Sistema 3x3 do professor com L, U, resolução completa
  [EX10] Fatoração LU — Sistema 6x6 para evidenciar ganho de tempo
  [EX11] Matriz Inversa — Método Gauss-Jordan paralelizado
  ─────────────────────────────────────────────────────────────────
  /// EXTRAS QUE PODEM CAIR NA PROVA ///
  [EX12] Demonstração VISUAL de race condition (com vs sem critical)
  [EX13] Soma de Vetores com threads > tamanho do vetor (if inicio < n)
  [EX14] omp_get_wtime() — Medição de tempo, speedup e eficiência
  [EX15] Lei de Amdahl — Tabela completa S×p, conclusões
  [EX16] Cláusula IF — paraleliza só quando a carga justifica o overhead
  [EX17] Escopo de variáveis — bug silencioso global vs private correto
  [EX18] reduction com max e min ao mesmo tempo em um único parallel for
  [EX19] Norma Euclidiana — reduction(+:soma_quadrados) + sqrt serial
  [EX20] Contagem de elementos com condição — reduction(+:contador)
  [EX21] firstprivate vs private — diferença com inicialização do valor original
  [EX22] Multiplicação Matriz × Vetor paralelizada
  [EX23] Inicialização paralela de matriz com collapse(2)
  ─────────────────────────────────────────────────────────────────
  [COLA] COLA DA PROVA — diretivas, pegadinhas, fórmulas, regras
  ─────────────────────────────────────────────────────────────────
*/

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <climits>    // INT_MIN, INT_MAX
#include <cfloat>     // FLT_MAX
#include <string>
#include <omp.h>

// ─── Utilitários de exibição ──────────────────────────────────────────────────
static void titulo(const char* t) {
    printf("\n╔══════════════════════════════════════════════════════════╗\n");
    printf("  %s\n", t);
    printf("╚══════════════════════════════════════════════════════════╝\n");
}
static void sep(const char* t) {
    printf("\n  ── %s ──\n", t);
}
static void linha() {
    printf("  %s\n", std::string(58, '-').c_str());
}


// ─────────────────────────────────────────────────────────────────────────────
//  [EX01] SOMA DE VETORES — Método Manual (limites por thread)
//
//  ORIGEM  : Exercício 1 do PDF, slide "OpenMP – Soma de Vetores"
//  ENUNCIADO: "Fazer um programa em C/C++ que alimente dois vetores de 1000
//             elementos cada e efetue a soma destes num terceiro vetor de
//             forma paralelizada"
//
//  TEORIA:
//    Cada thread calcula seu próprio intervalo [inicio..fim].
//    tamanho = n / nt           — parcela base
//    inicio  = id * tamanho
//    fim     = inicio + tamanho - 1
//    PROBLEMA: se n não é divisível por nt, sobram elementos.
//    SOLUÇÃO : para a ÚLTIMA thread (id == nt-1), forçar fim = n-1.
//
//  O QUE PODE SER PARALELIZADO: o loop de soma — cada posição é independente.
//  O QUE NÃO PODE: nada impede; não há dependência entre C[i] e C[j].
// ─────────────────────────────────────────────────────────────────────────────
void ex01_soma_vetores_manual() {
    titulo("[EX01] SOMA DE VETORES — Metodo Manual (limites por thread)");

    const int n = 1000;
    static float A[n], B[n], C[n];

    // Alimenta vetores com valores quaisquer (conforme slide do professor)
    for (int i = 0; i < n; i++) {
        A[i] = (float)(i * sin((double)i));
        B[i] = A[i] - (float)cos((double)(i * i));
    }

    // ── VERSÃO SERIAL (referência) ──
    static float C_serial[n];
    for (int i = 0; i < n; i++) C_serial[i] = A[i] + B[i];

    // ── VERSÃO PARALELA — método manual ──
    // FORK: variáveis de controle declaradas DENTRO → privadas automaticamente
    #pragma omp parallel num_threads(4)
    {
        int id      = omp_get_thread_num();   // ID desta thread (0..nt-1)
        int nt      = omp_get_num_threads();  // número total de threads no time
        int tamanho = n / nt;                 // parcela base (pode ser truncada)
        int inicio  = id * tamanho;           // primeiro índice desta thread
        int fim     = inicio + tamanho - 1;   // último índice (provisório)

        // CORREÇÃO para a última thread: garante que todos os elementos são cobertos.
        // Exemplo: n=1000, nt=3 → tamanho=333. Threads 0,1 cobrem [0..333] e [334..666].
        // Thread 2: sem correção ficaria em [667..999] mas tamanho seria 333, indo até 999. OK.
        // Para n=100 e nt=3: thread 0→[0..33], thread 1→[33..66], thread 2→[66..99] via correção.
        if (id == nt - 1) {
            fim     = n - 1;
            tamanho = fim - inicio + 1;
        }

        // Proteção: se (por alguma razão) threads > n, thread sem trabalho não faz nada
        if (inicio <= fim && inicio < n) {
            for (int i = inicio; i <= fim; i++) {
                C[i] = A[i] + B[i];   // soma da parcela desta thread
            }
        }
    } // JOIN: todas as threads terminam aqui

    // Validação
    int erros = 0;
    for (int i = 0; i < n; i++)
        if (C[i] != C_serial[i]) erros++;

    printf("  Threads: 4  |  Vetor: n=%d\n", n);
    printf("  C[0]   = %.4f\n", C[0]);
    printf("  C[499] = %.4f\n", C[499]);
    printf("  C[999] = %.4f\n", C[999]);
    printf("  Erros vs serial: %d (deve ser 0)\n", erros);
}


// ─────────────────────────────────────────────────────────────────────────────
//  [EX02] SOMA DE VETORES — Método Implícito (#pragma omp parallel for)
//
//  ORIGEM  : Exercício 1 do PDF — versão com construtor de trabalho for
//
//  TEORIA:
//    O compilador divide as iterações do for automaticamente entre as threads.
//    O índice "i" é tratado como variável PRIVADA automaticamente.
//    Dois modos equivalentes:
//      a) #pragma omp parallel for      (merged — mais comum)
//      b) #pragma omp parallel { #pragma omp for }   (separado)
//
//  QUAL É MELHOR? O professor diz que as duas são equivalentes em desempenho.
//  A forma manual (EX01) permite maior controle para casos especiais.
// ─────────────────────────────────────────────────────────────────────────────
void ex02_soma_vetores_parallel_for() {
    titulo("[EX02] SOMA DE VETORES — Metodo Implicito (#pragma omp parallel for)");

    const int n = 1000;
    static float A[n], B[n], C[n];

    for (int i = 0; i < n; i++) {
        A[i] = (float)(i * sin((double)i));
        B[i] = A[i] - (float)cos((double)(i * i));
    }

    // FORMA 1 — #pragma omp parallel for (merged)
    // "i" é automaticamente privada; OpenMP distribui as iterações
    #pragma omp parallel for num_threads(4)
    for (int i = 0; i < n; i++) {
        C[i] = A[i] + B[i];   // sem race condition: cada "i" é único por thread
    }
    printf("  Forma 1 (parallel for merged):   C[500] = %.4f\n", C[500]);

    // FORMA 2 — parallel + for separados (equivalente à forma 1)
    #pragma omp parallel num_threads(4)
    {
        // A variável i declarada dentro → privada automaticamente
        #pragma omp for   // construtor de trabalho: distribui as iterações
        for (int i = 0; i < n; i++) {
            C[i] = A[i] + B[i];
        }
    }
    printf("  Forma 2 (parallel + for sep.):   C[500] = %.4f\n", C[500]);
    printf("  [OK] Ambas produzem resultado identico\n");
}


// ─────────────────────────────────────────────────────────────────────────────
//  [EX03] SOMA DE MATRIZES 1000×1000 — 4 Opções do Professor
//
//  ORIGEM  : Exercício 2 do PDF — slides "OpenMP – Soma de Matrizes"
//  ENUNCIADO: "Fazer um programa que alimente duas matrizes de 1000x1000 com:
//             A[i][j] = i;  B[i][j] = j
//             Em seguida faça a soma das matrizes A e B na matriz C
//             de forma paralelizada"
//
//  C[i][j] = A[i][j] + B[i][j] = i + j
//
//  OPÇÕES DO PROFESSOR (análise de qual for paralelizar):
//  ┌─────────────────────────────────────────────────────┐
//  │ Opção 1: for externo (linhas)     → RECOMENDADO    │
//  │ Opção 2: for interno (colunas)    → NÃO RECOMENDADO│
//  │ Opção 3: ambos os fors            → NÃO RECOMENDADO│
//  │ Opção 4: collapse(2)              → SOLUÇÃO IDEAL  │
//  └─────────────────────────────────────────────────────┘
// ─────────────────────────────────────────────────────────────────────────────
void ex03_soma_matrizes() {
    titulo("[EX03] SOMA DE MATRIZES 1000x1000 — 4 Opcoes do Professor");

    // Usando nl=nc=100 para não travar demos; para prova use 1000
    const int nl = 100, nc = 100;

    // Alocação dinâmica para evitar stack overflow com matrizes grandes
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

    // ── OPÇÃO 1: Paraleliza o for EXTERNO (linhas) ── RECOMENDADO ──
    sep("Opcao 1 — for externo (linhas) — RECOMENDADO");
    // Distribui linhas entre as threads. O for interno percorre só as colunas,
    // sem overhead de sincronização dentro de cada linha.
    #pragma omp parallel for num_threads(4)
    for (int i = 0; i < nl; i++) {        // paralelo: cada thread pega algumas linhas
        for (int j = 0; j < nc; j++) {    // serial dentro da thread
            C[i][j] = A[i][j] + B[i][j];
        }
    }
    printf("  C[5][3] = %.0f  (esperado: 5+3=8)\n", C[5][3]);
    printf("  C[10][7]= %.0f  (esperado: 10+7=17)\n", C[10][7]);

    // ── OPÇÃO 2: Paraleliza o for INTERNO (colunas) ── NÃO RECOMENDADO ──
    sep("Opcao 2 — for interno (colunas) — NAO RECOMENDADO");
    // Para cada uma das nl linhas, cria e destrói um time de threads.
    // Overhead de fork/join nl vezes → muito custoso!
    for (int i = 0; i < nl; i++) {        // serial externo
        #pragma omp parallel for num_threads(4)
        for (int j = 0; j < nc; j++) {    // paralelo interno (nl fork/joins!)
            C[i][j] = A[i][j] + B[i][j];
        }
    }
    printf("  C[5][3] = %.0f  (correto mas LENTO — %d fork/joins)\n", C[5][3], nl);
    printf("  MOTIVO: para cada linha, OpenMP cria e destroi o time de threads.\n");

    // ── OPÇÃO 3: Paraleliza AMBOS os fors ── NÃO RECOMENDADO ──
    sep("Opcao 3 — ambos os fors — NAO RECOMENDADO");
    // O for externo distribui linhas para as threads.
    // O for interno, dentro de uma região já paralela, tenta criar MAIS threads.
    // Isso gera overhead de criação/sincronização que supera o ganho.
    #pragma omp parallel for num_threads(4)
    for (int i = 0; i < nl; i++) {
        #pragma omp parallel for num_threads(2)  // paralelo ANINHADO — alto custo
        for (int j = 0; j < nc; j++) {
            C[i][j] = A[i][j] + B[i][j];
        }
    }
    printf("  C[5][3] = %.0f  (correto mas MAIS LENTO — criacao de threads aninhadas)\n", C[5][3]);
    printf("  MOTIVO: overhead de criar threads internas > ganho de paralelismo.\n");

    // ── OPÇÃO 4: collapse(2) ── SOLUÇÃO IDEAL ──
    sep("Opcao 4 — collapse(2) — SOLUCAO IDEAL do professor");
    // collapse(2) transforma os 2 fors aninhados em 1 único loop de nl*nc iterações.
    // Garante distribuição uniforme SEM overhead adicional.
    // REQUISITO: os fors devem ser perfeitamente aninhados (sem código entre eles).
    #pragma omp parallel for collapse(2) num_threads(4)
    for (int i = 0; i < nl; i++) {
        for (int j = 0; j < nc; j++) {   // collapse(2): trata como loop único de nl*nc
            C[i][j] = A[i][j] + B[i][j];
        }
    }
    printf("  C[5][3]   = %.0f  (esperado: 8)\n", C[5][3]);
    printf("  C[99][99] = %.0f  (esperado: 198)\n", C[99][99]);
    printf("  MOTIVO: distribui nl*nc=%d iteracoes igualmente — zero overhead extra.\n", nl*nc);

    // Libera memória
    for (int i = 0; i < nl; i++) { delete[] A[i]; delete[] B[i]; delete[] C[i]; }
    delete[] A; delete[] B; delete[] C;
}


// ─────────────────────────────────────────────────────────────────────────────
//  [EX04] MULTIPLICAÇÃO DE MATRIZES — private(i, j, k)
//
//  ORIGEM  : Slides "OpenMP – Multiplicação de Matrizes"
//
//  FÓRMULA : C[i][j] = Σ_{k=0}^{ncA-1} A[i][k] * B[k][j]
//
//  PONTO CRÍTICO DO PROFESSOR:
//    Variáveis i, j, k declaradas ANTES do #pragma omp parallel
//    → são GLOBAIS → todas as threads compartilham → RACE CONDITION!
//    SOLUÇÃO: usar cláusula private(i, j, k)
//
//  O QUE PODE SER PARALELIZADO: loop de i (linhas de C) e de j (colunas de C)
//  O QUE NÃO PODE: loop de k (acumulação de produto) — dependência na soma
// ─────────────────────────────────────────────────────────────────────────────
void ex04_multiplicacao_matrizes() {
    titulo("[EX04] MULTIPLICACAO DE MATRIZES — private(i,j,k) obrigatorio");

    // A[la×ca] × B[ca×cb] = C[la×cb]
    const int la = 3, ca = 3, cb = 3;
    float A[la][ca] = {{1,2,3},{4,5,6},{7,8,9}};
    float B[ca][cb] = {{9,8,7},{6,5,4},{3,2,1}};
    float C[la][cb];

    // ── VERSÃO SEM PARALELISMO (referência) ──
    sep("Versao serial (referencia)");
    for (int i = 0; i < la; i++) {
        for (int j = 0; j < cb; j++) {
            C[i][j] = 0;
            for (int k = 0; k < ca; k++)
                C[i][j] += A[i][k] * B[k][j];
        }
    }
    printf("  C[0][0]=%.0f  C[1][1]=%.0f  C[2][2]=%.0f\n", C[0][0], C[1][1], C[2][2]);

    // ── VERSÃO PARALELA — variáveis declaradas FORA (como o professor ensina) ──
    sep("Versao paralela — i,j,k declaradas FORA = precisam de private");

    // O professor declara i, j, k ANTES do parallel (estilo C90 / situação real)
    // Isso as torna GLOBAIS → se não usar private, threads pisam nos valores umas das outras
    int i, j, k;  // declaradas FORA → globais → precisam de private!

    // #pragma omp parallel for + private: cria cópias INDEPENDENTES de i, j, k em cada thread
    // NOTA: O professor mostra que se pode combinar parallel e for em um mesmo #pragma,
    //       desde que não haja mais um construtor de trabalho dentro dessa região.
    #pragma omp parallel for num_threads(4) private(i, j, k)
    for (i = 0; i < la; i++) {              // distribuído entre as threads
        for (j = 0; j < cb; j++) {          // privado desta thread
            C[i][j] = 0;
            for (k = 0; k < ca; k++) {      // acumulação — NÃO pode ser paralelizado
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    printf("  Resultado C = A * B:\n");
    for (int ii = 0; ii < la; ii++) {
        printf("  [");
        for (int jj = 0; jj < cb; jj++) printf("%6.1f", C[ii][jj]);
        printf("  ]\n");
    }
    printf("  Linha 0 esperada: [30, 24, 18]\n");
    printf("  Linha 1 esperada: [84, 69, 54]\n");
    printf("  Linha 2 esperada: [138,114, 90]\n");

    // VERSÃO ALTERNATIVA: variáveis declaradas DENTRO do bloco parallel
    sep("Versao alternativa — variaveis declaradas DENTRO (sem private necessario)");
    #pragma omp parallel num_threads(4)
    {
        // Declarar i,j,k DENTRO do bloco parallel → privadas automaticamente
        // Esta é a forma mais segura e recomendada
        int ii, jj, kk;
        #pragma omp for
        for (ii = 0; ii < la; ii++) {
            for (jj = 0; jj < cb; jj++) {
                C[ii][jj] = 0;
                for (kk = 0; kk < ca; kk++)
                    C[ii][jj] += A[ii][kk] * B[kk][jj];
            }
        }
    }
    printf("  C[2][2] = %.0f (esperado: 90)\n", C[2][2]);
}


// ─────────────────────────────────────────────────────────────────────────────
//  [EX05] PRODUTO ESCALAR — Demonstração de race condition + critical + paux
//
//  ORIGEM  : Exercício 1 do slide "OpenMP – Região Crítica"
//  ENUNCIADO: "Implementar um programa que alimente dois vetores A e B com os
//             respectivos valores de suas posições:  A[i]=i;  B[i]=i
//             Em seguida calcule o produto escalar utilizando a diretiva para
//             proteção da região crítica."
//
//  PRODUTO ESCALAR: p = A[0]*B[0] + A[1]*B[1] + ... + A[n-1]*B[n-1]
//                      = Σ i² para i=0..n-1
//
//  REGIÃO CRÍTICA: a variável "p" é pública. Sem proteção, múltiplas threads
//  fazem p += valor ao mesmo tempo → lêem um valor desatualizado → ERRO!
//
//  3 VERSÕES:
//  a) COM RACE CONDITION (errada — demonstração)
//  b) COM critical (correto porém lento — critical executa n vezes)
//  c) COM paux + critical (bom desempenho — critical executa nt vezes)
// ─────────────────────────────────────────────────────────────────────────────
void ex05_produto_escalar_critical() {
    titulo("[EX05] PRODUTO ESCALAR — race condition + critical + paux");

    const int n = 100;
    static float A[n], B[n];

    // A[i] = i;  B[i] = i  (conforme enunciado do professor)
    for (int i = 0; i < n; i++) { A[i] = (float)i; B[i] = (float)i; }

    // Resultado esperado: Σ i² para i=0..n-1 = n*(n-1)*(2n-1)/6
    // Para n=100: 0+1+4+9+...+9801 = 328350
    float esperado = 0;
    for (int i = 0; i < n; i++) esperado += A[i] * B[i];
    printf("  Resultado CORRETO (serial): p = %.0f\n\n", esperado);

    // ── a) COM RACE CONDITION — execute várias vezes para ver resultados variados ──
    sep("a) COM RACE CONDITION — resultado incorreto (demonstracao)");
    printf("  Explicacao: p += A[i]*B[i] nao e atomico.\n");
    printf("  Thread 0 le p=100, Thread 1 le p=100 (mesmo valor!).\n");
    printf("  Thread 0 escreve p=100+25=125. Thread 1 escreve p=100+36=136.\n");
    printf("  O resultado de Thread 0 (125) e PERDIDO — apenas 36 e somado.\n\n");
    float p_errado = 0;
    // Rodamos 3 vezes para mostrar que o resultado pode variar
    for (int tent = 0; tent < 3; tent++) {
        p_errado = 0;
        #pragma omp parallel for num_threads(4)
        for (int i = 0; i < n; i++) {
            p_errado += A[i] * B[i];  // RACE CONDITION! operação leia-modifique-escreva não é atômica
        }
        printf("  Tentativa %d (race condition): p = %.0f  %s\n",
               tent+1, p_errado, (p_errado == esperado) ? "(acerto por acaso)" : "(ERRADO!)");
    }

    // ── b) COM #pragma omp critical ── CORRETO PORÉM LENTO ──
    sep("b) COM critical — correto mas lento (critical executa n vezes)");
    float p_critical = 0;

    #pragma omp parallel for num_threads(4)
    for (int i = 0; i < n; i++) {
        float parcela = A[i] * B[i];   // cálculo pode ser paralelo

        #pragma omp critical  // apenas UMA thread executa este bloco por vez
        {
            p_critical += parcela;   // atualização atômica da variável pública
        }
        // PROBLEMA: critical é executado n=100 vezes → muito tempo em exclusão mútua
    }
    printf("  p_critical = %.0f  %s\n", p_critical,
           p_critical == esperado ? "(CORRETO)" : "(ERRADO!)");
    printf("  DESVANTAGEM: critical executa %d vezes — overhead maior que o ganho!\n", n);

    // ── c) COM paux + critical — BUEN DESEMPENHO ──
    sep("c) COM paux + critical — bom desempenho (critical executa apenas nt vezes)");
    float p_paux = 0;

    #pragma omp parallel num_threads(4)
    {
        float paux = 0;   // variável LOCAL desta thread (privada automaticamente)

        #pragma omp for
        for (int i = 0; i < n; i++) {
            paux += A[i] * B[i];   // acumula LOCALMENTE — sem race condition
        }

        // Após o for, cada thread contribui com sua parcela ao total global
        // Critical só é executado nt=4 vezes (1 por thread) — muito menos overhead!
        #pragma omp critical
        {
            p_paux += paux;
        }
    }
    printf("  p_paux = %.0f  %s\n", p_paux,
           p_paux == esperado ? "(CORRETO)" : "(ERRADO!)");
    printf("  VANTAGEM: critical executa apenas %d vezes (= num threads)!\n\n",
           omp_get_max_threads());
    printf("  RESUMO DA COMPARACAO:\n");
    printf("  a) Race condition: ERRADO\n");
    printf("  b) Critical em cada iteracao: correto, lento (critical executado n=%d vezes)\n", n);
    printf("  c) paux + critical final: correto, rapido (critical executado nt=%d vezes)\n",
           omp_get_max_threads());
}


// ─────────────────────────────────────────────────────────────────────────────
//  [EX06] PRODUTO ESCALAR — Cláusula reduction (solução ideal)
//
//  ORIGEM  : Exercício 2 do slide "Região Crítica"
//  ENUNCIADO: "Faça o mesmo do anterior, só que utilize agora a cláusula reduction"
//
//  COMO FUNCIONA reduction(+:p):
//    1. Compilador cria UMA cópia privada de "p" em cada thread (iniciada com 0)
//    2. Cada thread acumula na SUA cópia local
//    3. Ao final do for, as cópias são somadas na variável "p" global
//    É equivalente ao paux + critical, mas é automático e de altíssimo nível.
//
//  LIMITAÇÃO: reduction funciona apenas com operadores da linguagem:
//    + (soma), * (produto), - (subtração), & | ^ (bitwise), && || (lógico),
//    max, min (OpenMP 3.1+)
//  Para lógica mais complexa → use paux + critical manualmente.
// ─────────────────────────────────────────────────────────────────────────────
void ex06_produto_escalar_reduction() {
    titulo("[EX06] PRODUTO ESCALAR — reduction (solucao ideal do professor)");

    const int n = 100;
    static float A[n], B[n];

    for (int i = 0; i < n; i++) { A[i] = (float)i; B[i] = (float)i; }

    float p_red = 0;  // DEVE ser declarada FORA do parallel (para ser reduzida)

    // reduction(+:p_red):
    //   - Cria p_red_privada = 0 em cada thread
    //   - Cada thread acumula em sua cópia: p_red_privada += A[i]*B[i]
    //   - No final: p_red = soma de todas as p_red_privadas
    #pragma omp parallel for reduction(+:p_red) num_threads(4)
    for (int i = 0; i < n; i++) {
        p_red += A[i] * B[i];   // sem race condition — cada thread usa sua cópia
    }

    printf("  reduction(+:p): p = %.0f\n", p_red);
    printf("  Esperado: n*(n-1)*(2n-1)/6 = %.0f\n",
           (double)n * (n-1) * (2*n-1) / 6.0);
    printf("  VANTAGEM: nenhuma region critica explicita — compilador faz tudo!\n");
}


// ─────────────────────────────────────────────────────────────────────────────
//  [EX07] CÁLCULO DE PI — Versão com critical (exercício 1 do professor)
//
//  ORIGEM  : Slide "Exercícios — Paralelizar o Algoritmo de Aproximação de π"
//  ENUNCIADO: "Paralelizar o algoritmo de aproximação de Pi utilizando
//              Região Crítica"
//
//  TEORIA — Integração Numérica:
//    f(x) = 4 / (1 + x²)
//    ∫₀¹ f(x)dx = π
//    Aproximação por retângulos: π ≈ Σ f(xi) * Δx,  xi = i/n + 0.5*Δx
//    Δx = 1/n
//    Quanto maior n, maior a precisão (mais retângulos = área mais precisa)
//
//  O QUE PODE SER PARALELIZADO: cada retângulo é calculado de forma independente.
//  O QUE NÃO PODE: a acumulação em "pi" (variável pública) — critical ou reduction.
// ─────────────────────────────────────────────────────────────────────────────
void ex07_pi_critical() {
    titulo("[EX07] CALCULO DE PI — versao com #pragma omp critical");

    const int n = 1000000;  // 10^6 retângulos — quanto maior, mais preciso
    double pi    = 0.0;     // variável pública — precisa de proteção!
    double delta = 1.0 / n; // Δx = largura de cada retângulo

    double t0 = omp_get_wtime();

    #pragma omp parallel for num_threads(4)
    for (int i = 0; i < n; i++) {
        // xi = ponto central do i-ésimo retângulo no intervalo [0,1]
        double x  = (double)i / n + 0.5 * delta;

        // f(xi) = 4 / (1 + xi²)
        double fx = 4.0 / (1.0 + x * x);

        // Acumula a área do retângulo i — REGIÃO CRÍTICA
        // pi é pública → threads não podem atualizar ao mesmo tempo
        #pragma omp critical
        {
            pi += fx * delta;  // área do retângulo i = f(xi) * Δx
        }
    }

    double tempo = omp_get_wtime() - t0;
    printf("  n = %d retangulos\n", n);
    printf("  pi_critical = %.10f\n", pi);
    printf("  pi_real     = 3.1415926536...\n");
    printf("  erro        = %.2e\n", fabs(pi - M_PI));
    printf("  tempo       = %.4f s\n", tempo);
    printf("  DESVANTAGEM: critical executa n=%d vezes — muito lento!\n", n);
}


// ─────────────────────────────────────────────────────────────────────────────
//  [EX08] CÁLCULO DE PI — Versão com reduction + comparação de tempo
//
//  ORIGEM  : Slide "Exercícios — Paralelizar com Reduction"
//  ENUNCIADO: "Paralelizar o algoritmo de aproximação de Pi utilizando Reduction"
//
//  Inclui comparação de tempo: single vs critical vs reduction
//  e cálculo de speedup.
// ─────────────────────────────────────────────────────────────────────────────
void ex08_pi_reduction() {
    titulo("[EX08] CALCULO DE PI — reduction + comparacao de tempo e speedup");

    const int n = 100000000;  // 10^8: grande o suficiente para medir tempo real
    double pi    = 0.0;
    double delta = 1.0 / n;
    double t0, t1;
    double t_single, t_parallel;

    // ── Versão Serial (baseline) ──
    pi = 0.0;
    t0 = omp_get_wtime();
    for (int i = 0; i < n; i++) {
        double x  = (double)i / n + 0.5 * delta;
        double fx = 4.0 / (1.0 + x * x);
        pi += fx * delta;
    }
    t1 = omp_get_wtime();
    t_single = t1 - t0;
    printf("  SERIAL:    pi = %.10f  tempo = %.4f s\n", pi, t_single);

    // ── Versão Paralela com reduction ──
    pi = 0.0;
    t0 = omp_get_wtime();
    #pragma omp parallel for reduction(+:pi) num_threads(4)
    for (int i = 0; i < n; i++) {
        double x  = (double)i / n + 0.5 * delta;
        double fx = 4.0 / (1.0 + x * x);
        pi += fx * delta;   // sem race condition — reduction cuida disso
    }
    t1 = omp_get_wtime();
    t_parallel = t1 - t0;

    double speedup    = t_single / t_parallel;
    double eficiencia = speedup / 4.0;

    printf("  PARALELO:  pi = %.10f  tempo = %.4f s  (4 threads)\n", pi, t_parallel);
    printf("  Referencia:    3.1415926536...\n");
    printf("  erro       = %.2e\n", fabs(pi - M_PI));
    printf("\n  METRICAS DE DESEMPENHO:\n");
    printf("  Speedup    = T_serial / T_paralelo = %.4f / %.4f = %.2fx\n",
           t_single, t_parallel, speedup);
    printf("  Eficiencia = Speedup / num_threads = %.2f / 4 = %.1f%%\n",
           speedup, eficiencia * 100.0);
    printf("  (Speedup ideal seria 4.00x com 4 threads — overhead reduz um pouco)\n");
}


// ─────────────────────────────────────────────────────────────────────────────
//  [EX09] FATORAÇÃO LU — Sistema 3×3 do professor
//
//  ORIGEM  : Slides 45–61 do PDF, "Fatoração LU"
//  Sistema  : 2x1 + 3x2 - x3 = 5 | 2x1 - 3x2 + x3 = -1 | 4x1 + 4x2 - 3x3 = 3
//  Solução  : x1=1, x2=2, x3=3
//
//  DECOMPÕE A = L × U onde:
//    L = triangular INFERIOR  (diagonal = 1, acima = 0)
//    U = triangular SUPERIOR  (abaixo = 0)
//
//  4 FASES:
//  1. Inicialização de L e U       → NÃO precisa paralelizar (só atribuições)
//  2. Cálculo de L e U             → PARALELIZÁVEL (loops i e j independentes)
//  3. Forward substitution (Ly=b)  → NÃO PODE: y[i] depende de y[i-1]
//  4. Backward substitution (Ux=y) → NÃO PODE: x[i] depende de x[i+1]
//
//  DESEMPENHO DO PROFESSOR (sistema 5000×5000):
//    Versão single:  133 segundos
//    Versão paralela: 27 segundos (speedup ~4.9×)
// ─────────────────────────────────────────────────────────────────────────────
void ex09_fatoracao_lu_3x3() {
    titulo("[EX09] FATORACAO LU + Sistema Linear 3x3 (exemplo do professor)");

    const int N = 3;
    // Sistema do professor:
    double A[N][N] = {{2, 3,-1},
                      {2,-3, 1},
                      {4, 4,-3}};
    double b[N] = {5, -1, 3};
    double L[N][N], U[N][N];
    double y[N], x[N];

    // FASE 1: Inicialização — não precisa paralelizar
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            L[i][j] = (i == j) ? 1.0 : 0.0;   // L: 1 na diagonal, 0 acima
            U[i][j] = 0.0;
        }

    // FASE 2: Cálculo de L e U
    // Loop do PIVÔ (k): NÃO PODE ser paralelizado
    // → o cálculo de cada coluna k depende das anteriores (k-1, k-2...)
    for (int k = 0; k < N; k++) {

        // Linha pivô → U: PARALELIZÁVEL (cada j é independente)
        #pragma omp parallel for num_threads(4)
        for (int j = k; j < N; j++) {
            U[k][j] = A[k][j];
        }

        // Multiplicadores e atualização: PARALELIZÁVEL (cada i é independente)
        // Nota: numerador e denominador precisam de private (declarados fora)
        double numerador, denominador;
        #pragma omp parallel for num_threads(4) private(numerador, denominador)
        for (int i = k + 1; i < N; i++) {
            numerador   = A[i][k];
            denominador = A[k][k];
            L[i][k]     = numerador / denominador;   // L[i][k] = fator de eliminação

            // Atualiza linha i de A eliminando o elemento da coluna k
            for (int j = k; j < N; j++) {
                A[i][j] = A[i][j] - L[i][k] * A[k][j];
            }
        }
    }

    // FASE 3: Forward substitution — Ly = b
    // y[i] = (b[i] - Σ L[i][j]*y[j]) / L[i][i]   para j = 0..i-1
    // NÃO PODE ser paralelizado: y[1] depende de y[0], y[2] depende de y[0] e y[1]...
    for (int i = 0; i < N; i++) {
        y[i] = b[i];
        for (int j = 0; j < i; j++)
            y[i] -= L[i][j] * y[j];
        y[i] /= L[i][i];   // L[i][i] = 1 sempre
    }

    // FASE 4: Backward substitution — Ux = y
    // x[i] = (y[i] - Σ U[i][j]*x[j]) / U[i][i]   para j = i+1..N-1
    // NÃO PODE ser paralelizado: x[N-2] depende de x[N-1], etc.
    for (int i = N - 1; i >= 0; i--) {
        x[i] = y[i];
        for (int j = i + 1; j < N; j++)
            x[i] -= U[i][j] * x[j];
        x[i] /= U[i][i];
    }

    printf("  Sistema: 2x1+3x2-x3=5, 2x1-3x2+x3=-1, 4x1+4x2-3x3=3\n\n");
    printf("  Matriz L (triangular inferior):\n");
    for (int i = 0; i < N; i++) {
        printf("  [");
        for (int j = 0; j < N; j++) printf("%8.4f", L[i][j]);
        printf(" ]\n");
    }
    printf("  Matriz U (triangular superior):\n");
    for (int i = 0; i < N; i++) {
        printf("  [");
        for (int j = 0; j < N; j++) printf("%8.4f", U[i][j]);
        printf(" ]\n");
    }
    printf("\n  Forward substitution (Ly=b): y = [%.2f, %.2f, %.2f]\n", y[0],y[1],y[2]);
    printf("  Backward substitution (Ux=y):\n");
    for (int i = 0; i < N; i++)
        printf("    x[%d] = %.4f  %s\n", i, x[i],
               (fabs(x[i] - (i+1)) < 1e-6) ? "(correto)" : "(ERRO!)");
    printf("  Esperado: x[0]=1, x[1]=2, x[2]=3\n");

    printf("\n  O QUE PODE SER PARALELIZADO:\n");
    printf("  -> Loop j (linha do pivo para U): SIM — cada j e independente\n");
    printf("  -> Loop i (multiplicadores e atualizacao de A): SIM — cada linha e independente\n");
    printf("  O QUE NAO PODE:\n");
    printf("  -> Loop k (pivo): NAO — cada pivo depende da eliminacao anterior\n");
    printf("  -> Forward substitution: NAO — y[i] depende de y[i-1]\n");
    printf("  -> Backward substitution: NAO — x[i] depende de x[i+1]\n");
}


// ─────────────────────────────────────────────────────────────────────────────
//  [EX10] FATORAÇÃO LU — Sistema 6×6 com Medição de Tempo
//
//  ORIGEM  : Extensão do PDF — professor menciona sistema com 5000 incógnitas
//  PROPÓSITO: Evidenciar ganho de tempo em sistemas maiores
//
//  Sistema 6×6 com solução conhecida: x = [1, 2, 3, 4, 5, 6]
// ─────────────────────────────────────────────────────────────────────────────
void ex10_fatoracao_lu_6x6() {
    titulo("[EX10] FATORACAO LU — Sistema 6x6 com medicao de tempo");

    const int N = 6;

    // Matriz construída de forma que a solução seja x[i] = i+1
    double Aorig[N][N] = {
        { 2, 1, 0, 0, 0, 0},
        { 1, 3, 1, 0, 0, 0},
        { 0, 1, 4, 1, 0, 0},
        { 0, 0, 1, 5, 1, 0},
        { 0, 0, 0, 1, 6, 1},
        { 0, 0, 0, 0, 1, 7}
    };
    // Calcula b = A * [1,2,3,4,5,6]
    double b[N];
    for (int i = 0; i < N; i++) {
        b[i] = 0;
        for (int j = 0; j < N; j++)
            b[i] += Aorig[i][j] * (j + 1);
    }

    double A[N][N], L[N][N], U[N][N], y[N], x[N];
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            A[i][j] = Aorig[i][j];

    // Inicializa L e U
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            L[i][j] = (i == j) ? 1.0 : 0.0;
            U[i][j] = 0.0;
        }

    double t0 = omp_get_wtime();

    // Fatoração paralela
    for (int k = 0; k < N; k++) {
        #pragma omp parallel for num_threads(4)
        for (int j = k; j < N; j++) U[k][j] = A[k][j];

        double num, den;
        #pragma omp parallel for num_threads(4) private(num, den)
        for (int i = k + 1; i < N; i++) {
            num = A[i][k];  den = A[k][k];
            L[i][k] = num / den;
            for (int j = k; j < N; j++)
                A[i][j] -= L[i][k] * A[k][j];
        }
    }
    double t_lu = omp_get_wtime() - t0;

    // Forward substitution (serial)
    for (int i = 0; i < N; i++) {
        y[i] = b[i];
        for (int j = 0; j < i; j++) y[i] -= L[i][j] * y[j];
        y[i] /= L[i][i];
    }

    // Backward substitution (serial)
    for (int i = N - 1; i >= 0; i--) {
        x[i] = y[i];
        for (int j = i + 1; j < N; j++) x[i] -= U[i][j] * x[j];
        x[i] /= U[i][i];
    }

    printf("  Tempo fatoracao LU paralela (N=%d): %.6f s\n", N, t_lu);
    printf("  Solucao x (esperado: 1,2,3,4,5,6):\n");
    for (int i = 0; i < N; i++)
        printf("    x[%d] = %.4f  %s\n", i, x[i],
               (fabs(x[i] - (i+1)) < 1e-4) ? "(OK)" : "(ERRO!)");

    printf("\n  NOTA DO PROFESSOR (desempenho com 5000 incognitas):\n");
    printf("  Versao Single:   133 segundos\n");
    printf("  Versao Paralelo:  27 segundos\n");
    printf("  Speedup:          ~ 4.9x com 12 nucleos (Intel i9)\n");
}


// ─────────────────────────────────────────────────────────────────────────────
//  [EX11] MATRIZ INVERSA — Gauss-Jordan Paralelizado
//
//  ORIGEM  : Slides 77-80 do PDF, "Matriz Inversa – Método de Gauss-Jordan"
//
//  MÉTODO:
//    Monta a matriz aumentada [A | I] e aplica operações elementares por linhas
//    até A → I, ao mesmo tempo transformando I → A⁻¹
//    [A|I] → [I|A⁻¹]
//
//  FÓRMULA de cada operação:
//    L_i ← L_i - (A[i][k] / A[k][k]) * L_k   para todo i ≠ k
//    Essas operações sobre linhas diferentes são INDEPENDENTES → paralelizável!
//
//  PARALELIZÁVEL: loop de eliminação (linhas i ≠ k)
//  NÃO PARALELIZÁVEL: loop do pivô k (dependência estrutural)
// ─────────────────────────────────────────────────────────────────────────────

// Função auxiliar (fora da main) para contornar limitação de pragma dentro de lambda
// Inverte matriz N×N pelo método de Gauss-Jordan paralelo (N <= 6)
static void gauss_jordan_inverter(int n, double aug[][12]) {
    for (int k = 0; k < n; k++) {
        // Normaliza a linha do pivô
        double pivo = aug[k][k];
        for (int j = 0; j < 2*n; j++) aug[k][j] /= pivo;

        // Elimina coluna k em todas as outras linhas — PARALELIZÁVEL
        // Cada linha i ≠ k é calculada de forma completamente independente
        #pragma omp parallel for num_threads(4)
        for (int i = 0; i < n; i++) {
            if (i != k) {
                double fator = aug[i][k];
                for (int j = 0; j < 2*n; j++)
                    aug[i][j] -= fator * aug[k][j];
            }
        }
    }
}

void ex11_matriz_inversa_gauss_jordan() {
    titulo("[EX11] MATRIZ INVERSA — Gauss-Jordan Paralelizado");

    // Exemplo 2×2 do professor: A = [[1,2],[3,4]]
    // A⁻¹ esperada = [[-2, 1], [1.5, -0.5]]
    const int N = 2;
    double A[N][N] = {{1,2},{3,4}};

    // Monta a matriz aumentada [A | I] em formato fixo (max 6×12)
    double aug[12][12] = {};
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) aug[i][j] = A[i][j];
        aug[i][N + i] = 1.0;  // parte identidade
    }

    // Aplica Gauss-Jordan paralelizado
    gauss_jordan_inverter(N, aug);

    printf("  Matriz A = [[1,2],[3,4]]\n");
    printf("  A^(-1) calculada:\n");
    for (int i = 0; i < N; i++) {
        printf("  [");
        for (int j = 0; j < N; j++) printf("%8.4f", aug[i][N+j]);
        printf(" ]\n");
    }
    printf("  A^(-1) esperada: [[-2, 1], [1.5, -0.5]]\n");

    // Verificação: A * A^(-1) deve ser identidade
    printf("\n  Verificacao A * A^(-1) (deve ser [[1,0],[0,1]]):\n");
    for (int i = 0; i < N; i++) {
        printf("  [");
        for (int j = 0; j < N; j++) {
            double val = 0;
            for (int kk = 0; kk < N; kk++) val += A[i][kk] * aug[kk][N+j];
            printf("%8.4f", val);
        }
        printf(" ]\n");
    }

    printf("\n  O QUE PODE SER PARALELIZADO:\n");
    printf("  -> Loop i (eliminacao por linha, i != k): SIM — linhas independentes\n");
    printf("  O QUE NAO PODE:\n");
    printf("  -> Loop k (pivo): NAO — cada pivotamento depende do anterior\n");
}


// ─────────────────────────────────────────────────────────────────────────────
//  [EX12] DEMONSTRAÇÃO VISUAL DE RACE CONDITION
//
//  ORIGEM  : Extra — para entender profundamente o problema de região crítica
//  PROPÓSITO: Mostrar com números que sem proteção o resultado é incorreto.
//
//  Race condition ocorre quando:
//  1. Thread A lê valor V de variável compartilhada X
//  2. Thread B lê o mesmo valor V antes de A terminar
//  3. Thread A escreve X = V + aA
//  4. Thread B escreve X = V + bB  (sobrescreve a contribuição de A!)
//  Resultado: X = V + bB em vez de V + aA + bB → PERDA da contribuição de A
// ─────────────────────────────────────────────────────────────────────────────
void ex12_race_condition_visual() {
    titulo("[EX12] RACE CONDITION VISUAL — com e sem protecao");

    const int n = 10000;
    const int RUNS = 5;   // executa 5 vezes para mostrar variação

    printf("  Calculando soma de 1..%d = %ld\n\n", n, (long)n*(n+1)/2);

    // Sem proteção — execute várias vezes para ver a variação
    printf("  SEM protecao (race condition):\n");
    for (int r = 0; r < RUNS; r++) {
        long soma = 0;
        #pragma omp parallel for num_threads(8)
        for (int i = 1; i <= n; i++) {
            soma += i;   // RACE CONDITION: soma não é atômica em paralelo
        }
        printf("    Run %d: soma = %ld  %s\n", r+1, soma,
               soma == (long)n*(n+1)/2 ? "(correto por acaso)" : "(ERRADO!)");
    }

    // Com reduction — sempre correto
    printf("\n  COM reduction (sempre correto):\n");
    for (int r = 0; r < RUNS; r++) {
        long soma = 0;
        #pragma omp parallel for reduction(+:soma) num_threads(8)
        for (int i = 1; i <= n; i++) soma += i;
        printf("    Run %d: soma = %ld  %s\n", r+1, soma,
               soma == (long)n*(n+1)/2 ? "(CORRETO)" : "(ERRO!)");
    }

    // Com critical — sempre correto mas lento
    printf("\n  COM critical (correto, mas lento):\n");
    {
        long soma = 0;
        #pragma omp parallel for num_threads(8)
        for (int i = 1; i <= n; i++) {
            #pragma omp critical
            soma += i;
        }
        printf("    soma = %ld  %s\n", soma,
               soma == (long)n*(n+1)/2 ? "(CORRETO)" : "(ERRO!)");
    }
}


// ─────────────────────────────────────────────────────────────────────────────
//  [EX13] SOMA DE VETORES — Threads > Tamanho do Vetor
//
//  ORIGEM  : Observação do professor no slide "Soma de Vetores"
//  SITUAÇÃO: O número de threads pode ser MAIOR que o tamanho do vetor.
//
//  REGRA: Atribuir 1 posição por thread enquanto houver posições.
//         Threads com id >= n NÃO devem fazer nada.
//  PROTEÇÃO: if(inicio < n) antes do processamento
// ─────────────────────────────────────────────────────────────────────────────
void ex13_threads_maior_que_vetor() {
    titulo("[EX13] Threads > Tamanho do Vetor — protecao if(inicio < n)");

    const int n = 4;   // vetor pequeno
    float A[n], B[n], C[n];
    for (int i = 0; i < n; i++) { A[i] = (float)i; B[i] = (float)(i * 2); }

    printf("  Vetor n=%d, usando 8 threads (8 > 4)\n", n);

    #pragma omp parallel num_threads(8)
    {
        int id      = omp_get_thread_num();
        int nt      = omp_get_num_threads();
        int tamanho = n / nt;   // 4/8 = 0 — problema!
        int inicio  = id;       // com tamanho=0, cada thread pega 1 elemento
        int fim     = inicio;   // 1 elemento por thread

        // Se n >= nt, usar distribuição normal; senão, 1 por thread
        if (n >= nt) {
            tamanho = n / nt;
            inicio  = id * tamanho;
            fim     = inicio + tamanho - 1;
            if (id == nt - 1) fim = n - 1;
        }

        // PROTEÇÃO: thread sem trabalho válido não faz nada
        if (inicio < n) {
            for (int i = inicio; i <= fim && i < n; i++) {
                C[i] = A[i] + B[i];
                #pragma omp critical
                printf("  Thread %2d processa C[%d] = %.0f + %.0f = %.0f\n",
                       id, i, A[i], B[i], C[i]);
            }
        } else {
            #pragma omp critical
            printf("  Thread %2d: SEM TRABALHO (inicio=%d >= n=%d)\n", id, inicio, n);
        }
    }
}


// ─────────────────────────────────────────────────────────────────────────────
//  [EX14] MEDIÇÃO DE TEMPO COM omp_get_wtime() — Speedup e Eficiência
//
//  ORIGEM  : Extra — mas fundamental para provas de desempenho
//
//  FÓRMULAS FUNDAMENTAIS:
//    Speedup    = T_serial / T_paralelo
//    Eficiência = Speedup / num_threads   (100% = ideal, nunca alcançado)
//    Overhead   = T_paralelo * num_threads - T_serial
//
//  omp_get_wtime(): retorna o tempo atual em segundos (double, alta precisão)
//  Use SEMPRE double para armazenar o resultado.
// ─────────────────────────────────────────────────────────────────────────────
void ex14_medicao_tempo_speedup() {
    titulo("[EX14] omp_get_wtime() — Speedup e Eficiencia");

    const int n = 20000000;  // 2×10^7
    float soma;
    double t0, t1;

    // ── Versão Serial ──
    soma = 0;
    t0 = omp_get_wtime();
    for (int i = 0; i < n; i++) soma += (float)sqrt((double)i);
    t1 = omp_get_wtime();
    double t_serial = t1 - t0;

    printf("  n = %d\n\n", n);
    printf("  %-12s %-10s %-10s %-10s\n", "Config", "Tempo(s)", "Speedup", "Efic.(%)");
    linha();
    printf("  %-12s %-10.4f %-10.2f %-10.1f\n", "Serial", t_serial, 1.0, 100.0);

    int threads[] = {2, 4, 8};
    for (int nt : threads) {
        soma = 0;
        t0 = omp_get_wtime();
        #pragma omp parallel for reduction(+:soma) num_threads(nt)
        for (int i = 0; i < n; i++) soma += (float)sqrt((double)i);
        t1 = omp_get_wtime();
        double t_par = t1 - t0;
        double sp = t_serial / t_par;
        double ef = sp / nt * 100.0;
        printf("  %-12s %-10.4f %-10.2f %-10.1f\n",
               (std::string("Paralelo(") + std::to_string(nt) + "t)").c_str(),
               t_par, sp, ef);
    }

    printf("\n  Eficiencia < 100%% por causa de:\n");
    printf("  - Overhead de criacao/sincronizacao de threads\n");
    printf("  - Contencao de cache (false sharing)\n");
    printf("  - Fracao serial do codigo (Lei de Amdahl)\n");
}


// ─────────────────────────────────────────────────────────────────────────────
//  [EX15] LEI DE AMDAHL — Tabela Completa e Conclusões
//
//  ORIGEM  : Conteúdo teórico que o professor pode cobrar em prova escrita
//
//  Speedup_max(p) = 1 / (S + (1-S)/p)
//  onde:
//    S = fração SERIAL (não paralelizável) do código           [0, 1]
//    p = número de processadores/threads
//
//  Eficiência(p) = Speedup(p) / p
//
//  OBSERVAÇÕES IMPORTANTES:
//  - p → ∞: Speedup_max → 1/S    (teto absoluto!)
//  - S=0%:  Speedup = p           (crescimento linear — impossível na prática)
//  - S=50%: Speedup_max = 2×      (nunca passa de 2× mesmo com infinitas CPUs!)
//  - Na prática: S nunca é zero (inicialização, I/O, seções críticas, etc.)
// ─────────────────────────────────────────────────────────────────────────────
void ex15_lei_de_amdahl() {
    titulo("[EX15] LEI DE AMDAHL — Tabela completa de Speedup e Eficiencia");

    int ps[] = {1, 2, 4, 8, 16, 32, 64};
    double Ss[] = {0.00, 0.05, 0.10, 0.25, 0.50, 0.75};

    printf("\n  TABELA DE SPEEDUP: Speedup = 1 / (S + (1-S)/p)\n");
    printf("  %-8s", "S\\p");
    for (int p : ps) printf("%7d", p);
    printf("\n");
    linha();
    for (double S : Ss) {
        printf("  S=%3.0f%%  ", S * 100);
        for (int p : ps) {
            double sp = 1.0 / (S + (1.0 - S) / p);
            printf("%7.2f", sp);
        }
        printf("\n");
    }

    printf("\n  TABELA DE EFICIENCIA: Efic = Speedup / p\n");
    printf("  %-8s", "S\\p");
    for (int p : ps) printf("%7d", p);
    printf("\n");
    linha();
    for (double S : Ss) {
        printf("  S=%3.0f%%  ", S * 100);
        for (int p : ps) {
            double sp = 1.0 / (S + (1.0 - S) / p);
            double ef = sp / p;
            printf("%6.0f%%", ef * 100.0);
        }
        printf("\n");
    }

    printf("\n  CONCLUSOES IMPORTANTES PARA A PROVA:\n");
    printf("  1. S=0%%  : Speedup = p (ideal — nunca ocorre na pratica)\n");
    printf("  2. S=5%%  : Maximo speedup com 32 cores ~ %.1fx\n",
           1.0 / (0.05 + 0.95/32));
    printf("  3. S=50%% : Teto absoluto = 2x (mesmo com infinitas CPUs!)\n");
    printf("  4. S=75%% : Teto absoluto = 4x\n");
    printf("  5. Botar mais CPUs com alto S e desperdicio de recursos\n");
    printf("  6. Regra de ouro: reduza S antes de adicionar mais threads\n");
    printf("     (otimize a parte serial primeiro!)\n");
}


// ─────────────────────────────────────────────────────────────────────────────
//  [EX16] CLÁUSULA IF — Paraleliza Só Quando Vale a Pena
//
//  ORIGEM  : Slide "OpenMP – Cláusula IF da #pragma omp parallel"
//
//  TEORIA:
//    Criar threads tem um custo (overhead).
//    Para vetores/matrizes PEQUENOS, esse custo é MAIOR que o ganho.
//    A cláusula if(condição) evita esse desperdício:
//    - condição = true  → executa em paralelo (com threads)
//    - condição = false → executa serial (sem criar threads — zero overhead)
//
//  EXEMPLO DO PROFESSOR: if(n > 1000) → só paraleliza se vetor for grande.
// ─────────────────────────────────────────────────────────────────────────────
void ex16_clausula_if() {
    titulo("[EX16] CLAUSULA IF — paralelo condicional (so quando compensa)");

    float soma;
    double t0, t1;
    int tamanhos[] = {100, 1000, 10000, 1000000};

    printf("  %-12s %-12s %-12s %-12s %-8s\n",
           "Tamanho n", "T_serial(s)", "T_paralelo(s)", "if(n>1000)", "Speedup");
    linha();

    for (int n : tamanhos) {
        // Versão serial
        soma = 0;
        t0 = omp_get_wtime();
        for (int i = 0; i < n; i++) soma += (float)sqrt((double)i);
        t1 = omp_get_wtime();
        double ts = t1 - t0;

        // Versão paralela com IF: só paraleliza se n > 1000
        soma = 0;
        t0 = omp_get_wtime();
        #pragma omp parallel for reduction(+:soma) if(n > 1000) num_threads(4)
        for (int i = 0; i < n; i++) soma += (float)sqrt((double)i);
        t1 = omp_get_wtime();
        double tp = t1 - t0;

        bool paralelo_ativo = (n > 1000);
        printf("  %-12d %-12.6f %-12.6f %-12s %-8.2f\n",
               n, ts, tp, paralelo_ativo ? "PARALELO" : "serial",
               ts / tp);
    }

    printf("\n  OBSERVACAO: para n<=1000, o overhead de criar threads supera o ganho.\n");
    printf("  Com if(n>1000): programa automaticamente usa serial para dados pequenos.\n");
}


// ─────────────────────────────────────────────────────────────────────────────
//  [EX17] ESCOPO DE VARIÁVEIS — Bug silencioso vs solução correta
//
//  ORIGEM  : Slides "OpenMP – Escopo de Variáveis" e "Cláusula private"
//
//  REGRAS DO PROFESSOR:
//  ┌─────────────────────────────────────────────────────────────────────┐
//  │ Declarada FORA do parallel → GLOBAL (todas as threads compartilham)│
//  │ Declarada DENTRO do parallel → PRIVADA (exclusiva de cada thread)  │
//  │ private(x): torna x global em variável PRIVADA (não inicializada)  │
//  │ firstprivate(x): como private + inicializada com o valor original  │
//  │ shared(x): torna explícita que x é compartilhada (padrão para fora)│
//  └─────────────────────────────────────────────────────────────────────┘
//
//  VARIÁVEIS QUE DEVEM SER SEMPRE PRIVADAS:
//    id, tamanho, inicio, fim — variáveis de controle da thread
//    i, j, k — índices de loop
//    variáveis acumuladoras locais (paux, etc.)
// ─────────────────────────────────────────────────────────────────────────────
void ex17_escopo_variaveis() {
    titulo("[EX17] ESCOPO DE VARIAVEIS — global vs private (nao caia nessa!)");

    int x = 999;   // declarada FORA do parallel → global → compartilhada por todas

    sep("1. Sem private — x e COMPARTILHADA (leitura: ok; escrita: perigo)");
    #pragma omp parallel num_threads(4)
    {
        // x é a MESMA variável para todas as threads
        // Leitura concorrente: ok; Escrita concorrente: RACE CONDITION!
        int id = omp_get_thread_num();
        (void)id; // evita warning de unused
        #pragma omp critical
        printf("  Thread %d ve x = %d (mesma variavel global)\n",
               omp_get_thread_num(), x);
    }
    printf("  x apos parallel = %d (nao foi modificado neste bloco)\n", x);

    sep("2. Com private(x) — cada thread tem sua COPIA nao inicializada");
    #pragma omp parallel num_threads(4) private(x)
    {
        // x é uma cópia LOCAL desta thread — NÃO INICIALIZADA (valor lixo!)
        x = omp_get_thread_num() * 100;   // cada thread define sua cópia
        #pragma omp critical
        printf("  Thread %d: x_privado = %d\n", omp_get_thread_num(), x);
    }
    printf("  x original apos parallel = 999 (inalterado!)\n");

    sep("3. Com firstprivate(x) — copia INICIALIZADA com valor original");
    x = 999;
    #pragma omp parallel num_threads(4) firstprivate(x)
    {
        // x começa com o valor ORIGINAL (999) em cada thread
        int id = omp_get_thread_num();
        x = x + id;   // aplica modificação sobre o valor original
        #pragma omp critical
        printf("  Thread %d: x = 999 + %d = %d\n", id, id, x);
    }
    printf("  x original ainda = 999 (firstprivate nao propaga de volta)\n");

    sep("4. FORMA CORRETA: vars de controle declaradas DENTRO (automaticamente privadas)");
    #pragma omp parallel num_threads(4)
    {
        // id, nt, tamanho, inicio, fim → todas privadas automaticamente
        int id      = omp_get_thread_num();
        int nt      = omp_get_num_threads();
        int tamanho = 100 / nt;
        int inicio  = id * tamanho;
        int fim     = (id == nt-1) ? 99 : inicio + tamanho - 1;
        (void)fim;
        #pragma omp critical
        printf("  Thread %d: inicio=%d, tamanho=%d\n", id, inicio, tamanho);
    }
}


// ─────────────────────────────────────────────────────────────────────────────
//  [EX18] REDUCTION COM MAX E MIN — em um único parallel for
//
//  ORIGEM  : Extra — extensão natural de reduction
//
//  reduction(max:variavel): inicializa variável com o menor valor possível,
//                           cada thread mantém seu max local, ao final
//                           OpenMP combina com max global.
//  reduction(min:variavel): análogo com mínimo.
//
//  MÚLTIPLAS REDUCTIONS: podem ser encadeadas na mesma diretiva.
// ─────────────────────────────────────────────────────────────────────────────
void ex18_max_min_reduction() {
    titulo("[EX18] reduction(max) e reduction(min) no mesmo parallel for");

    const int n = 12;
    int v[n] = {5, -3, 12, 7, -8, 0, 4, 19, -1, 6, 11, 3};

    // Inicialização: max deve começar com o MENOR possível, min com o MAIOR
    int maximo = INT_MIN;   // -2147483648 → qualquer valor será maior
    int minimo = INT_MAX;   // +2147483647 → qualquer valor será menor
    long soma  = 0;

    // Calcular máximo, mínimo e soma em um único parallel for
    #pragma omp parallel for num_threads(4)  \
        reduction(max:maximo)                 \
        reduction(min:minimo)                 \
        reduction(+:soma)
    for (int i = 0; i < n; i++) {
        if (v[i] > maximo) maximo = v[i];
        if (v[i] < minimo) minimo = v[i];
        soma += v[i];
    }

    double media = (double)soma / n;
    int amplitude = maximo - minimo;

    printf("  Vetor: 5 -3 12 7 -8 0 4 19 -1 6 11 3\n\n");
    printf("  maximo    = %d    (esperado: 19)\n", maximo);
    printf("  minimo    = %d   (esperado: -8)\n", minimo);
    printf("  soma      = %ld   (esperado: 55)\n", soma);
    printf("  media     = %.2f  (esperado: 4.58)\n", media);
    printf("  amplitude = %d   (max-min = 19-(-8) = 27)\n", amplitude);
}


// ─────────────────────────────────────────────────────────────────────────────
//  [EX19] NORMA EUCLIDIANA — reduction(+:soma_quadrados) + sqrt serial
//
//  ORIGEM  : Extra — extensão do produto escalar
//
//  FÓRMULA : ||v|| = sqrt( v[0]² + v[1]² + ... + v[n-1]² )
//
//  ESTRATÉGIA:
//  1. Fase paralela: calcular a soma dos quadrados (reduction)
//  2. Fase serial: aplicar sqrt ao resultado (operação única)
//  A sqrt FINAL não precisa ser paralelizada — é só 1 operação.
// ─────────────────────────────────────────────────────────────────────────────
void ex19_norma_euclidiana() {
    titulo("[EX19] NORMA EUCLIDIANA — reduction(+:soma_quad) + sqrt serial");

    const int n = 6;
    float v[n] = {1, 2, 3, 4, 5, 6};

    float soma_quadrados = 0.0f;

    // FASE 1: Soma dos quadrados — paralelizável com reduction
    // Cada v[i]² é independente dos outros
    #pragma omp parallel for reduction(+:soma_quadrados) num_threads(4)
    for (int i = 0; i < n; i++) {
        soma_quadrados += v[i] * v[i];   // sem race condition
    }

    // FASE 2: sqrt — operação serial (1 único valor)
    // Não faz sentido paralelizar uma única operação aritmética
    float norma = (float)sqrt((double)soma_quadrados);

    printf("  v = [1, 2, 3, 4, 5, 6]\n");
    printf("  soma dos quadrados = 1+4+9+16+25+36 = %.0f\n", soma_quadrados);
    printf("  ||v|| = sqrt(%.0f) = %.6f\n", soma_quadrados, norma);
    printf("  Esperado: %.6f\n", (float)sqrt(91.0));
    printf("  [Fase paralela: soma dos quadrados]  [Fase serial: sqrt final]\n");
}


// ─────────────────────────────────────────────────────────────────────────────
//  [EX20] CONTAGEM DE ELEMENTOS COM CONDIÇÃO — reduction(+:contador)
//
//  ORIGEM  : Extra — padrão muito comum em provas
//
//  PADRÃO: contar quantos elementos de um vetor satisfazem uma condição.
//  Usar reduction(+:contador) para evitar race condition no contador.
// ─────────────────────────────────────────────────────────────────────────────
void ex20_contagem_condicional() {
    titulo("[EX20] CONTAGEM com condicao — reduction(+:contador)");

    const int n = 20;
    int v[n] = {3, 7, 1, 15, 8, 2, 11, 4, 9, 6, 14, 5, 12, -1, 0, 18, 3, 7, 2, 10};

    int count_positivos = 0;
    int count_pares     = 0;
    int count_maior10   = 0;

    // Conta múltiplas condições em um único parallel for
    #pragma omp parallel for num_threads(4) \
        reduction(+:count_positivos)         \
        reduction(+:count_pares)             \
        reduction(+:count_maior10)
    for (int i = 0; i < n; i++) {
        if (v[i] > 0)   count_positivos++;   // condição 1
        if (v[i] % 2 == 0) count_pares++;    // condição 2
        if (v[i] > 10)  count_maior10++;     // condição 3
    }

    printf("  Positivos: %d   (esperado: 18 — todos exceto -1 e 0)\n", count_positivos);
    printf("  Pares:     %d    (esperado: 9  — 8,2,4,6,14,12,0,18,2,10)\n", count_pares);
    printf("  > 10:      %d    (esperado: 5  — 15,11,14,12,18)\n", count_maior10);
}


// ─────────────────────────────────────────────────────────────────────────────
//  [EX21] firstprivate vs private — Diferença de Inicialização
//
//  ORIGEM  : Extra — pegadinha frequente em provas
//
//  private(x)      → cópia NÃO inicializada (valor lixo!)
//  firstprivate(x) → cópia inicializada com o valor de x no momento do fork
//
//  QUANDO USAR firstprivate:
//  - Quando a thread precisa começar com o valor atual da variável
//    e depois modificá-lo localmente sem afetar a global
//  - Ex: cada thread parte de um valor base e aplica uma transformação
// ─────────────────────────────────────────────────────────────────────────────
void ex21_firstprivate_vs_private() {
    titulo("[EX21] firstprivate vs private — diferenca de inicializacao");

    int base = 1000;   // valor base definido ANTES do parallel

    sep("private(base) — copia NAO inicializada (valor imprevisivel)");
    #pragma omp parallel num_threads(4) private(base)
    {
        // Nao use base antes de atribuir — valor e lixo de memoria!
        base = omp_get_thread_num();   // atribuicao obrigatoria antes de usar
        #pragma omp critical
        printf("  Thread %d: base (private) = %d  (sem ideia do que era antes)\n",
               omp_get_thread_num(), base);
    }
    printf("  base original = 1000 (nao alterado)\n");

    sep("firstprivate(base) — copia inicializada com 1000");
    base = 1000;
    #pragma omp parallel num_threads(4) firstprivate(base)
    {
        // Cada thread começa com base=1000 (cópia do valor no fork)
        base = base + omp_get_thread_num() * 100;  // modifica localmente
        #pragma omp critical
        printf("  Thread %d: base (firstprivate) = 1000 + %d*100 = %d\n",
               omp_get_thread_num(), omp_get_thread_num(), base);
    }
    printf("  base original = 1000 (firstprivate nao propaga de volta)\n");

    sep("USO TIPICO: cada thread calcula sobre um intervalo que comeca de 'acumulado'");
    float acumulado = 50.0f;  // valor base que cada thread deve conhecer
    float contribuicoes[4] = {};
    #pragma omp parallel num_threads(4) firstprivate(acumulado)
    {
        int id = omp_get_thread_num();
        // Cada thread parte de 50.0 e adiciona sua contribuição
        acumulado += id * 10.0f;   // 50, 60, 70, 80
        contribuicoes[id] = acumulado;
        #pragma omp critical
        printf("  Thread %d: acumulado = 50 + %d*10 = %.0f\n", id, id, acumulado);
    }
    printf("  Original acumulado = 50.0 (inalterado)\n");
}


// ─────────────────────────────────────────────────────────────────────────────
//  [EX22] MULTIPLICAÇÃO MATRIZ × VETOR — Paralelizada
//
//  ORIGEM  : Extra — extensão direta da multiplicação de matrizes
//
//  FÓRMULA: c[i] = Σ_{k=0}^{n-1} A[i][k] * b[k]
//
//  O QUE PODE: loop externo (i=linhas) — cada c[i] é independente.
//  O QUE NÃO PODE: loop de acumulação (k) — dependência na soma por linha.
// ─────────────────────────────────────────────────────────────────────────────
void ex22_multiplicacao_matriz_vetor() {
    titulo("[EX22] MULTIPLICACAO MATRIZ x VETOR — paralelizada");

    const int nl = 4, nc = 4;
    float A[nl][nc] = {{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16}};
    float b[nc]     = {1, 1, 1, 1};
    float c[nl]     = {0};

    // Variáveis declaradas FORA → precisam de private
    int i, k;

    // Loop externo (i): paralelizável — cada c[i] é independente
    // Loop de acumulação (k): NÃO paralelizar — dependência em c[i]
    #pragma omp parallel for num_threads(4) private(i, k)
    for (i = 0; i < nl; i++) {
        c[i] = 0;
        for (k = 0; k < nc; k++) {
            c[i] += A[i][k] * b[k];   // acumulação — loop k não pode ser paralelizado
        }
    }

    printf("  A = [[1..4],[5..8],[9..12],[13..16]],  b = [1,1,1,1]\n");
    printf("  Resultado c = A * b:\n");
    float esperados[] = {10, 26, 42, 58};
    for (int ii = 0; ii < nl; ii++)
        printf("    c[%d] = %.1f  %s\n", ii, c[ii],
               fabs(c[ii] - esperados[ii]) < 1e-3 ? "(OK)" : "(ERRO!)");
}


// ─────────────────────────────────────────────────────────────────────────────
//  [EX23] INICIALIZAÇÃO PARALELA COM collapse(2)
//
//  ORIGEM  : Extra — reforça o conceito de collapse e uso prático
//
//  SITUAÇÃO: Inicializar uma matriz grande em paralelo.
//  collapse(2): transforma os 2 loops em 1 com N*M iterações,
//               permitindo que OpenMP distribua mais uniformemente entre threads.
//
//  REQUISITO do collapse: os loops devem ser "perfeitamente aninhados"
//  (sem código entre eles — apenas a abertura do loop interno).
// ─────────────────────────────────────────────────────────────────────────────
void ex23_inicializacao_paralela_collapse() {
    titulo("[EX23] INICIALIZACAO PARALELA com collapse(2)");

    const int N = 5;
    float I[N][N];

    // Sem collapse: só paraleliza o for de i (N=5 iterações — sub-utilizado)
    sep("Sem collapse — so o for externo e paralelizado (5 iteracoes)");
    #pragma omp parallel for num_threads(4)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            I[i][j] = (i == j) ? 1.0f : 0.0f;
        }
    }
    printf("  I[2][2]=%.0f (esperado 1)  I[2][3]=%.0f (esperado 0)\n", I[2][2], I[2][3]);

    // Com collapse(2): paraleliza N*N=25 iterações — melhor distribuição
    sep("Com collapse(2) — N*N=25 iteracoes distribuidas (melhor para N pequeno)");
    #pragma omp parallel for collapse(2) num_threads(4)
    for (int i = 0; i < N; i++) {          // \__ tratados como 1 loop de 25 iter
        for (int j = 0; j < N; j++) {      // /   pelas 4 threads
            I[i][j] = (i == j) ? 1.0f : 0.0f;
        }
    }

    printf("  Matriz Identidade %dx%d:\n", N, N);
    for (int i = 0; i < N; i++) {
        printf("  [");
        for (int j = 0; j < N; j++) printf(" %.0f", I[i][j]);
        printf("  ]\n");
    }

    printf("\n  QUANDO USAR collapse(n):\n");
    printf("  - Loops externos com poucas iteracoes (< num_threads)\n");
    printf("  - Matrizes ou tensores onde o for externo sozinho sub-utiliza threads\n");
    printf("  - NUNCA com codigo entre os fors aninhados!\n");
}


// ─────────────────────────────────────────────────────────────────────────────
//  MAIN — Executa todos os exercícios em ordem
// ─────────────────────────────────────────────────────────────────────────────
int main() {
    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("  GABARITO COMPLETO — OpenMP\n");
    printf("  Prof. Dr. Andre Mendes Garcia\n");
    printf("  Nucleos logicos detectados: %d\n", omp_get_max_threads());
    printf("╚══════════════════════════════════════════════════════════╝\n");

    // ── Exercícios do PDF ─────────────────────────────────────────────
    ex01_soma_vetores_manual();
    ex02_soma_vetores_parallel_for();
    ex03_soma_matrizes();
    ex04_multiplicacao_matrizes();
    ex05_produto_escalar_critical();
    ex06_produto_escalar_reduction();
    ex07_pi_critical();
    ex08_pi_reduction();
    ex09_fatoracao_lu_3x3();
    ex10_fatoracao_lu_6x6();
    ex11_matriz_inversa_gauss_jordan();

    // ── Extras que podem cair na prova ───────────────────────────────
    ex12_race_condition_visual();
    ex13_threads_maior_que_vetor();
    ex14_medicao_tempo_speedup();
    ex15_lei_de_amdahl();
    ex16_clausula_if();
    ex17_escopo_variaveis();
    ex18_max_min_reduction();
    ex19_norma_euclidiana();
    ex20_contagem_condicional();
    ex21_firstprivate_vs_private();
    ex22_multiplicacao_matriz_vetor();
    ex23_inicializacao_paralela_collapse();

    printf("\n╔══════════════════════════════════════════════════════════╗\n");
    printf("  FIM — todos os exercicios executados com sucesso!\n");
    printf("╚══════════════════════════════════════════════════════════╝\n");
    return 0;
}


/*
╔══════════════════════════════════════════════════════════════════════════════╗
  COLA DA PROVA — REFERÊNCIA RÁPIDA COMPLETA
  (Tudo que você precisa para a P1 de Tópicos Avançados em Computação)
╚══════════════════════════════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1. SINTAXE DE TODAS AS DIRETIVAS E CLÁUSULAS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  #pragma omp parallel [cláusulas]
  { ... }                             → abre região paralela (fork/join)

  #pragma omp parallel num_threads(N) → usa N threads
  #pragma omp parallel if(expr)       → só paralelo se expr=true

  #pragma omp for                     → distribui for DENTRO de um parallel
  #pragma omp parallel for            → abre parallel E distribui for (merged)
  #pragma omp parallel for collapse(N)→ trata N loops aninhados como 1

  #pragma omp critical                → exclusão mútua (1 thread por vez)
  #pragma omp atomic                  → operação atômica simples (mais rápido)

  CLÁUSULAS DE ESCOPO:
    private(x)         → cópia não inicializada em cada thread
    firstprivate(x)    → cópia inicializada com valor antes do fork
    shared(x)          → explicitamente compartilhada (padrão para vars de fora)
    reduction(op:x)    → redução paralela; ops: + - * & | ^ && || max min

  FUNÇÕES OpenMP:
    omp_get_thread_num() → ID da thread atual (0..nt-1)
    omp_get_num_threads()→ total de threads no time atual
    omp_get_max_threads()→ máximo de threads disponíveis na máquina
    omp_get_wtime()      → tempo atual em segundos (use para medir tempo)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  2. AS 15 PRINCIPAIS PEGADINHAS (ERROS QUE O PROFESSOR PODE COBRAR)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  1. VARIÁVEL GLOBAL COM ESCRITA PARALELA → RACE CONDITION!
     ❌ float p = 0; #pragma omp parallel for → p += ...;
     ✅ Usar reduction(+:p) ou private + critical

  2. PRIVATE NÃO INICIALIZA A VARIÁVEL
     ❌ int x=5; #pragma omp parallel private(x) { x += 1; }  →  lixo!
     ✅ Usar firstprivate(x) quando precisar do valor original

  3. LOOP COM DEPENDÊNCIA SEQUENCIAL NÃO PODE SER PARALELIZADO
     ❌  for(i=1; i<n; i++) v[i] = v[i-1] + 1;  →  v[i] depende de v[i-1]
     ❌  Forward/Backward substitution (Fatoração LU)

  4. ÚLTIMO THREAD NÃO COBRINDO OS ELEMENTOS RESTANTES
     ❌  fim = inicio + tamanho - 1  (sem correção para o último)
     ✅  if(id == nt-1) { fim = n-1; tamanho = fim - inicio + 1; }

  5. THREADS > TAMANHO DO VETOR: thread inativa pode acessar índice fora
     ❌  forçar inicio e fim mesmo quando inicio >= n
     ✅  if(inicio < n) { processar; } else { pular; }

  6. PARALELIZAR O FOR INTERNO EM LOOPS ANINHADOS (collapse errado)
     ❌  for(i) { #pragma omp parallel for  for(j) {...} }  → nt fork/joins!
     ✅  #pragma omp parallel for  for(i) { for(j) {...} }  → 1 fork/join

  7. CRITICAL EXECUTADO N VEZES → DESEMPENHO PIOR QUE SERIAL
     ❌  #pragma omp for + #pragma omp critical { p += parcela; }
     ✅  paux privada + soma final em critical (executado nt vezes, não n)
     ✅ reduction(+:p) — melhor ainda!

  8. PARALLELIZAR OS DOIS FORS ANINHADOS SEM COLLAPSE (opção 3)
     ❌  #pragma omp parallel for  for(i){  #pragma omp parallel for  for(j){}}
     ✅  #pragma omp parallel for collapse(2)  for(i){ for(j){} }

  9. NÃO PASSAR -fopenmp AO COMPILADOR → pragmas ignorados silenciosamente
     ✅  g++ -fopenmp -O2 -o programa programa.cpp

 10. LOOP DO PIVÔ DA FATORAÇÃO LU: NUNCA PARALELIZAR o loop k externo
     ❌  #pragma omp parallel for  for(k=0;k<N;k++) { U[k][j]... L[i][k]... }
     ✅  Paralelizar APENAS os loops internos i e j

 11. SÃO DIFERENTES: parallel, parallel for, for (dentro de parallel)
     parallel             → só cria as threads (fork) — sem divisão automática
     parallel for         → cria E distribui iterações do for
     (dentro) for         → SÓ distribui iterações (threads já existem)
     ⚠️  Não use dois construtores de trabalho dentro de um mesmo parallel!

 12. COLLAPSE SÓ FUNCIONA COM FORS PERFEITAMENTE ANINHADOS
     ❌  for(i) { int x=f(i); for(j) {...} }  →  código entre os fors!
     ✅  for(i) { for(j) {...} }              →  nada entre os fors

 13. VARIÁVEIS DECLARADAS DENTRO DO FOR SÃO PRIVADAS (não precisa de private)
     for(int i ...) { int x; ... }  →  x é privada automaticamente

 14. omp_get_num_threads() FORA DO PARALLEL RETORNA 1 (só 1 thread serial)
     ✅  Chame sempre DENTRO da região parallel para obter o número real

 15. omp_get_wtime() deve ser armazenado em double (não float!)
     ❌  float t = omp_get_wtime();  →  perda de precisão!
     ✅  double t = omp_get_wtime();

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  3. QUANDO PARALELIZAR E QUANDO NÃO PARALELIZAR
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ✅ PODE PARALELIZAR:
  - Soma de vetores/matrizes (C[i] = A[i] + B[i]) — cada i é independente
  - Multiplicação de matrizes (loop de linhas i e colunas j)
  - Médio k da Fatoração LU (loops i e j dentro do pivô k)
  - Inicialização com collapse(2) — cada (i,j) é independente
  - Produto escalar com reduction — sem dependência entre pares
  - Cálculo de Pi (cada retângulo é independente)
  - Norma euclidiana (soma dos quadrados com reduction)
  - Contagem com condição (reduction de contadores)
  - Eliminação de Gauss-Jordan (linhas i ≠ k são independentes)

  ❌ NÃO PODE PARALELIZAR:
  - v[i] = v[i-1] + c    (dependência recorrente — i depende de i-1)
  - Forward substitution: y[i] = b[i] - Σ L[i][j]*y[j]  (y[i] depende de y[j<i])
  - Backward substitution: x[i] = y[i] - Σ U[i][j]*x[j] (x[i] depende de x[j>i])
  - Loop do pivô k da Fatoração LU (cada eliminação depende da anterior)
  - Atualização acumulada: soma += v[i] SEM proteção (race condition!)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  4. FÓRMULAS MATEMÁTICAS — MEMORIZE!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  DIVISÃO DO TRABALHO ENTRE THREADS:
    tamanho = n / nt                   (parcela base, inteiro)
    inicio  = id * tamanho
    fim     = inicio + tamanho - 1
    [última thread]: fim = n - 1; tamanho = fim - inicio + 1;

  SPEEDUP:
    Speedup = T_serial / T_paralelo    (ideal = num_threads)

  EFICIÊNCIA:
    Eficiência = Speedup / num_threads (ideal = 1.0 = 100%)

  LEI DE AMDAHL:
    Speedup_max(p) = 1 / (S + (1-S)/p)
    onde S = fração serial, p = número de processadores
    Limite: p → ∞ → Speedup_max = 1/S
    S=10%: max=10×; S=25%: max=4×; S=50%: max=2×; S=75%: max=1.33×

  CÁLCULO DE PI (integração numérica):
    f(x) = 4/(1+x²);  Δx = 1/n;  xi = i/n + 0.5*Δx
    π ≈ Σ f(xi)*Δx  para i=0..n-1

  PRODUTO ESCALAR:
    p = Σ A[i]*B[i]  para i=0..n-1

  MULTIPLICAÇÃO DE MATRIZES:
    C[i][j] = Σ A[i][k]*B[k][j]  para k=0..ncA-1

  NORMA EUCLIDIANA:
    ||v|| = sqrt(Σ v[i]²)

  FATORAÇÃO LU:
    A = L*U   (L=triangular inferior, diagonal=1; U=triangular superior)
    L[i][k] = A[i][k] / A[k][k]          (multiplicador da linha i)
    A[i][j] = A[i][j] - L[i][k]*A[k][j] (eliminação)
    Forward:  y[i] = (b[i] - Σ L[i][j]*y[j]) / L[i][i]   j<i
    Backward: x[i] = (y[i] - Σ U[i][j]*x[j]) / U[i][i]   j>i

  MATRIZ INVERSA (Gauss-Jordan):
    L_i ← L_i - (A[i][k]/A[k][k]) * L_k   para todo i ≠ k
    Após escalonamento total: [A|I] → [I|A⁻¹]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  5. DIFERENÇA ENTRE parallel, parallel for, e for (dentro de parallel)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  #pragma omp parallel               → FORK: todo o bloco é executado por TODAS
  { ... }                                as threads (N cópias do código)

  #pragma omp parallel for           → FORK + distribuição: o for é dividido
  for(i=0;i<n;i++){...}                 entre as threads automaticamente

  #pragma omp parallel               → FORK: entra na região paralela
  {
    #pragma omp for                  → distribuição (dentro de parallel já ativo)
    for(i=0;i<n;i++){...}
  }

  ⚠️ ERROS COMUNS:
  - Usar dois #pragma omp for dentro de um só #pragma omp parallel → erro
  - Usar #pragma omp for FORA de uma região parallel → ignorado/erro

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  6. REGRAS DE VARIÁVEIS GLOBAIS VS LOCAIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  int A[n], B[n], C[n];     // declaradas fora → GLOBAIS → COMPARTILHADAS
  float p = 0;              // declarada fora → GLOBAL → cuidado com escrita!

  #pragma omp parallel num_threads(4)
  {                         // ──────── DENTRO DO PARALLEL ────────────────
      int id = ...;         // LOCAL → PRIVADA automaticamente
      int x;                // LOCAL → PRIVADA (não inicializada)
      float paux = 0;       // LOCAL → PRIVADA (inicializada com 0)

      // Acessar A, B, C: ok (só leitura)  → sem race condition
      // Escrever em A[i] onde i não é compartilhado: ok → sem race condition
      // Escrever em p (global): race condition! → precisar de critical ou reduction
  }

  RESUMO:
  Operação       | Variável   | Precisa proteção?
  Leitura        | global     | NÃO (nunca causa race condition)
  Escrita única  | global[i]  | NÃO (se threads escrevem em índices distintos)
  Escrita acum.  | global     | SIM → reduction, critical, ou paux+critical
  Qualquer coisa | local      | NÃO (é privada por natureza)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  7. TEMPLATE PADRÃO — SOMA DE VETORES MÉTODO MANUAL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  #pragma omp parallel num_threads(4)
  {
      int id      = omp_get_thread_num();
      int nt      = omp_get_num_threads();
      int tamanho = n / nt;
      int inicio  = id * tamanho;
      int fim     = inicio + tamanho - 1;
      if (id == nt - 1) { fim = n-1; tamanho = fim-inicio+1; } // última thread
      if (inicio < n) {
          for (int i = inicio; i <= fim; i++) C[i] = A[i] + B[i];
      }
  }

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  8. TEMPLATE PADRÃO — PRODUTO ESCALAR (SOLUÇÃO IDEAL)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  float p = 0;
  #pragma omp parallel for reduction(+:p) num_threads(4)
  for (int i = 0; i < n; i++) p += A[i] * B[i];

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  9. TEMPLATE PADRÃO — CÁLCULO DE PI
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  double pi = 0.0, delta = 1.0/n;
  #pragma omp parallel for reduction(+:pi) num_threads(4)
  for (int i = 0; i < n; i++) {
      double x = (double)i/n + 0.5*delta;
      pi += 4.0/(1.0 + x*x) * delta;
  }

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  10. TEMPLATE PADRÃO — FATORAÇÃO LU (LOOP PARALELIZÁVEL)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  for (int k = 0; k < N; k++) {       // pivô: NÃO paralelizar!
      #pragma omp parallel for         //  ← paraleliza linha do pivô
      for (int j = k; j < N; j++) U[k][j] = A[k][j];

      double num, den;
      #pragma omp parallel for private(num, den)   // ← paraleliza eliminação
      for (int i = k+1; i < N; i++) {
          num = A[i][k];  den = A[k][k];
          L[i][k] = num/den;
          for (int j = k; j < N; j++)
              A[i][j] -= L[i][k] * A[k][j];
      }
  }

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
*/
