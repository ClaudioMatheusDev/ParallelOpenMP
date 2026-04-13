/*
================================================================================
  TÓPICOS AVANÇADOS EM COMPUTAÇÃO — Prof. Dr. André Mendes Garcia
  EXERCÍCIOS DO PDF + EXTRAS QUE PODEM CAIR NA PROVA
  Compilar: g++ -fopenmp -O2 -o exercicios exercicios_prova_openmp.cpp
================================================================================

  ÍNDICE DE EXERCÍCIOS E TÓPICOS:
  ─────────────────────────────────────────────────────────────────
  [EX01] Soma de vetores — Método manual (limites por thread)
  [EX02] Soma de vetores — Método implícito (#pragma omp parallel for)
  [EX03] Soma de matrizes 1000x1000 com A[i][j]=i, B[i][j]=j
  [EX04] Multiplicação de matrizes paralelizada (private i,j,k)
  [EX05] Produto escalar com #pragma omp critical
  [EX06] Produto escalar com reduction (A[i]=i, B[i]=i)
  [EX07] Cálculo de Pi — versão critical
  [EX08] Cálculo de Pi — versão reduction
  [EX09] Fatoração LU paralela + resolução de sistema linear
  [EX10] Matriz Inversa — Gauss-Jordan paralelizado
  ─────────────────────────────────────────────────────────────────
  EXTRAS (tópicos que podem cair):
  [EX11] Medir e comparar tempo single vs paralelo (speedup/eficiência)
  [EX12] Cláusula IF — paraleliza só quando vale a pena
  [EX13] Escopo de variáveis — demonstração do perigo de variável global
  [EX14] Soma de vetor com número de threads > tamanho do vetor
  [EX15] Máximo e mínimo de vetor com reduction
  [EX16] Multiplicação matriz por vetor (paralelizada)
  [EX17] Norma Euclidiana de vetor (paralelizada)
  [EX18] Contagem de elementos que satisfazem condição
  [EX19] Inicialização paralela de matriz identidade
  [EX20] Lei de Amdahl — cálculo teórico de speedup e eficiência
  ─────────────────────────────────────────────────────────────────
*/

#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <climits>   // INT_MIN, INT_MAX
#include <omp.h>

using namespace std;

// ─── utilitário ────────────────────────────────────────────
void titulo(const char* t) {
    cout << "\n╔══════════════════════════════════════════════╗" << endl;
    cout << "  " << t << endl;
    cout << "╚══════════════════════════════════════════════╝" << endl;
}
void subtitulo(const char* t) {
    cout << "\n  ── " << t << " ──" << endl;
}

/*
================================================================================
  [EX01] SOMA DE VETORES — Método Manual (limites por thread)
  (Exercício 1 do PDF — slide soma de vetores)

  ENUNCIADO DO PROFESSOR:
  "Fazer um programa em C/C++ que alimente dois vetores de 1000 elementos
  cada e efetue a soma destes num terceiro vetor de forma paralelizada"

  MÉTODO EXPLÍCITO: cada thread calcula seu próprio inicio e fim.
  REGRA:
    tamanho = n / nt
    inicio  = id * tamanho
    fim     = inicio + tamanho - 1
    Para a ÚLTIMA thread (id == nt-1): fim = n-1  (evita perder elementos)
================================================================================
*/
void ex01_soma_vetores_manual() {
    titulo("[EX01] SOMA DE VETORES — Metodo Manual (limites por thread)");

    const int n = 1000;
    float A[n], B[n], C[n];

    // Alimenta vetores com valores quaisquer (conforme slide do professor)
    for (int i = 0; i < n; i++) {
        A[i] = i * sin(i);
        B[i] = A[i] - cos(i * i);
    }

    // FORK — variáveis de controle declaradas DENTRO = privadas automaticamente
    #pragma omp parallel num_threads(4)
    {
        int id       = omp_get_thread_num();   // ID desta thread (privado)
        int nt       = omp_get_num_threads();  // total de threads (privado)
        int tamanho  = n / nt;                 // parcela base
        int inicio   = id * tamanho;           // onde esta thread começa
        int fim      = inicio + tamanho - 1;   // onde termina

        // CORREÇÃO para a última thread: garante que todos os elementos são cobertos
        // (necessário quando n não é divisível por nt, ex: 1000/3 = 333 com resto 1)
        if (id == nt - 1) {
            fim     = n - 1;
            tamanho = fim - inicio + 1;
        }

        // Proteção: se threads > n, thread sem trabalho não faz nada
        if (inicio < n) {
            for (int i = inicio; i <= fim; i++) {
                C[i] = A[i] + B[i];
            }
        }
    } // JOIN

    // Verifica alguns resultados
    cout << "  C[0]   = " << C[0]   << endl;
    cout << "  C[499] = " << C[499] << endl;
    cout << "  C[999] = " << C[999] << endl;
    cout << "  [OK] Soma de 1000 elementos realizada" << endl;
}

/*
================================================================================
  [EX02] SOMA DE VETORES — Método Implícito (#pragma omp parallel for)
  (Exercício 1 do PDF — versão com construtor de trabalho for)

  O compilador divide as iterações automaticamente.
  "i" é criada como variável PRIVADA em cada thread automaticamente.
================================================================================
*/
void ex02_soma_vetores_parallel_for() {
    titulo("[EX02] SOMA DE VETORES — Metodo Implicito (#pragma omp parallel for)");

    const int n = 1000;
    float A[n], B[n], C[n];

    for (int i = 0; i < n; i++) {
        A[i] = i * sin(i);
        B[i] = A[i] - cos(i * i);
    }

    // VERSÃO 1: parallel for combinado (mais compacta)
    #pragma omp parallel for num_threads(4)
    for (int i = 0; i < n; i++) {
        C[i] = A[i] + B[i];   // "i" é automaticamente privada
    }

    cout << "  Parallel for combinado: C[500] = " << C[500] << endl;

    // VERSÃO 2: parallel + for separados (equivalente)
    #pragma omp parallel num_threads(4)
    {
        int i; // declarada dentro = privada
        #pragma omp for
        for (i = 0; i < n; i++) {
            C[i] = A[i] + B[i];
        }
    }
    cout << "  Parallel + for separado: C[500] = " << C[500] << endl;
    cout << "  [OK] Ambas as formas produzem o mesmo resultado" << endl;
}

/*
================================================================================
  [EX03] SOMA DE MATRIZES 1000x1000
  (Exercício 2 do PDF — slide soma de matrizes)

  ENUNCIADO DO PROFESSOR:
  "Fazer um programa que alimente duas matrizes de 1000x1000 com os
   seguintes valores: A[i][j] = i; B[i][j] = j
   Em seguida faça a soma das matrizes A e B na matriz C de forma paralelizada"

  ANÁLISE DAS OPÇÕES (conforme professor):
  - Opção 1: paralelizar for externo (linhas) → OK
  - Opção 2: paralelizar for interno (colunas) → NÃO RECOMENDADO (alto overhead)
  - Opção 3: paralelizar ambos os for → NÃO RECOMENDADO (overhead de criação)
  - Opção 4: collapse(2) → SOLUÇÃO IDEAL (paraleliza os 2 fors como um só)
================================================================================
*/
void ex03_soma_matrizes() {
    titulo("[EX03] SOMA DE MATRIZES 1000x1000 (A[i][j]=i, B[i][j]=j)");

    // Usando alocação estática menor para demo (ajuste para 1000 se tiver RAM)
    const int nl = 100;  // use 1000 para o exercício real
    const int nc = 100;

    // Alocação dinâmica para matrizes grandes
    float** A = new float*[nl];
    float** B = new float*[nl];
    float** C = new float*[nl];
    for (int i = 0; i < nl; i++) {
        A[i] = new float[nc];
        B[i] = new float[nc];
        C[i] = new float[nc];
    }

    // Alimenta conforme o professor: A[i][j]=i, B[i][j]=j
    for (int i = 0; i < nl; i++)
        for (int j = 0; j < nc; j++) {
            A[i][j] = (float)i;
            B[i][j] = (float)j;
        }

    // ── OPÇÃO 1: Paraleliza só o for externo (linhas) ──
    subtitulo("Opcao 1 — for externo (linhas) — RECOMENDADO");
    #pragma omp parallel for num_threads(4)
    for (int i = 0; i < nl; i++) {
        for (int j = 0; j < nc; j++) {
            C[i][j] = A[i][j] + B[i][j];
        }
    }
    cout << "  C[5][3] = " << C[5][3] << " (esperado: 5+3=8)" << endl;

    // ── OPÇÃO 4: collapse(2) — SOLUÇÃO IDEAL do professor ──
    subtitulo("Opcao 4 — collapse(2) — SOLUCAO IDEAL");
    // collapse(2): trata os 2 fors como se fossem 1 único for de nl*nc iterações
    // sem custo computacional adicional (distribuição automática)
    #pragma omp parallel for collapse(2) num_threads(4)
    for (int i = 0; i < nl; i++) {
        for (int j = 0; j < nc; j++) {
            C[i][j] = A[i][j] + B[i][j];
        }
    }
    cout << "  C[5][3] = " << C[5][3] << " (esperado: 5+3=8)" << endl;
    cout << "  C[99][99] = " << C[99][99] << " (esperado: 99+99=198)" << endl;

    // Libera memória
    for (int i = 0; i < nl; i++) { delete[] A[i]; delete[] B[i]; delete[] C[i]; }
    delete[] A; delete[] B; delete[] C;
}

/*
================================================================================
  [EX04] MULTIPLICAÇÃO DE MATRIZES — Paralelizada com private(i,j,k)
  (Exercício do PDF — slide multiplicação de matrizes)

  FÓRMULA: C[i][j] = Σ A[i][k] * B[k][j]   k=0..ncA-1

  PONTO CRÍTICO DO PROFESSOR:
  - Variáveis i, j, k declaradas ANTES do parallel → precisam de private()
  - Loop externo (i) e médio (j) são paralelizáveis
  - Loop de acumulação (k) NÃO pode ser paralelizado diretamente
================================================================================
*/
void ex04_multiplicacao_matrizes() {
    titulo("[EX04] MULTIPLICACAO DE MATRIZES — private(i,j,k)");

    // A[la][ca] * B[ca][cb] = C[la][cb]
    const int la = 3, ca = 3, cb = 3;
    float A[la][ca] = {{1,2,3},{4,5,6},{7,8,9}};
    float B[ca][cb] = {{9,8,7},{6,5,4},{3,2,1}};
    float C[la][cb];

    // Inicializa C
    for (int i = 0; i < la; i++)
        for (int j = 0; j < cb; j++)
            C[i][j] = 0.0;

    // Variáveis declaradas FORA do parallel → globais → precisam de private!
    int i, j, k;

    // #pragma omp parallel for: inicia a região paralela E aplica o construtor for
    // (em uma única pragma — conforme o professor ensina no slide da Fatoração LU)
    #pragma omp parallel for num_threads(4) private(i, j, k)
    for (i = 0; i < la; i++) {           // loop externo — paralelizado
        for (j = 0; j < cb; j++) {       // loop médio — privado por thread
            C[i][j] = 0;
            for (k = 0; k < ca; k++) {   // acumulação — NÃO paralelizar
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    cout << "  Resultado C = A * B:" << endl;
    for (int ii = 0; ii < la; ii++) {
        cout << "  ";
        for (int jj = 0; jj < cb; jj++)
            printf("%8.1f", C[ii][jj]);
        cout << endl;
    }
    // Linha 0 esperada: [30, 24, 18]
    // Linha 1 esperada: [84, 69, 54]
    // Linha 2 esperada: [138,114, 90]
}

/*
================================================================================
  [EX05] PRODUTO ESCALAR — com #pragma omp critical
  (Exercício 1 do slide "Região Crítica")

  ENUNCIADO DO PROFESSOR:
  "Implementar um programa que alimente dois vetores A e B com os
   respectivos valores de suas posições: A[i]=i; B[i]=i
   Em seguida calcule o produto escalar utilizando a diretiva para
   proteção da região crítica."

  PRODUTO ESCALAR: p = A[0]*B[0] + A[1]*B[1] + ... + A[n-1]*B[n-1]

  PROBLEMA: a variável "p" é pública (global). Sem proteção,
  threads podem sobrescrever o valor umas das outras → RACE CONDITION.

  SOLUÇÃO CRÍTICA: #pragma omp critical garante que só 1 thread
  atualiza "p" por vez. Correto, mas LENTO (critical executado n vezes).
================================================================================
*/
void ex05_produto_escalar_critical() {
    titulo("[EX05] PRODUTO ESCALAR — com #pragma omp critical");

    const int n = 8;  // use valores maiores para o exercício real
    float A[n], B[n];

    // A[i] = i;  B[i] = i   (conforme enunciado do professor)
    for (int i = 0; i < n; i++) {
        A[i] = (float)i;
        B[i] = (float)i;
    }

    // Resultado esperado: 0*0 + 1*1 + 2*2 + ... + 7*7 = 0+1+4+9+16+25+36+49 = 140

    // ── Versão Single (referência) ──
    float p_single = 0;
    for (int i = 0; i < n; i++)
        p_single += A[i] * B[i];
    printf("  Single:   p = %.1f\n", p_single);

    // ── Versão com PROBLEMA (race condition — NÃO FAZER ISSO) ──
    // float p_errado = 0;
    // #pragma omp parallel for num_threads(4)
    // for (int i = 0; i < n; i++)
    //     p_errado += A[i] * B[i];  // RACE CONDITION! várias threads escrevem em p ao mesmo tempo
    // printf("  Errado (race condition): p = %.1f  ← valor incorreto!\n", p_errado);

    // ── Versão CORRIGIDA com #pragma omp critical ──
    // p deve ser GLOBAL (declarada fora) para ser acumulada
    float p_critical = 0;

    #pragma omp parallel for num_threads(4)
    for (int i = 0; i < n; i++) {
        float parcela = A[i] * B[i];  // cálculo pode ser paralelo

        #pragma omp critical  // só 1 thread entra aqui por vez
        {
            p_critical += parcela;  // atualização serial
        }
        // Desvantagem: esta região crítica é executada n vezes (= dim do vetor)
        // Pior que single pois o overhead da região crítica supera o ganho
    }
    printf("  Critical: p = %.1f (correto, mas lento — critical executa %d vezes)\n",
           p_critical, n);
}

/*
================================================================================
  [EX06] PRODUTO ESCALAR — com reduction
  (Exercício 2 do slide "Região Crítica" — solução ideal do professor)

  ENUNCIADO DO PROFESSOR:
  "Faça o mesmo do anterior, só que utilize agora a cláusula reduction"

  TAMBÉM INCLUI a versão com variável auxiliar paux (bom desempenho)
  que o professor ensina como solução intermediária.
================================================================================
*/
void ex06_produto_escalar_reduction() {
    titulo("[EX06] PRODUTO ESCALAR — reduction (solucao ideal) + paux");

    const int n = 8;
    float A[n], B[n];

    for (int i = 0; i < n; i++) { A[i] = (float)i; B[i] = (float)i; }

    // ── Versão paux (variável auxiliar privada — BOM DESEMPENHO) ──
    // O professor ensina isto como solução intermediária melhor que critical:
    // cada thread acumula em sua paux privada, critical só executado nt vezes
    float p_paux = 0;
    #pragma omp parallel num_threads(4)
    {
        float paux = 0;  // variável LOCAL desta thread (privada automaticamente)
        #pragma omp for
        for (int i = 0; i < n; i++) {
            paux += A[i] * B[i];  // acumula na variável privada desta thread
        }
        // Critical executado apenas UMA vez por thread (= nt vezes, não n vezes)
        #pragma omp critical
        {
            p_paux += paux;  // soma as parcelas de cada thread
        }
    }
    printf("  Paux:      p = %.1f (critical executa apenas %d vezes = num threads)\n",
           p_paux, omp_get_max_threads());

    // ── Versão reduction (MELHOR — solução do professor) ──
    // reduction(+:p) faz o compilador:
    //   1. Criar uma cópia privada de p em cada thread (iniciada com 0)
    //   2. Cada thread acumula em sua cópia
    //   3. No final, soma todas as cópias na variável global p
    float p_red = 0;
    #pragma omp parallel for reduction(+:p_red) num_threads(4)
    for (int i = 0; i < n; i++) {
        p_red += A[i] * B[i];
    }
    printf("  Reduction: p = %.1f (MELHOR — altissimo nivel, zero critica explicita)\n",
           p_red);
    // Esperado: 140
}

/*
================================================================================
  [EX07] CÁLCULO DE PI — com #pragma omp critical
  (Exercício 1 do slide "Aproximação de Pi" — exercício do professor)

  ENUNCIADO DO PROFESSOR:
  "Paralelizar o algoritmo de aproximação de Pi utilizando Região Crítica"

  FÓRMULA:
    f(x) = 4 / (1 + x²)
    ∫₀¹ f(x)dx = π
    Δx = 1/n
    xi = i/n + 0.5*Δx   (centro do i-ésimo retângulo)
    π ≈ Σ f(xi)*Δx   para i = 0..n-1
================================================================================
*/
void ex07_pi_critical() {
    titulo("[EX07] CALCULO DE PI — com critical");

    const int n = 1000000;   // quanto maior, mais preciso
    double pi    = 0.0;
    double delta = 1.0 / n;  // Δx = (1-0)/n

    #pragma omp parallel for num_threads(4)
    for (int i = 0; i < n; i++) {
        double x  = (double)i / n + 0.5 * delta;  // xi = centro do retângulo i
        double fx = 4.0 / (1.0 + x * x);          // f(xi) = 4/(1+x²)

        #pragma omp critical   // protege a variável pública pi
        {
            pi += fx * delta;  // acumula a área do retângulo i
        }
    }

    printf("  n=%d  pi_critical = %.10f\n", n, pi);
    printf("  pi real           = 3.1415926536...\n");
    printf("  erro              = %.2e\n", fabs(pi - M_PI));
}

/*
================================================================================
  [EX08] CÁLCULO DE PI — com reduction
  (Exercício 2 do slide "Aproximação de Pi" — solução ideal do professor)

  ENUNCIADO DO PROFESSOR:
  "Paralelizar o algoritmo de aproximação de Pi utilizando Reduction"

  Também mostra comparativo de tempo single vs paralelo
================================================================================
*/
void ex08_pi_reduction() {
    titulo("[EX08] CALCULO DE PI — com reduction (solucao ideal)");

    const int n = 100000000; // 10^8 para ver diferença de tempo
    double pi    = 0.0;
    double delta = 1.0 / n;

    double t_inicio, t_fim;

    // ── Versão single ──
    pi = 0.0;
    t_inicio = omp_get_wtime();  // omp_get_wtime(): cronômetro OpenMP
    for (int i = 0; i < n; i++) {
        double x  = (double)i / n + 0.5 * delta;
        double fx = 4.0 / (1.0 + x * x);
        pi += fx * delta;
    }
    t_fim = omp_get_wtime();
    double t_single = t_fim - t_inicio;
    printf("  Single:    pi = %.10f  tempo = %.4fs\n", pi, t_single);

    // ── Versão paralela com reduction ──
    pi = 0.0;
    t_inicio = omp_get_wtime();
    #pragma omp parallel for reduction(+:pi) num_threads(4)
    for (int i = 0; i < n; i++) {
        double x  = (double)i / n + 0.5 * delta;
        double fx = 4.0 / (1.0 + x * x);
        pi += fx * delta;
    }
    t_fim = omp_get_wtime();
    double t_paralelo = t_fim - t_inicio;
    printf("  Reduction: pi = %.10f  tempo = %.4fs\n", pi, t_paralelo);
    printf("  Speedup = %.2fx  (single/paralelo)\n", t_single / t_paralelo);
    printf("  Esperado: 3.1415926536...\n");
}

/*
================================================================================
  [EX09] FATORAÇÃO LU — Sistema Linear 3x3
  (Conteúdo dos slides 45-61 do PDF)

  Decompõe A = L * U onde:
  - L: triangular INFERIOR (diagonal = 1, acima = 0)
  - U: triangular SUPERIOR (abaixo = 0)

  PARALELIZÁVEL: loop i e j do cálculo de L[i][k] e atualização de A
  NÃO PARALELIZÁVEL:
  - Loop externo k (pivô): dependência entre colunas
  - Forward substitution: y[i] depende de y[i-1]
  - Backward substitution: x[i] depende de x[i+1]

  DESEMPENHO DO PROFESSOR:
  - Sistema com 5000 incógnitas: Single=133s → Paralelo=27s (speedup ~4.9x)
================================================================================
*/
void ex09_fatoracao_lu() {
    titulo("[EX09] FATORACAO LU + resolucao de sistema linear");

    // Sistema do professor (slides):
    // 2x1 + 3x2 - x3  =  5
    // 2x1 - 3x2 + x3  = -1
    // 4x1 + 4x2 - 3x3 =  3
    // Solução: x1=1, x2=2, x3=3

    const int N = 3;
    double A[N][N] = {{2, 3,-1},
                      {2,-3, 1},
                      {4, 4,-3}};
    double b[N]  = {5, -1, 3};
    double L[N][N], U[N][N];
    double y[N], x[N];

    // ── Inicializa L e U ──────────────────────────────────────
    // (não precisa paralelizar — apenas atribuições escalares)
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            L[i][j] = (i == j) ? 1.0 : 0.0;  // diagonal de L = 1
            U[i][j] = 0.0;
        }

    // ── Cálculo de L e U ── LOOP DO PIVÔ: NÃO PARALELIZÁVEL ──
    for (int k = 0; k < N; k++) {

        // Linha pivô → vai para U (loop j: PARALELIZÁVEL)
        #pragma omp parallel for num_threads(4)
        for (int j = k; j < N; j++) {
            U[k][j] = A[k][j];
        }

        // Multiplicadores e atualização de A (loop i: PARALELIZÁVEL)
        // NOTA: em uma mesma #pragma inicia-se a região paralela E o construtor for
        // (conforme o professor ensina nos slides da Fatoração LU)
        double numerador, denominador;
        #pragma omp parallel for num_threads(4) private(numerador, denominador)
        for (int i = k + 1; i < N; i++) {
            numerador   = A[i][k];
            denominador = A[k][k];
            L[i][k]     = numerador / denominador;   // multiplicador → coluna k de L

            for (int j = k; j < N; j++) {
                A[i][j] = A[i][j] - L[i][k] * A[k][j];  // elimina coluna k
            }
        }
    }

    // ── Forward substitution: Ly = b ── NÃO PARALELIZÁVEL ──
    // y[i] = (b[i] - Σ L[i][j]*y[j]) / L[i][i]   para j < i
    for (int i = 0; i < N; i++) {
        y[i] = b[i];
        for (int j = 0; j < i; j++)
            y[i] -= L[i][j] * y[j];
        y[i] /= L[i][i];
    }

    // ── Backward substitution: Ux = y ── NÃO PARALELIZÁVEL ──
    // x[i] = (y[i] - Σ U[i][j]*x[j]) / U[i][i]   para j > i
    for (int i = N - 1; i >= 0; i--) {
        x[i] = y[i];
        for (int j = i + 1; j < N; j++)
            x[i] -= U[i][j] * x[j];
        x[i] /= U[i][i];
    }

    cout << "  Matriz L (triangular inferior):" << endl;
    for (int i = 0; i < N; i++) {
        cout << "  ";
        for (int j = 0; j < N; j++) printf("%8.3f", L[i][j]);
        cout << endl;
    }
    cout << "  Matriz U (triangular superior):" << endl;
    for (int i = 0; i < N; i++) {
        cout << "  ";
        for (int j = 0; j < N; j++) printf("%8.3f", U[i][j]);
        cout << endl;
    }
    cout << "  Solucao do sistema:" << endl;
    for (int i = 0; i < N; i++)
        printf("    x[%d] = %.3f\n", i, x[i]);
    printf("  Esperado: x[0]=1, x[1]=2, x[2]=3\n");
}

/*
================================================================================
  [EX10] MATRIZ INVERSA — Gauss-Jordan Paralelizado
  (Conteúdo final do PDF — slides 77-80)

  MÉTODO:
  - Monta a matriz aumentada [A | I]
  - Aplica operações elementares por linhas até A → I
  - Resultado: a parte direita vira A⁻¹ → [I | A⁻¹]

  PARALELIZÁVEL:
  - Loop de eliminação (linhas i ≠ k): independentes entre si

  FÓRMULA da operação:
  L[i] = L[i] - (A[i][k] / A[k][k]) * L[k]   para i ≠ k
================================================================================
*/
void ex10_matriz_inversa_gauss_jordan() {
    titulo("[EX10] MATRIZ INVERSA — Gauss-Jordan Paralelizado");

    // Exemplo do professor: A = [[1,2],[3,4]]
    // A⁻¹ esperada = [[-2, 1],[1.5, -0.5]]
    const int N = 2;
    double A[N][N] = {{1, 2},
                      {3, 4}};

    // Monta a matriz aumentada [A | I]
    double aug[N][2*N];
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)   aug[i][j]   = A[i][j];
        for (int j = 0; j < N; j++)   aug[i][N+j] = (i == j) ? 1.0 : 0.0;
    }

    cout << "  Matriz aumentada [A | I] inicial:" << endl;
    for (int i = 0; i < N; i++) {
        cout << "  ";
        for (int j = 0; j < 2*N; j++) printf("%8.3f", aug[i][j]);
        cout << endl;
    }

    // Gauss-Jordan: pivô k percorre cada coluna
    for (int k = 0; k < N; k++) {
        // Normaliza linha do pivô (divide pelo elemento diagonal)
        double pivo = aug[k][k];
        for (int j = 0; j < 2*N; j++) aug[k][j] /= pivo;

        // Elimina coluna k em todas as outras linhas — PARALELIZÁVEL
        #pragma omp parallel for num_threads(4)
        for (int i = 0; i < N; i++) {
            if (i != k) {
                double fator = aug[i][k];
                for (int j = 0; j < 2*N; j++) {
                    aug[i][j] -= fator * aug[k][j];
                }
            }
        }
    }

    cout << "  Matriz Inversa A^(-1) (parte direita da aumentada):" << endl;
    for (int i = 0; i < N; i++) {
        cout << "  ";
        for (int j = 0; j < N; j++) printf("%8.3f", aug[i][N+j]);
        cout << endl;
    }
    printf("  Esperado: [[-2, 1], [1.5, -0.5]]\n");

    // Verificação: A * A⁻¹ deve dar identidade
    cout << "  Verificacao A * A^(-1) (deve ser identidade):" << endl;
    for (int i = 0; i < N; i++) {
        cout << "  ";
        for (int j = 0; j < N; j++) {
            double val = 0;
            for (int kk = 0; kk < N; kk++)
                val += A[i][kk] * aug[kk][N+j];
            printf("%8.3f", val);
        }
        cout << endl;
    }
}

/*
================================================================================
  [EX11] SPEEDUP E EFICIÊNCIA — Comparação de Tempo
  (Conceitos que o professor pode cobrar na prova teórica)

  SPEEDUP     = T_serial / T_paralelo
  EFICIÊNCIA  = Speedup / número_de_threads   (ideal = 1.0 = 100%)
  LEI DE AMDAHL: Speedup_max = 1 / (S + (1-S)/p)
    onde S = fração serial, p = número de processadores
================================================================================
*/
void ex11_speedup_eficiencia() {
    titulo("[EX11] SPEEDUP, EFICIENCIA — comparacao single vs paralelo");

    const int n = 50000000;
    float soma;
    double t_single, t_paralelo;

    // ── Versão Single ──
    soma = 0;
    double t0 = omp_get_wtime();
    for (int i = 0; i < n; i++) soma += sqrt((float)i);
    t_single = omp_get_wtime() - t0;

    // ── Versão Paralela ──
    soma = 0;
    t0 = omp_get_wtime();
    #pragma omp parallel for reduction(+:soma) num_threads(4)
    for (int i = 0; i < n; i++) soma += sqrt((float)i);
    t_paralelo = omp_get_wtime() - t0;

    int nt = 4;
    double speedup    = t_single / t_paralelo;
    double eficiencia = speedup / nt;

    printf("  Tempo single:    %.4f s\n", t_single);
    printf("  Tempo paralelo:  %.4f s  (%d threads)\n", t_paralelo, nt);
    printf("  Speedup:         %.2fx  (ideal = %.1fx)\n", speedup, (double)nt);
    printf("  Eficiencia:      %.1f%%  (ideal = 100%%)\n", eficiencia * 100.0);

    // ── Lei de Amdahl ──
    cout << "\n  LEI DE AMDAHL — Speedup maximo teorico:" << endl;
    cout << "  Speedup_max = 1 / (S + (1-S)/p)" << endl;
    cout << "  onde S = fracao serial (nao paralelizavel)" << endl;
    double S_vals[] = {0.05, 0.10, 0.20, 0.50};
    int    p_vals[] = {2, 4, 8, 16};
    for (double S : S_vals) {
        printf("  S=%.0f%%:  ", S*100);
        for (int p : p_vals) {
            double sp = 1.0 / (S + (1.0-S)/p);
            printf("p=%2d→%.2fx  ", p, sp);
        }
        cout << endl;
    }
}

/*
================================================================================
  [EX12] CLÁUSULA IF — paraleliza só quando vale a pena
  (Slide do professor "Cláusula IF da #pragma omp parallel")

  Premissa: só compensa criar threads quando a carga é GRANDE.
  if(condicao): se false, executa serial (sem overhead de criar threads)
================================================================================
*/
void ex12_clausula_if() {
    titulo("[EX12] CLAUSULA IF — paralelo condicional");

    int n_pequeno = 100;
    int n_grande  = 100000;
    float soma = 0;

    // Só paraleliza se n > 1000 (overhead não justifica para vetores pequenos)
    cout << "  n=" << n_grande << " (n > 1000 = TRUE  → paralelo ATIVO)" << endl;
    soma = 0;
    #pragma omp parallel for reduction(+:soma) if(n_grande > 1000) num_threads(4)
    for (int i = 0; i < n_grande; i++) soma += i;
    printf("  soma = %.0f\n", soma);

    cout << "\n  n=" << n_pequeno << " (n > 1000 = FALSE → execucao SERIAL)" << endl;
    soma = 0;
    #pragma omp parallel for reduction(+:soma) if(n_pequeno > 1000) num_threads(4)
    for (int i = 0; i < n_pequeno; i++) soma += i;
    printf("  soma = %.0f\n", soma);
}

/*
================================================================================
  [EX13] ESCOPO DE VARIÁVEIS — Demonstração do perigo de variável global
  (Slides "Escopo de Variáveis" e "Cláusula private")

  REGRA DO PROFESSOR:
  - Fora do parallel → global (compartilhada por todas as threads)
  - Dentro do parallel → privada (exclusiva de cada thread)
  - Para tornar uma variável global em privada: use private(var)
  - variáveis id, inicio, fim, tamanho → SEMPRE declarar dentro ou usar private
================================================================================
*/
void ex13_escopo_variaveis() {
    titulo("[EX13] ESCOPO DE VARIAVEIS — global vs private");

    int x_global = 999;  // declarada FORA → compartilhada por todas as threads

    subtitulo("SEM private — x_global e compartilhada (pode causar bug)");
    #pragma omp parallel num_threads(4)
    {
        // x_global é a MESMA variável para todas — threads interferem entre si
        // Para simples leitura tudo bem; para escrita pode gerar resultados incorretos
        #pragma omp critical
        printf("  Thread %d ve x_global = %d\n", omp_get_thread_num(), x_global);
    }

    subtitulo("COM private(x_global) — cada thread tem sua copia");
    #pragma omp parallel num_threads(4) private(x_global)
    {
        // Agora x_global é uma cópia LOCAL de cada thread (NÃO inicializada!)
        x_global = omp_get_thread_num() * 100;  // cada thread modifica a sua
        #pragma omp critical
        printf("  Thread %d: x_privado = %d (nao afeta o original)\n",
               omp_get_thread_num(), x_global);
    }
    printf("  x_global original apos o parallel = %d (inalterado)\n", x_global);

    subtitulo("CORRETO: variaveis de controle declaradas DENTRO do parallel");
    #pragma omp parallel num_threads(4)
    {
        // id, nt, inicio, fim, tamanho → privadas automaticamente (declaradas dentro)
        int id      = omp_get_thread_num();
        int nt      = omp_get_num_threads();
        int tamanho = 100 / nt;
        int inicio  = id * tamanho;
        int fim     = (id == nt-1) ? 99 : inicio + tamanho - 1;
        #pragma omp critical
        printf("  Thread %d: inicio=%d fim=%d tamanho=%d\n", id, inicio, fim, fim-inicio+1);
    }
}

/*
================================================================================
  [EX14] SOMA DE VETOR — threads > tamanho do vetor
  (Observação do professor no slide "Soma de Vetores")

  Quando o número de threads for MAIOR que o tamanho do vetor:
  - Cada thread processa 1 posição
  - Threads com id >= n NÃO fazem nada (if de proteção)
================================================================================
*/
void ex14_threads_maior_que_vetor() {
    titulo("[EX14] Threads > tamanho do vetor (protecao if(inicio < n))");

    const int n = 4;    // vetor pequeno
    float A[n], B[n], C[n];
    for (int i = 0; i < n; i++) { A[i] = i; B[i] = i * 2; }

    // 8 threads para um vetor de 4 — threads 4,5,6,7 não fazem nada
    #pragma omp parallel num_threads(8)
    {
        int id      = omp_get_thread_num();
        int nt      = omp_get_num_threads();
        int tamanho = n / nt;            // 4/8 = 0 (problema!)
        int inicio  = id * tamanho;      // 0 para todas se tamanho=0

        // Com tamanho truncado, distribuímos 1 elemento por thread
        if (tamanho == 0) {
            // 1 elemento por thread até esgotar o vetor
            inicio  = id;
            tamanho = 1;
        }
        int fim = inicio + tamanho - 1;

        // PROTEÇÃO: threads com id >= n não processam nada
        if (inicio < n) {
            C[inicio] = A[inicio] + B[inicio];
            #pragma omp critical
            printf("  Thread %d processa posicao %d: C[%d]=%.0f\n",
                   id, inicio, inicio, C[inicio]);
        } else {
            #pragma omp critical
            printf("  Thread %d: sem trabalho (inicio=%d >= n=%d)\n", id, inicio, n);
        }
    }
}

/*
================================================================================
  [EX15] MÁXIMO E MÍNIMO DE VETOR — com reduction
  (Extensão natural dos exercícios de reduction)

  reduction(max:variavel) e reduction(min:variavel)
================================================================================
*/
void ex15_max_min_reduction() {
    titulo("[EX15] MAXIMO E MINIMO com reduction");

    const int n = 10;
    int v[n] = {3, 7, 1, 15, 8, 2, 11, 4, 9, 6};

    int maximo = INT_MIN;  // inicia com o menor int possível
    int minimo = INT_MAX;  // inicia com o maior int possível
    int soma   = 0;

    #pragma omp parallel for num_threads(4)       \
        reduction(max:maximo)                       \
        reduction(min:minimo)                       \
        reduction(+:soma)
    for (int i = 0; i < n; i++) {
        if (v[i] > maximo) maximo = v[i];
        if (v[i] < minimo) minimo = v[i];
        soma += v[i];
    }

    double media = (double)soma / n;
    printf("  Vetor: 3 7 1 15 8 2 11 4 9 6\n");
    printf("  Maximo = %d (esperado: 15)\n", maximo);
    printf("  Minimo = %d (esperado: 1)\n",  minimo);
    printf("  Soma   = %d (esperado: 66)\n", soma);
    printf("  Media  = %.1f\n", media);
}

/*
================================================================================
  [EX16] MULTIPLICAÇÃO MATRIZ × VETOR — paralelizada
  (Extensão direta da multiplicação de matrizes)

  C[i] = Σ A[i][k] * b[k]   para k = 0..n-1
  Loop externo (i) é paralelizável; loop k acumula → não paralelizar
================================================================================
*/
void ex16_matriz_vetor() {
    titulo("[EX16] MULTIPLICACAO MATRIZ x VETOR");

    const int nl = 4, nc = 4;
    float A[nl][nc] = {{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16}};
    float b[nc]     = {1, 1, 1, 1};
    float c[nl]     = {0};

    int i, k;  // declaradas fora → precisam de private
    #pragma omp parallel for num_threads(4) private(i, k)
    for (i = 0; i < nl; i++) {
        c[i] = 0;
        for (k = 0; k < nc; k++) {
            c[i] += A[i][k] * b[k];  // acumulação — loop k NÃO é paralelizado
        }
    }

    cout << "  Resultado c = A * b (b=[1,1,1,1]):" << endl;
    for (int ii = 0; ii < nl; ii++)
        printf("    c[%d] = %.1f\n", ii, c[ii]);
    // Esperado: 10, 26, 42, 58
}

/*
================================================================================
  [EX17] NORMA EUCLIDIANA — com reduction
  (Extensão do produto escalar — redução com sqrt)

  ||v|| = sqrt( v[0]² + v[1]² + ... + v[n-1]² )
================================================================================
*/
void ex17_norma_euclidiana() {
    titulo("[EX17] NORMA EUCLIDIANA com reduction");

    const int n = 6;
    float v[n] = {1, 2, 3, 4, 5, 6};

    float soma_quadrados = 0.0f;

    // Fase 1: soma dos quadrados (paralelizável com reduction)
    #pragma omp parallel for reduction(+:soma_quadrados) num_threads(4)
    for (int i = 0; i < n; i++) {
        soma_quadrados += v[i] * v[i];
    }

    // Fase 2: raiz quadrada (operação serial — 1 valor)
    float norma = sqrt(soma_quadrados);

    printf("  ||v|| = sqrt(1+4+9+16+25+36) = sqrt(91) = %.6f\n", norma);
    printf("  Esperado: %.6f\n", sqrt(91.0f));
}

/*
================================================================================
  [EX18] CONTAGEM DE ELEMENTOS — com reduction
  (Padrão comum em provas: contar quantos elementos satisfazem condição)
================================================================================
*/
void ex18_contagem_condicional() {
    titulo("[EX18] CONTAGEM de elementos com condicao — reduction");

    const int n = 10;
    int v[n] = {3, 7, 1, 15, 8, 2, 11, 4, 9, 6};

    int count_pares = 0;
    int count_maior7 = 0;

    #pragma omp parallel for num_threads(4)         \
        reduction(+:count_pares)                     \
        reduction(+:count_maior7)
    for (int i = 0; i < n; i++) {
        if (v[i] % 2 == 0) count_pares++;    // é par?
        if (v[i] > 7)      count_maior7++;   // é maior que 7?
    }

    printf("  Vetor: 3 7 1 15 8 2 11 4 9 6\n");
    printf("  Pares:        %d (esperado: 3 — {8, 2, 4})\n", count_pares);
    printf("  Maiores que 7:%d (esperado: 4 — {15, 8, 11, 9})\n", count_maior7);
}

/*
================================================================================
  [EX19] INICIALIZAÇÃO PARALELA DE MATRIZ IDENTIDADE
  (Padrão comum: inicializar estruturas de dados em paralelo)
================================================================================
*/
void ex19_matriz_identidade_paralela() {
    titulo("[EX19] INICIALIZACAO PARALELA — Matriz Identidade");

    const int N = 5;
    float I[N][N];

    // Inicializa a matriz identidade em paralelo com collapse(2)
    #pragma omp parallel for collapse(2) num_threads(4)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            I[i][j] = (i == j) ? 1.0f : 0.0f;
        }
    }

    cout << "  Matriz Identidade " << N << "x" << N << ":" << endl;
    for (int i = 0; i < N; i++) {
        cout << "  ";
        for (int j = 0; j < N; j++) printf("%4.0f", I[i][j]);
        cout << endl;
    }
}

/*
================================================================================
  [EX20] LEI DE AMDAHL — demonstração teórica completa
  (Conteúdo teórico que cai em provas conceituais)

  Speedup(p) = 1 / (S + (1-S)/p)
  onde:
    S = fração SERIAL do código (não paralelizável)
    p = número de processadores/threads
  
  Eficiência(p) = Speedup(p) / p

  Conclusão do professor: quanto maior a fração serial, menor o ganho.
  Com S=50%, nunca passará de 2x de speedup, independente do número de CPUs.
================================================================================
*/
void ex20_lei_de_amdahl() {
    titulo("[EX20] LEI DE AMDAHL — tabela de speedup e eficiencia");

    printf("\n  %-8s", "S\\p");
    int ps[] = {1, 2, 4, 8, 16, 32, 64};
    for (int p : ps) printf("%6d", p);
    cout << endl;
    printf("  %s\n", string(50, '-').c_str());

    double Ss[] = {0.0, 0.05, 0.10, 0.25, 0.50, 0.75};
    for (double S : Ss) {
        printf("  S=%.0f%%  ", S*100);
        for (int p : ps) {
            double sp = 1.0 / (S + (1.0 - S) / p);
            printf("%6.2f", sp);
        }
        cout << endl;
    }

    cout << "\n  CONCLUSOES:" << endl;
    cout << "  - S=0% (100% paralelo): speedup = p (ideal, mas impossivel na pratica)" << endl;
    cout << "  - S=5%: com 32 cores, speedup maximo ~10x" << endl;
    cout << "  - S=50%: com infinitos cores, speedup maximo = 2x (limite fisico!)" << endl;
    cout << "  - Eficiencia = Speedup / p  (ideal = 1.0 = 100%)" << endl;

    printf("\n  Exemplo pratico (p=4 threads):\n");
    for (double S : Ss) {
        double sp = 1.0 / (S + (1.0 - S) / 4);
        double ef = sp / 4.0;
        printf("    S=%.0f%%: speedup=%.2fx  eficiencia=%.1f%%\n", S*100, sp, ef*100);
    }
}

/*
================================================================================
  MAIN — executa todos os exercícios em ordem
================================================================================
*/
int main() {
    cout << "╔═══════════════════════════════════════════════════════╗" << endl;
    cout << "   EXERCICIOS PROVA — OpenMP" << endl;
    cout << "   Prof. Dr. Andre Mendes Garcia" << endl;
    cout << "   Nucleos disponíveis: " << omp_get_max_threads() << endl;
    cout << "╚═══════════════════════════════════════════════════════╝" << endl;

    // ── Exercícios do PDF ──────────────────────────────────
    ex01_soma_vetores_manual();
    ex02_soma_vetores_parallel_for();
    ex03_soma_matrizes();
    ex04_multiplicacao_matrizes();
    ex05_produto_escalar_critical();
    ex06_produto_escalar_reduction();
    ex07_pi_critical();
    ex08_pi_reduction();
    ex09_fatoracao_lu();
    ex10_matriz_inversa_gauss_jordan();

    // ── Extras que podem cair na prova ────────────────────
    ex11_speedup_eficiencia();
    ex12_clausula_if();
    ex13_escopo_variaveis();
    ex14_threads_maior_que_vetor();
    ex15_max_min_reduction();
    ex16_matriz_vetor();
    ex17_norma_euclidiana();
    ex18_contagem_condicional();
    ex19_matriz_identidade_paralela();
    ex20_lei_de_amdahl();

    cout << "\n[FIM — todos os exercicios executados]" << endl;
    return 0;
}

/*
================================================================================
  COLA RÁPIDA — RESUMO DAS PEGADINHAS DE PROVA
================================================================================

  1. VARIÁVEL GLOBAL DENTRO DE PARALLEL → RACE CONDITION!
     Sempre declare dentro do bloco parallel OU use private(var)

  2. #pragma omp parallel for NÃO substitui um parallel com construtor for
     se dentro desse parallel houver outro construtor (ex: sections)
     → Nesse caso use: #pragma omp parallel { #pragma omp for ... }

  3. LOOP COM DEPENDÊNCIA → não paralelize!
     Ex: v[i] = v[i-1] + 1   ← v[i] depende de v[i-1]
     Ex: forward/backward substitution da Fatoração LU

  4. collapse(n): só com fors perfeitamente aninhados (sem código entre eles)

  5. reduction funciona com: + - * & | ^ && || max min
     Para lógica mais complexa → use paux + critical

  6. omp_get_wtime() mede tempo em segundos (use para calcular speedup)

  7. ÚLTIMA THREAD: sempre trate o caso de divisão não exata
     if(id == nt-1) { fim = n-1; tamanho = fim - inicio + 1; }

  8. private(var): cópia NÃO inicializada.
     firstprivate(var): cópia inicializada com o valor original.

  9. Fatoração LU: só o loop INTERNO (i e j) é paralelizável.
     O loop do pivô (k externo) tem dependência → nunca paralelizar.

 10. Speedup = T_serial / T_paralelo
     Eficiência = Speedup / num_threads   (ideal: 1.0 = 100%)
================================================================================
*/
