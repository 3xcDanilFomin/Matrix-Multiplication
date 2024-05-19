#include <iostream>
#include <mkl.h>
#include <omp.h>
#include <math.h>
#include <immintrin.h>

using namespace std;

#define PREFETCH_DISTANCE 64
#define BLOCK_SIZE 32
#define N 4096

void printLine() {
    for (int i = 0; i < 100; i++)
    {
        cout << '-';
    }
    cout << endl;
}

double randNumber(double min, double max)
{
    double number = (double)rand() / RAND_MAX;
    return min + number * (max - min);
}

void fillMatrix(double* matrix, double min = 0, double max = 0) {

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            matrix[i * N + j] =  randNumber(min, max);
        }
    }
}

void linearMatrixMultiplication(const double* matrixA, const double* matrixB, double* matrixC)
{
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
        {
            matrixC[i * N + j] = 0;

            for (int k = 0; k < N; k++) {
                matrixC[i * N + j] += matrixA[i * N + k] * matrixB[k * N + j];
            }
        }
}

double scalarProduct(double* matrixA, double* matrixB) {
    double s[4] = { 0.0, 0.0, 0.0, 0.0 };
    int n8 = (N / 8) * 8;

    __m256d sum1, sum2, sum3, sum4;
    sum1 = _mm256_setzero_pd();
    sum2 = _mm256_setzero_pd();
    sum3 = _mm256_setzero_pd();
    sum4 = _mm256_setzero_pd();

    for (int i = 0; i < n8; i += 32) {
        _mm_prefetch((char*)&matrixA[i + PREFETCH_DISTANCE], _MM_HINT_T0);
        _mm_prefetch((char*)&matrixB[i + PREFETCH_DISTANCE], _MM_HINT_T0);

        __m256d a1, a2, a3, a4, a5, a6, a7, a8;
        __m256d b1, b2, b3, b4, b5, b6, b7, b8;

        a1 = _mm256_load_pd(&matrixA[i]);
        a2 = _mm256_load_pd(&matrixA[i + 4]);
        a3 = _mm256_load_pd(&matrixA[i + 8]);
        a4 = _mm256_load_pd(&matrixA[i + 12]);
        a5 = _mm256_load_pd(&matrixA[i + 16]);
        a6 = _mm256_load_pd(&matrixA[i + 20]);
        a7 = _mm256_load_pd(&matrixA[i + 24]);
        a8 = _mm256_load_pd(&matrixA[i + 28]);

        b1 = _mm256_load_pd(&matrixB[i]);
        b2 = _mm256_load_pd(&matrixB[i + 4]);
        b3 = _mm256_load_pd(&matrixB[i + 8]);
        b4 = _mm256_load_pd(&matrixB[i + 12]);
        b5 = _mm256_load_pd(&matrixB[i + 16]);
        b6 = _mm256_load_pd(&matrixB[i + 20]);
        b7 = _mm256_load_pd(&matrixB[i + 24]);
        b8 = _mm256_load_pd(&matrixB[i + 28]);

        sum1 = _mm256_fmadd_pd(a1, b1, sum1);
        sum2 = _mm256_fmadd_pd(a2, b2, sum2);
        sum3 = _mm256_fmadd_pd(a3, b3, sum3);
        sum4 = _mm256_fmadd_pd(a4, b4, sum4);
        sum1 = _mm256_fmadd_pd(a5, b5, sum1);
        sum2 = _mm256_fmadd_pd(a6, b6, sum2);
        sum3 = _mm256_fmadd_pd(a7, b7, sum3);
        sum4 = _mm256_fmadd_pd(a8, b8, sum4);
    }

    __m256d total_sum = _mm256_add_pd(_mm256_add_pd(sum1, sum2), _mm256_add_pd(sum3, sum4));
    _mm256_storeu_pd(s, total_sum);

    for (int i = n8; i < N; i += 4) {
        __m256d a = _mm256_loadu_pd(&matrixA[i]);
        __m256d b = _mm256_loadu_pd(&matrixB[i]);
        __m256d result = _mm256_mul_pd(a, b);
        double* res_ptr = (double*)&result;
        s[0] += res_ptr[0];
        s[1] += res_ptr[1];
        s[2] += res_ptr[2];
        s[3] += res_ptr[3];
    }

    return (s[0] + s[1]) + (s[2] + s[3]);
}

double* matrixTransposition(const double* matrix, double* transposedMatrix)
{
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            transposedMatrix[j * N + i] = matrix[i * N + j];
        }
    return transposedMatrix;
}

void optimizedMatrixMultiplication(double* matrixA, double* matrixB, double* matrixC)
{
#pragma omp parallel for num_threads(32)
    for (int ii = 0; ii < N; ii +=BLOCK_SIZE)
    {
        for (int jj = 0; jj < N; jj += BLOCK_SIZE) {
            for (int i = 0; i < BLOCK_SIZE; ++i) {
                for (int j = 0; j < BLOCK_SIZE; ++j)
                {
                    matrixC[(ii + i) * N + (jj + j)] = scalarProduct(&matrixA[(ii + i) * N], &matrixB[(jj + j) * N]);
                }
            }
        }
    }
    
}

bool matrixEqual(double* matrixA, double* matrixB, float eps = 1.e-3)
{
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            if (abs(matrixA[i * N + j] - matrixB[i * N + j]) > eps)
                return false;
    return true;
}

int main()
{
    setlocale(LC_ALL, "RU");
    srand(time(NULL));

    double* matrixA = new double[N * N];
    double* matrixB = new double[N * N];
    double* matrixBT = new double[N * N];

    double* matrixC1 = new double[N * N];
    double* matrixC2 = new double[N * N];
    double* matrixC3 = new double[N * N];

    long long c = 2 * pow(N, 3);

    clock_t start, end;

    fillMatrix(matrixA, 0, 1);
    fillMatrix(matrixB, 0, 1);
    fillMatrix(matrixC1);
    fillMatrix(matrixC2);
    fillMatrix(matrixC3);

    cout << "Сложность алгоритма составляет: " << c << endl;
    printLine();
    cout << "Элементы заполненных матриц: " << endl;
    cout << "matrixA[20] = " << matrixA[20] << endl;
    cout << "matrixB[20] = " << matrixB[20] << endl;
    printLine();

    start = clock();
    //linearMatrixMultiplication(matrixA, matrixB, matrixC1);
    end = clock();

    double elapsed_secs;
    elapsed_secs = double(end - start) / CLOCKS_PER_SEC;

    double p1 = c / elapsed_secs * pow(10, -6);

    cout << "Время умножения матриц с помощью линейного алгоритма: " << elapsed_secs << " секунд" << endl;
    cout << "Производительность алгоритма составляет: " << p1 << " MFlops\n";
    cout << "matrixC1[20] = " << matrixC1[20] << endl;
    printLine();

    double alpha = 1.0;
    double beta = 0.0;

    start = clock();
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, alpha, matrixA, N, matrixB, N, beta, matrixC2, N);
    end = clock();

    elapsed_secs = double(end - start) / CLOCKS_PER_SEC;

    double p2 = c / elapsed_secs * pow(10, -6);

    cout << "Время умножения матриц с помощью функции cblas_dgemm из библиотеки MKL: " << elapsed_secs << " секунд" << endl;
    cout << "Производительность алгоритма составляет: " << p2 << " MFlops\n";
    cout << "matrixC2[20] = " << matrixC2[20] << endl;
    printLine();

    start = clock();
    matrixTransposition(matrixB, matrixBT);
    optimizedMatrixMultiplication(matrixA, matrixBT, matrixC3);
    end = clock();

    elapsed_secs = double(end - start) / CLOCKS_PER_SEC;

    double p3 = c / elapsed_secs * pow(10, -6);

    cout << "Время умножения матриц с помощью оптимизированного алгоритма: " << elapsed_secs << " секунд" << endl;
    cout << "Производительность алгоритма составляет: " << p3 << " MFlops\n";
    cout << "matrixC3[20] = " << matrixC3[20] << endl;
    printLine();

    cout << "Равны-ли матрицы matrixC2 и matrixC3: " << (matrixEqual(matrixC2, matrixC3, 1.e-3) ? "Равны" : "Неравны") << endl;
    printLine();

    cout << "Производительность оптимизированного алгоритма составляет " << (p3 / p2 * 100) << "% от 2-го варианта" << endl;
    printLine();

    delete[] matrixA;
    delete[] matrixB;
    delete[] matrixBT;
    delete[] matrixC1;
    delete[] matrixC2;
    delete[] matrixC3;

    return 0;
}
