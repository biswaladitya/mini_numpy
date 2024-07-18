#include <omp.h>
#include <x86intrin.h>

#include "compute.h"

//Computes the dot product of vec1 and vec2, both of size n using SIMD instructions
int dot(uint32_t n, int32_t *vec1, int32_t *vec2) {
    int32_t sum = 0;

    // Compute the dot product using SIMD instructions
    __m256i acc = _mm256_setzero_si256();
    for (int i = 0; i < n/8*8; i += 8) {
        __m256i v1 = _mm256_loadu_si256((__m256i*)(vec1 + i));
        __m256i v2 = _mm256_loadu_si256((__m256i*)(vec2 + i));
        __m256i prod = _mm256_mullo_epi32(v1, v2);
        acc = _mm256_add_epi32(acc, prod);
    }
    int32_t sum_array[8] __attribute__((aligned(32)));
    _mm256_store_si256((__m256i*)sum_array, acc);
    for (int i = 0; i < 8; i++) {
        sum += sum_array[i];
    }

    // Compute the dot product for any remaining elements
    for (int i = n/8*8; i < n; i++) {
        sum += vec1[i] * vec2[i];
    }

    return sum;
}


// Computes the convolution of two matrices
int convolve(matrix_t *a_matrix, matrix_t *b_matrix, matrix_t **output_matrix) {
  // Allocate memory for the output matrix
  *output_matrix = (matrix_t *) malloc(sizeof(matrix_t));

  (*output_matrix)->rows = a_matrix->rows - b_matrix->rows + 1;

  (*output_matrix)->cols = a_matrix->cols - b_matrix->cols + 1;

  (*output_matrix)->data = (int32_t *) malloc((*output_matrix)->rows * (*output_matrix)->cols * sizeof(int32_t));

  int i = 0; 
  int j = (b_matrix->rows * b_matrix->cols) - 1; 
  int temp;

  //reverse the B matrix (i.e. flip it horizontally and vertically
  while (i < j) { 
    temp = b_matrix->data[j]; 
    b_matrix->data[j] = b_matrix->data[i]; 
    b_matrix->data[i] = temp; 
    i++; 
    j--;
  }

  // Perform the convolution operation
  int b_cols = b_matrix->cols;
  int a_cols = a_matrix->cols;
  int b_rows = b_matrix->rows;
  int a_rows = a_matrix->rows;
  int32_t* a_data = a_matrix->data;
  int32_t* b_data = b_matrix->data;
  int out_rows = (*output_matrix)->rows;
  int out_cols = (*output_matrix)->cols;

  int32_t *temp_output_matrices[b_rows];
for (int i = 0; i < b_rows; i++) {
    temp_output_matrices[i] = (int32_t *)calloc(out_rows * out_cols, sizeof(int32_t));
}

#pragma omp parallel for
for (int b_row = 0; b_row < b_rows; b_row++) { // iterate through rows of kernel matrix
  for (int col = 0; col < out_cols; col++) { // to multiply row of matrix left to right properly
    for (int row = 0; row < out_rows; row++) { // to do the right # of multiplication (each row is only multiplied a_rows - b_row # of times)
      int32_t element_value = dot(b_cols, a_data + (b_row + row) * a_cols + col, b_data + (b_row * b_cols)); // multiplies right data from a with correct row of b
      temp_output_matrices[b_row][row * out_cols + col] = element_value;
    }
  }
}

// Initialize the output matrix with zeros
memset((*output_matrix)->data, 0, sizeof(int32_t) * out_rows * out_cols);

// Add all temporary output matrices together using SIMD
for (int k = 0; k < b_rows; k++) {
    for (int i = 0; i < out_rows * out_cols / 8 * 8; i += 8) {
        __m256i out_data = _mm256_loadu_si256((__m256i*)((*output_matrix)->data + i));
        __m256i temp_data = _mm256_loadu_si256((__m256i*)(temp_output_matrices[k] + i));
        __m256i added_data = _mm256_add_epi32(out_data, temp_data);
        _mm256_storeu_si256((__m256i*)((*output_matrix)->data + i), added_data);
    }
    for (int i = out_rows * out_cols / 8 * 8; i < out_rows * out_cols; i++) {
        (*output_matrix)->data[i] += temp_output_matrices[k][i];
    }
}
// Free memory allocated for temporary output matrices
for (int i = 0; i < b_rows; i++) {
    free(temp_output_matrices[i]);
}

return 0;










//  #pragma omp parallel for collapse(2)
//     for (i = 0; i < out_rows; i++) {
//       for (j = 0; j < out_cols; j++) {
//         int32_t sum = 0;
//         #pragma omp parallel for reduction(+:sum)
//         for (int k = 0; k < b_rows; k++) {
//           int leftIndex = (i+k) * (a_cols) + j; 
//           sum += dot(b_cols, a_data + leftIndex, b_data + (k * b_cols)); 
//         }
//         (*output_matrix)->data[i * out_cols + j] = sum;
//       }
//     }
    
//   return 0; 
}

// Executes a task
int execute_task(task_t *task) {
  matrix_t *a_matrix, *b_matrix, *output_matrix;

  if (read_matrix(get_a_matrix_path(task), &a_matrix))
    return -1;
  if (read_matrix(get_b_matrix_path(task), &b_matrix))
    return -1;

  if (convolve(a_matrix, b_matrix, &output_matrix))
    return -1;

  if (write_matrix(get_output_matrix_path(task), output_matrix))
    return -1;

  free(a_matrix->data);
  free(b_matrix->data);
  free(output_matrix->data);
  free(a_matrix);
  free(b_matrix);
  free(output_matrix);
  return 0;
}
