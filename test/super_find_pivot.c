#include <stdio.h>
#include <assert.h>
#include "spasm.h"

int main(int argc, char **argv){

  spasm *A;
  super_spasm *L, *U;
  spasm_triplet *T1, *T2;
  int r, n, m, i, j, unz, lnz, prime, li, ui, top, npiv, found, test;
  int *xi, *qinv, *Lp, *Up;
  spasm_GFp *x, *y, *w, *v;

  assert(argc > 1);
  test = atoi(argv[1]);

  /*loading matrix */
  T1 = spasm_load_sms(stdin, 42013);
  T2 = spasm_triplet_alloc(T1->n, T1->m, T1->nzmax, T1->prime, 1);
  j = 0;
  for(i = 0; i < T1->nz; i++){
    if(T1->i[i] & 0x1){
      T2->i[j] = T1->i[i];
      T2->j[j] = T1->j[i];
      T2->x[j] = T1->x[i];
      j++;
    }
  }
  T2->nz = j;
  T2->nzmax = j;

  spasm_triplet_free(T1);
  A = spasm_compress(T2);
  spasm_triplet_free(T2);

  n = A->n;
  m = A->m;

  r = spasm_min(n, m);
  prime = A->prime;

  /* Allouer U et L */
  unz = 4 * spasm_nnz(A) + m;
  lnz = 4 * spasm_nnz(A) + m; // educated gess.

  U = super_spasm_alloc(n, r, m, unz, prime, 1);
  L = super_spasm_alloc(n, n, n, lnz, prime, 1);

  Up = U->M->p;
  Lp = L->M->p;

  /* Get workspace */
  x = spasm_malloc(m * sizeof(spasm_GFp));
  xi = spasm_malloc(3 * m * sizeof(int));
  spasm_vector_zero(xi, 3*m);
  spasm_vector_zero(x, m);
  qinv = spasm_malloc(m * sizeof(int));
  w = spasm_malloc(n * sizeof(int));
  spasm_vector_zero(w, n);

  /* initialize workspace */
  for(i = 0; i < m; i++){
    qinv[i] = -1;
  }

  for(i = 0; i < r; i++){
    Up[i] = 0;
  }

  for(i = 0; i < n; i++){
    Lp[i] = 0;
  }

  unz = lnz = 0;
  li = ui = 0;
  npiv = 0;

  /* main loop : compute L[i] and U[i] */
  for(i = 0; i < n; i++){
    if(!(i & 0x1)){
      continue;
    }
    
    Lp[li] = lnz;
    Up[ui] = unz;

 /* not enough room in L/U ? realloc twice the size */
    if (lnz + m > L->M->nzmax) {
      spasm_csr_realloc(L->M, 2 * L->M->nzmax + m);
    }
    if (unz + m > U->M->nzmax) {
      spasm_csr_realloc(U->M, 2 * U->M->nzmax + m);
    }

    /* triangular solve */
    top = spasm_sparse_forward_solve(U->M, A, i, xi, x, qinv);

    /* search for pivot */
    found = super_spasm_find_pivot(xi, x, top, U, L, &unz, &lnz, li, ui, i, qinv);

    li++;
    ui += found;
    npiv += found;
    w[i] = found; // w[i] : nombre de pivots sur la ligne i.

  }

/* Finalize L and U */

  Up[ui] = unz;
  spasm_csr_resize(U->M, ui, m);
  spasm_csr_realloc(U->M, -1);

  Lp[li] = lnz;
  spasm_csr_resize(L->M, li, n);
  spasm_csr_realloc(L->M, -1);
  Lp = L->M->p;

  free(x);

  /* Check result */
  // LU = spasm_LU(A, NULL, 1);

  // assert(npiv == LU->U->n);
  r = npiv;
  
  /*get workspace */
  x = malloc(n * sizeof(spasm_GFp));
  y = malloc(m * sizeof(spasm_GFp));
  v = malloc(m * sizeof(spasm_GFp));
 

for(i = 0; i < L->M->n; i++) {
    for(j = 0; j < n; j++) {
      x[j] = 0; // clear workspace.
      //      u[j] = 0;
    }
    for(j = 0; j < m; j++) {
      y[j] = 0;
      v[j] = 0;
    }
    for(j = Lp[i]; j < Lp[i+1]; j++){
      //scatter L[i:] in x:
      x[L->M->j[j]] = L->M->x[j];
    }
    x[L->p[i]] = (w[L->p[i]] == 1)? 1 : 0; //rajoute un 1 sur l'entrée diag si pivot
    for(j = A->p[L->p[i]]; j < A->p[L->p[i] + 1]; j++){
      //scatter same row of A in y
      y[A->j[j]] = A->x[j];
    }

    super_sparse_gaxpy_dense(U, x, v); // v <- x*U

    for(j = 0; j < m; j++) {
      if (y[j] != v[j]) {
	printf("not ok %d - L*U == A (col %d) row %d \n", test, j, i);
	//printf("y : %d, v : %d\n", y[j], v[j]);
	exit(0);
      }
    }
  }


 printf("ok %d super_find_pivot \n", test);

  /* free memory */
  super_spasm_free(U);
  spasm_csr_free(A);
  super_spasm_free(L);
  free(x);
  free(xi);
  free(y);
  free(v);
  free(w);
  free(qinv);
  //  free(w);

}
