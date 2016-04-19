#include <assert.h>
#include "spasm.h"

#define INSERT_SORT_THRESHOLD 42 // TODO : tune this value


/* sort up to index right, excluded */
static void insertion_sort(const spasm *A, int *p, const int left, const int right) {
  int i, j, u, v;

  /*  if (left <= 0) { */
    for(i = left + 1; i < right; i++) {
      u = p[i];
      v = spasm_row_weight(A, p[i]);
      j = i - 1;
      while (j >= 0 && spasm_row_weight(A, p[j]) > v) {
	      p[j + 1] = p[j];
	      j--;
      }
      p[j + 1] = u;
    }
    /*
  } else {
    TODO, possible optimization : if ( left>0 ), then we know that
    on the left of the current subfile, there is an element smaller
    than all the elements of the subfile (because this element was a pivot).
    Therefore, we don't have to check explicitly that we attained the left
    boundary of the array...
    */
}


/* standard median-of-three pivoting strategy. Returns the pivot index */
static int choose_pivot(const spasm *A, int *p, const int left, const int right) {
  int mid = (left+right)/2;

  if (spasm_row_weight(A, p[mid - 1]) > spasm_row_weight(A, p[mid])) {
    spasm_swap(p, mid - 1, mid);
  }

  if (spasm_row_weight(A, p[mid - 1]) > spasm_row_weight(A, p[mid + 1])) {
    spasm_swap(p, mid - 1, mid + 1);
  }
  if (spasm_row_weight(A, p[mid]) > spasm_row_weight(A, p[mid + 1])) {
    spasm_swap(p, mid, mid + 1);
  }

  return mid;
}


/* returns final position of pivot */
static int pivoting(const spasm *A, int *p, const int initial_left, const int initial_right, const int pivotIndex) {
  int pivotValue, left, right;

  spasm_swap(p, pivotIndex, initial_right - 1);
  pivotValue = spasm_row_weight(A, p[initial_right - 1]);

  right = initial_right - 2;
  left = initial_left;

  while(left < right) {
    while(spasm_row_weight(A, p[left]) < pivotValue) {
      left++;
    }
    while(spasm_row_weight(A, p[right]) > pivotValue) {
      right--;
    }

    if (left < right) {
      spasm_swap(p, left, right);
      left++;
    }
  }

  if (spasm_row_weight(A, p[right]) < pivotValue) {
    right++;
  }
  spasm_swap(p, right, initial_right - 1);
  return right;
}

static void spasm_quicksort(const spasm *A, int *p, const int left, const int right) {
  int pivotIndex, new_pivotIndex;

  if (right-left > INSERT_SORT_THRESHOLD) {

    pivotIndex = choose_pivot(A, p, left, right);
    new_pivotIndex = pivoting(A, p, left, right, pivotIndex);

    spasm_quicksort(A, p, left, new_pivotIndex);
    spasm_quicksort(A, p, new_pivotIndex + 1, right);
  } else {
    insertion_sort(A, p, left, right);
  }
}


int * spasm_row_sort (const spasm *A) {
  int *p;
  int i, n;

  n = A->n;
  p = spasm_malloc(n * sizeof(int));
  for(i = 0; i < n; i++) {
    p[i] = i;
  }
  spasm_quicksort(A, p, 0, n);
  return p;
}


int * spasm_cheap_pivots(const spasm *A, int *cheap_ptr) {
  int n, m, i, j, k, I, idx_j,  px, n_cheap, pxI, head, tail;
  int *q, *p, *Ap, *Aj, *w, *queue;
  spasm_GFp *Ax;

  n = A->n;
  m = A->m;
  Ap = A->p;
  Aj = A->j;
  Ax = A->x;

  q = spasm_malloc(m * sizeof(int));
  p = spasm_malloc(n * sizeof(int));
  w = spasm_malloc(m * sizeof(int));
  queue = spasm_malloc(m * sizeof(int));

  /* --- Cheap pivot selection ----------------------------------- */
  for(j = 0; j < m; j++) {
    q[j] = -1;
  }
  for(i = 0; i < n; i++) {
  
    /* find leftmost entry */
    j = -1;
    for(px = Ap[i]; px < Ap[i + 1]; px++) {
      if (j == -1 || Aj[px] < j) {
	      j = Aj[px];
	      idx_j = px;
      }
    }
    /* Skip empty rows */
    if (j == -1) {
      continue;
    }

    /* make sure leftmost entry is the first of the row */
    spasm_swap(Aj, Ap[i], idx_j);
    spasm_swap(Ax, Ap[i], idx_j);

    /* check if it is a sparser pivot */
    if (q[j] == -1 || spasm_row_weight(A, i) < spasm_row_weight(A, q[j])) {
      q[j] = i;
    }
  }
  
  k = 0;
  for(j = 0; j < m; j++) {
    if (q[j] != -1) {
      // printf("selected (%d, %d)\n", q[j], j);
      k++;
    }
  }
  fprintf(stderr, "[LU] found %d cheap pivots (stage1)\n", k);

  /* --- find less-cheap pivots ----------------------------------- */  

  n_cheap = k;
  /* workspace initialization */
  for(j=0; j<m; j++) {
    w[j] = 0;
  }

  for(i = 0; i < n; i++) {  
    if (q[ Aj[ Ap[i]] ] == i) { /* this row is already pivotal: skip */
      continue;
    }
    
    if (i % (n/100) == 0) {
      fprintf(stderr, "\rcheap : %d / %d --- rank >= %d", i, n, n_cheap);
      fflush(stderr);
    }
    // printf("------------------------ %d\n", i);

    /* scatters non-pivotal columns A[i] into w */
    for(px = Ap[i]; px < Ap[i + 1]; px++) {
      j = Aj[px];
      if (w[j] != 0) {
        printf("w[%d] == %d\n", j, w[j]);
        exit(1);
      }
      if (q[j] != -1) { /* column is pivotal: skip */
        continue;
      }
      w[j] = 1;
    }
    
    head = 0;
    tail = 0;
    for(px = Ap[i]; px < Ap[i + 1]; px++) {
      j = Aj[px];
      if (q[j] == -1) { /* column is not pivotal: skip */
        continue;
      }

      if (w[j] < 0) { /* already marked: skip */
        continue;
      }

      queue[tail] = j;
      tail++;
      w[j] = -1;
      // printf("amorçage en %d\n", j);
      /* BFS */
      while (head < tail) {
        j = queue[head];
        assert(w[j] < 0);
        head++;

        I = q[j];
        if (I == -1) {
          continue; /* nothing to do */
        }

        for (pxI = Ap[I]; pxI < Ap[I + 1]; pxI++) {
          j = Aj[pxI];
          if (w[j] < 0) {
            continue; /* already marked */
          }
          queue[tail] = j;
          tail++;
          w[j] = -1;
          // printf("marquage en %d\n", j);
        }
      }
    }

    /* scan w for surviving entries */
    k = -1;
    for(px = Ap[i]; px < Ap[i + 1]; px++) {
      j = Aj[px];
      if ((k == -1) && (w[j] == 1)) {
        k = j;
        idx_j = px;
      }
      w[j] = 0;
    }
     
    /* reset w */
    for(px = 0; px < tail; px++) {
      j = queue[px];
      w[j] = 0;
    }

    if (k != -1) {
      assert(q[k] == -1);
      // fprintf(stderr, "Also possible: %d / %d\n", i, k);
      q[k] = i;
    
      /* make sure leftmost entry is the first of the row */
      spasm_swap(Aj, Ap[i], idx_j);
      spasm_swap(Ax, Ap[i], idx_j);
      n_cheap++;
    }



  }


  /* --- build corresponding row permutation ---------------------- */

  /* put cheap pivot rows in increasing column order */
  k = 0;
  for(j = 0; j < m; j++) {
    if (q[j] != -1) {
      assert (k<n);
      p[k] = q[j];
      assert(q[j] >= 0);
      assert(q[j] < n);
      k++;
    }
  }

  n_cheap = k;
  *cheap_ptr = n_cheap;

  /* put other (non-empty) rows afterwards */
  for(i = 0; i < n; i++) {
    if (Ap[i] == Ap[i + 1]) {
      continue;
    }
    if (q[ Aj[ Ap[i] ] ] != i) {
      assert (k<n);
      p[k] = i;
      k++;
    }
  }

  /* put empty rows last */
  for(i = 0; i < n; i++) {
    if (Ap[i] == Ap[i + 1]) {
      p[k] = i;
      k++;
    }
  }
  
  free(q);
  fprintf(stderr, "[LU] found %d cheap pivots (stage2)\n", n_cheap);

  return p;
}
