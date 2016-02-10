/*
  This file is part of MADNESS.

  Copyright (C) 2007,2010 Oak Ridge National Laboratory

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA

  For more information please contact:

  Robert J. Harrison
  Oak Ridge National Laboratory
  One Bethel Valley Road
  P.O. Box 2008, MS-6367

  email: harrisonrj@ornl.gov
  tel:   865-241-3937
  fax:   865-572-0680

  $Id$
*/

#define VALUE_TEST
#define SPEED_TEST

#include <madness/madness_config.h>

// Disable for now to facilitate CI 
#if !(defined(X86_32) || defined(X86_64))

#include <iostream>
int main() {std::cout << "x86 only\n"; return 0;}

#else

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
//#include <xmmintrin.h>
#include <sys/mman.h>

#include <madness/world/safempi.h>
#include <madness/world/posixmem.h>
#include <madness/tensor/cblas.h>
#include <madness/tensor/tensor.h>
#include <madness/tensor/mtxmq.h>

using namespace madness;


#ifdef TIME_DGEMM

void mTxm_dgemm(long ni, long nj, long nk, double* c, const double* a, const double*b, const long a_pad, const long b_pad) {
  double one=1.0;
  cblas::gemm(cblas::NoTrans,cblas::Trans,nj,ni,nk,one,b,nj+b_pad,a,ni+a_pad,one,c,nj+b_pad);
  //cblas::gemm(cblas::Trans,cblas::NoTrans,ni,nj,nk,one,a,nk+a_pad,b,nk+b_pad,one,c,nk+b_pad);
}

#endif

double ran()
{
  static unsigned long seed = 76521;

  seed = seed *1812433253 + 12345;

  return ((double) (seed & 0x7fffffff)) * 4.6566128752458e-10;
}

void ran_fill(int n, double *a) {
    while (n--) *a++ = ran();
}

int calc_pad(int jmax){
	return  (int)(64 * ceil(8 * jmax / 64.0) - 8*jmax)/8;
}

int ran_fill_pad(int imax, int jmax, double *a) {
	int pad = (int)(64 * ceil(8 * jmax / 64.0) - 8*jmax)/8;
	for(int i=0; i<imax; i++){
		for(int j=0; j<jmax; j++){
			*a++ = ran();
		}
		for(int j=0; j<pad; j++){
			*a++ = 0;
		}
	}
	return pad;
}

void mTxm(long dimi, long dimj, long dimk,
          double* c, const double* a, const double* b, const int a_pad, const int b_pad) {
    int i, j, k;
    for (k=0; k<dimk; ++k) {
        for (j=0; j<dimj; ++j) {
            for (i=0; i<dimi; ++i) {
                c[i*(dimj+b_pad)+j] += a[k*(dimi+a_pad)+i]*b[k*(dimj+b_pad)+j];
            }
        }
    }
}

void mTxm_kji_res(long dimi, long dimj, long dimk,
          double* __restrict__ c, const double* __restrict__  a, const double* __restrict__ b) {
    int i, j, k;
    for (k=0; k<dimk; ++k) {
        for (j=0; j<dimj; ++j) {
            for (i=0; i<dimi; ++i) {
                c[i*dimj+j] += a[k*dimi+i]*b[k*dimj+j];
            }
        }
    }
}

//bad program
void mTxm_ijk(long dimi, long dimj, long dimk,
          double* c, const double* a, const double* b) {
    int i, j, k;
		for (i=0; i<dimi; ++i) {
   		 for (j=0; j<dimj; ++j) {
    				for (k=0; k<dimk; ++k) {
                c[i*dimj+j] += a[k*dimi+i]*b[k*dimj+j];
            }
        }
    }
}

void mTxm_ijk_res(long dimi, long dimj, long dimk,
          double* __restrict__ c, const double* __restrict__ a, const double* __restrict__ b) {
    int i, j, k;
		for (i=0; i<dimi; ++i) {
   		 for (j=0; j<dimj; ++j) {
    				for (k=0; k<dimk; ++k) {
                c[i*dimj+j] += a[k*dimi+i]*b[k*dimj+j];
            }
        }
    }
}

void mTxm_kij(long dimi, long dimj, long dimk,
          double* c, const double* a, const double* b) {
    int i, j, k;
		for (k=0; k<dimk; ++k) {
   		 for (i=0; i<dimi; ++i) {
    				for (j=0; j<dimj; ++j) {
                c[i*dimj+j] += a[k*dimi+i]*b[k*dimj+j];
            }
        }
    }
}

void mTxm_kij_res(long dimi, long dimj, long dimk,
          double* __restrict__ c, const double* __restrict__ a, const double* __restrict__ b) {
    int i, j, k;
		for (k=0; k<dimk; ++k) {
   		 for (i=0; i<dimi; ++i) {
    				for (j=0; j<dimj; ++j) {
                c[i*dimj+j] += a[k*dimi+i]*b[k*dimj+j];
            }
        }
    }
}

void crap(double rate, double fastest, double start) {
    if (rate == 0) printf("darn compiler bug %e %e %lf\n",rate,fastest,start);
}


//t=1000, loop=500 is better
void timer(const char* s, long ni, long nj, long nk, double * __restrict__ a, double * __restrict__ b, double * __restrict__ c, const int a_pad, const int b_pad, const int c_pad) {
  double fastest=0.0, fastest_dgemm=0.0;

  double nflop = 2.0*ni*nj*nk;
  long loop;
  for (int t=0; t<100; t++) {
    double rate;
    double start = SafeMPI::Wtime();
    for (loop=0; loop<100; ++loop) {
      mTxmq(ni,nj,nk,c,a,b, a_pad, b_pad, c_pad);
    }
    start = SafeMPI::Wtime() - start;
    rate = 1.e-9*nflop/(start/100.0); 
    crap(rate,fastest,start);
    if (rate > fastest) fastest = rate;
  }
#ifdef TIME_DGEMM
/*
  for (int t=0; t<100; t++) {
    double rate;
    double start = SafeMPI::Wtime();
    for (loop=0; loop<100; ++loop) {
      mTxm_dgemm(ni,nj,nk,c,a,b,a_pad, b_pad);
    }
    start = SafeMPI::Wtime() - start;
    rate = 1.e-9*nflop/(start/100.0);
    crap(rate,fastest_dgemm,start);
    if (rate > fastest_dgemm) fastest_dgemm = rate;
  }
*/
#endif
  printf("%20s %3ld %3ld %3ld %8.2f %8.2f\n",s, ni,nj,nk, fastest, fastest_dgemm);
}

void trantimer(const char* s, long ni, long nj, long nk, double *a, double *b, double *c, const int a_pad, const int b_pad, const int c_pad) {
  double fastest=0.0, fastest_dgemm=0.0;

  double nflop = 3.0*2.0*ni*nj*nk;
  long loop;
  for (int t=0; t<100; t++) {
    double rate;
    double start = SafeMPI::Wtime();
    for (loop=0; loop<100; ++loop) {
      mTxmq(ni,nj,nk,c,a,b, a_pad, b_pad, c_pad);
      mTxmq(ni,nj,nk,a,c,b, a_pad, b_pad, c_pad);
      mTxmq(ni,nj,nk,c,a,b, a_pad, b_pad, c_pad);
    }
    start = SafeMPI::Wtime() - start;
    rate = 1.e-9*nflop/(start/100.0);
    crap(rate,fastest,start);
    if (rate > fastest) fastest = rate;
  }
#ifdef TIME_DGEMM
/*
  for (int t=0; t<100; t++) {
    double rate;
    double start = SafeMPI::Wtime();
    for (loop=0; loop<100; ++loop) {
      mTxm_dgemm(ni,nj,nk,c,a,b,a_pad,b_pad);
      mTxm_dgemm(ni,nj,nk,a,c,b,a_pad,b_pad);
      mTxm_dgemm(ni,nj,nk,c,a,b,a_pad,b_pad);
    }
    start = SafeMPI::Wtime() - start;
    rate = 1.e-9*nflop/(start/100.0);
    crap(rate,fastest_dgemm,start);
    if (rate > fastest_dgemm) fastest_dgemm = rate;
  }
 */
#endif
  printf("%20s %3ld %3ld %3ld %8.2f %8.2f\n",s, ni,nj,nk, fastest, fastest_dgemm);
}

int main(int argc, char * argv[]) {
	/*
    const long nimax=30*30;
    const long njmax=24;
    const long nkmax=100;
	*/
    const long nimax=1000;
    const long njmax=1000;
    const long nkmax=1000;
    long ni, nj, nk, i, m;
    double *a, *b, *c, *d;

    SafeMPI::Init_thread(argc, argv, MPI_THREAD_SINGLE);

    posix_memalign((void **) &a, 32, nkmax*nimax*sizeof(double));
    posix_memalign((void **) &b, 32, nkmax*njmax*sizeof(double));
    posix_memalign((void **) &c, 32, nimax*njmax*sizeof(double));
    posix_memalign((void **) &d, 32, nimax*njmax*sizeof(double));
	/*
	madvise((void**)a, nkmax*nimax*sizeof(double), MADV_HUGEPAGE);
	madvise((void**)b, nkmax*njmax*sizeof(double), MADV_HUGEPAGE);
	madvise((void**)c, nimax*njmax*sizeof(double), MADV_HUGEPAGE);
	madvise((void**)d, nimax*njmax*sizeof(double), MADV_HUGEPAGE);
	*/
	/*
	a = (double*)_mm_malloc(nkmax*nimax*sizeof(double), 32);
	b = (double*)_mm_malloc(nkmax*njmax*sizeof(double), 32);
	c = (double*)_mm_malloc(nimax*njmax*sizeof(double), 32);
	d = (double*)_mm_malloc(nimax*njmax*sizeof(double), 32);
	*/
	/*
	a = (double*)mmap(NULL, nimax*njmax*sizeof(double), PROT_READ|PROT_WRITE, MAP_ANONYMOUS|MAP_PRIVATE, -1, 0);
	b = (double*)mmap(NULL, nkmax*njmax*sizeof(double), PROT_READ|PROT_WRITE, MAP_ANONYMOUS|MAP_PRIVATE|MAP_HUGETLB, -1, 0);
	c = (double*)mmap(NULL, nimax*njmax*sizeof(double), PROT_READ|PROT_WRITE, MAP_ANONYMOUS|MAP_PRIVATE|MAP_HUGETLB, -1, 0);
	d = (double*)mmap(NULL, nimax*njmax*sizeof(double), PROT_READ|PROT_WRITE, MAP_ANONYMOUS|MAP_PRIVATE|MAP_HUGETLB, -1, 0);
	*/

    ran_fill(nkmax*nimax, a);
    ran_fill(nkmax*njmax, b);


/*     ni = nj = nk = 2; */
/*     for (i=0; i<ni*nj; ++i) d[i] = c[i] = 0.0; */
/*     mTxm (ni,nj,nk,c,a,b); */
/*     mTxmq(ni,nj,nk,d,a,b); */
/*     for (i=0; i<ni; ++i) { */
/*       long j; */
/*       for (j=0; j<nj; ++j) { */
/* 	printf("%2ld %2ld %.6f %.6f\n", i, j, c[i*nj+j], d[i*nj+j]); */
/*       } */
/*     } */
/*     return 0; */

#ifdef VALUE_TEST
    printf("Starting to test ... \n");
    for (ni=1; ni<60; ni+=1) {
        for (nj=1; nj<100; nj+=1) {
            for (nk=1; nk<100; nk+=1) {
                //for (i=0; i<ni*nj; ++i){ d[i] = c[i] = 0.0;}
				int a_pad = calc_pad(ni);
				int b_pad = calc_pad(nj);
				memset(c, 0, ni*(nj+b_pad)*sizeof(double));
				memset(d, 0, ni*(nj+b_pad)*sizeof(double));
                mTxm(ni,nj,nk,c,a,b,a_pad,b_pad);
				//mTxm_dgemm(ni,nj,nk,c,a,b,a_pad,b_pad);
                mTxmq(ni,nj,nk,d,a,b,a_pad,b_pad,0);
                for (i=0; i<ni*nj; ++i) {
                    double err = std::abs(d[i]-c[i]);
                    /* This test is sensitive to the compilation options.
                       Be sure to have the reference code above compiled
                       -msse2 -fpmath=sse if using GCC.  Otherwise, to
                       pass the test you may need to change the threshold
                       to circa 1e-13.
                    */
                    if (err > 1e-13) {
                        printf("test_mtxmq: error %ld %ld %ld %e\n",ni,nj,nk,err);
                        exit(1);
                    }
                }
            }
        }
    }
    printf("... OK!\n");
#endif

#ifdef SPEED_TEST

    printf("%20s %3s %3s %3s %8s %8s (GF/s)\n", "type", "M", "N", "K", "LOOP", "BLAS");
    for (ni=1; ni<60; ni+=1){
		int a_pad = calc_pad(ni);
		int b_pad = calc_pad(ni);
		timer("(m*m)T*(m*m)", ni,ni,ni,a,b,c,a_pad,b_pad,0);
	}
    for (m=1; m<=30; m+=1){
		int a_pad = calc_pad(m*m);
		int b_pad = calc_pad(m);
		timer("(m*m,m)T*(m*m)", m*m,m,m,a,b,c,a_pad,b_pad,0);
	}
    for (m=1; m<=30; m+=1){
		int a_pad = calc_pad(m*m);
		int b_pad = calc_pad(m);
		trantimer("tran(m,m,m)", m*m,m,m,a,b,c,a_pad,b_pad,0);
	}
    for (m=1; m<=20; m+=1){
		int a_pad = calc_pad(20*20);
		int b_pad = calc_pad(m);
		timer("(20*20,20)T*(20,m)", 20*20,m,20,a,b,c,a_pad,b_pad,0);
	}
#endif

	//for VTune
	/*
	for (i=0; i<100; i++){
		int nj = 26;
		int a_pad = calc_pad(nj);
		int b_pad = calc_pad(nj*nj);
		int c_pad = b_pad;
		//timer("(m*m)T*(m*m)", 20,20,20,a,b,c, a_pad, b_pad, c_pad);
		timer("(m*m,m)T*(m*m)", nj,nj*nj,nj,c,a,b,a_pad,b_pad,c_pad);

	}
	*/

    SafeMPI::Finalize();

    return 0;
}


#endif
