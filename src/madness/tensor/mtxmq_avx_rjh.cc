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
#include <madness/madness_config.h>
#include <madness/tensor/tensor.h>
#include <madness/tensor/mtxmq.h>
#include <madness/world/worldprofile.h>

#include <complex.h>
#include <immintrin.h>

// For x86-32/64 have assembly versions for double precision
// For x86-64 have assembly versions for complex double precision

#if defined(X86_32) || defined(X86_64)

#ifdef X86_64
extern "C" void mTxm26(long dimi, long dimj, long dimk,
                       double* c, const double* a, const double* b) ;

extern "C" void mTxm24(long dimi, long dimj, long dimk,
                       double* c, const double* a, const double* b) ;

extern "C" void mTxm22(long dimi, long dimj, long dimk,
                       double* c, const double* a, const double* b) ;

extern "C" void mTxm20(long dimi, long dimj, long dimk,
                       double* c, const double* a, const double* b) ;

extern "C" void mTxm18(long dimi, long dimj, long dimk,
                       double* c, const double* a, const double* b) ;

extern "C" void mTxm16(long dimi, long dimj, long dimk,
                       double* c, const double* a, const double* b) ;

extern "C" void mTxm14(long dimi, long dimj, long dimk,
                       double* c, const double* a, const double* b) ;

extern "C" void mTxm12(long dimi, long dimj, long dimk,
                       double* c, const double* a, const double* b) ;
#endif // X86_64
extern "C" void mTxm10(long dimi, long dimj, long dimk,
                       double* c, const double* a, const double* b) ;

extern "C" void mTxm8(long dimi, long dimj, long dimk,
                      double* c, const double* a, const double* b) ;

extern "C" void mTxm6(long dimi, long dimj, long dimk,
                      double* c, const double* a, const double* b) ;

extern "C" void mTxm4(long dimi, long dimj, long dimk,
                      double* c, const double* a, const double* b) ;

extern "C" void mTxm2(long dimi, long dimj, long dimk,
                      double* c, const double* a, const double* b) ;


#ifdef X86_64
extern "C" void TmTxm26(long dimi, long dimj, long dimk,
                        double* c, const double* a, const double* b) ;

extern "C" void TmTxm24(long dimi, long dimj, long dimk,
                        double* c, const double* a, const double* b) ;

extern "C" void TmTxm22(long dimi, long dimj, long dimk,
                        double* c, const double* a, const double* b) ;

extern "C" void TmTxm20(long dimi, long dimj, long dimk,
                        double* c, const double* a, const double* b) ;

extern "C" void TmTxm18(long dimi, long dimj, long dimk,
                        double* c, const double* a, const double* b) ;

extern "C" void TmTxm16(long dimi, long dimj, long dimk,
                        double* c, const double* a, const double* b) ;

extern "C" void TmTxm14(long dimi, long dimj, long dimk,
                        double* c, const double* a, const double* b) ;

extern "C" void TmTxm12(long dimi, long dimj, long dimk,
                        double* c, const double* a, const double* b) ;
#endif // X86_64
extern "C" void TmTxm10(long dimi, long dimj, long dimk,
                        double* c, const double* a, const double* b) ;

extern "C" void TmTxm8(long dimi, long dimj, long dimk,
                       double* c, const double* a, const double* b) ;

extern "C" void TmTxm6(long dimi, long dimj, long dimk,
                       double* c, const double* a, const double* b) ;

extern "C" void TmTxm4(long dimi, long dimj, long dimk,
                       double* c, const double* a, const double* b) ;

extern "C" void TmTxm2(long dimi, long dimj, long dimk,
                       double* c, const double* a, const double* b) ;

namespace madness {

#include <immintrin.h>
#include <stdio.h>


//#define FMA(a,b,c) _mm256_fmadd_pd (a, b, c)
#define FMA(a,b,c) _mm256_add_pd(_mm256_mul_pd(a, b), c)

void mTxmq_core(bool is_trans, long dimi, long dimj, long dimk,
           double * __restrict__ c, const double * __restrict__ a, const double * __restrict__ b, long numi, long numj) {
	
    int i, k;
    int dimi2 = (numi>>1)<<1;
    int dimj2 = dimj<<1;
	double tmp[4];

    __m256d ci0j0, ci0j1, ci0j2, ci0j3, ci0j4, ci0j5;
    __m256d ci1j0, ci1j1, ci1j2, ci1j3, ci1j4, ci1j5;
    __m256d aki0, aki1, bkj;
    __m256i mask = _mm256_set_epi32(0,0,-1,-1,-1,-1,-1,-1);
    // __m256d tmp; //temporary from aki*bkj
    
	double* __restrict__ ci = c;

	//pointer converter 
	const auto conv_addr_trans2normal = [dimi, c](long i, long j){return c + dimi * j + i;};

	switch (numj) {
	case 24:
	case 23:
	case 22:
	case 21:
		if      (numj == 24) mask = _mm256_set_epi32(-1,-1,-1,-1,-1,-1,-1,-1);
		else if (numj == 23) mask = _mm256_set_epi32( 0, 0,-1,-1,-1,-1,-1,-1);
		else if (numj == 22) mask = _mm256_set_epi32( 0, 0, 0, 0,-1,-1,-1,-1);
		else if (numj == 21) mask = _mm256_set_epi32( 0, 0, 0, 0, 0, 0,-1,-1);

		for (i=0; i<dimi2; i+=2,ci+=dimj2) {
			const double* __restrict__ pbkj = b;
			const double* __restrict__ paki = a+i;
			ci0j0 = _mm256_setzero_pd();
			ci0j1 = _mm256_setzero_pd();
			ci0j2 = _mm256_setzero_pd();
			ci0j3 = _mm256_setzero_pd();
			ci0j4 = _mm256_setzero_pd();
			ci0j5 = _mm256_setzero_pd();
			
			ci1j0 = _mm256_setzero_pd();
			ci1j1 = _mm256_setzero_pd();
			ci1j2 = _mm256_setzero_pd();
			ci1j3 = _mm256_setzero_pd();
			ci1j4 = _mm256_setzero_pd();
			ci1j5 = _mm256_setzero_pd();

			for (k=0; k<dimk; k++,pbkj+=dimj,paki+=dimi) {
				aki0 = _mm256_broadcast_sd(paki);
				aki1 = _mm256_broadcast_sd(paki+1);
				
				bkj = _mm256_loadu_pd(pbkj   );
				ci0j0 = FMA(aki0, bkj, ci0j0);
				ci1j0 = FMA(aki1, bkj, ci1j0);
				
				bkj = _mm256_loadu_pd(pbkj+ 4);
				ci0j1 = FMA(aki0, bkj, ci0j1);
				ci1j1 = FMA(aki1, bkj, ci1j1);
				
				bkj = _mm256_loadu_pd(pbkj+ 8);
				ci0j2 = FMA(aki0, bkj, ci0j2);
				ci1j2 = FMA(aki1, bkj, ci1j2);
				
				bkj = _mm256_loadu_pd(pbkj+12);
				ci0j3 = FMA(aki0, bkj, ci0j3);
				ci1j3 = FMA(aki1, bkj, ci1j3);
				
				bkj = _mm256_loadu_pd(pbkj+16);
				ci0j4 = FMA(aki0, bkj, ci0j4);
				ci1j4 = FMA(aki1, bkj, ci1j4);

				bkj = _mm256_maskload_pd(pbkj+20,mask);
				ci0j5 = FMA(aki0, bkj, ci0j5);
				ci1j5 = FMA(aki1, bkj, ci1j5);
			}
			if(!is_trans){
				_mm256_storeu_pd(c+i*dimj   , ci0j0);
				_mm256_storeu_pd(c+i*dimj+ 4, ci0j1);
				_mm256_storeu_pd(c+i*dimj+ 8, ci0j2);
				_mm256_storeu_pd(c+i*dimj+12, ci0j3);
				_mm256_storeu_pd(c+i*dimj+16, ci0j4);
				_mm256_maskstore_pd(c+i*dimj+20, mask, ci0j5);
				
				_mm256_storeu_pd(c+(i+1)*dimj   , ci1j0);
				_mm256_storeu_pd(c+(i+1)*dimj+ 4, ci1j1);
				_mm256_storeu_pd(c+(i+1)*dimj+ 8, ci1j2);
				_mm256_storeu_pd(c+(i+1)*dimj+12, ci1j3);
				_mm256_storeu_pd(c+(i+1)*dimj+16, ci1j4);
				_mm256_maskstore_pd(c+(i+1)*dimj+20, mask, ci1j5);
			}else{
				//temporary use aki0, aki1 and mask
				//the variables don't mean aki0 or aki1
				//please ignore the meaning of name
				aki0 = _mm256_shuffle_pd(ci0j0, ci1j0, 0);
				aki1 = _mm256_shuffle_pd(ci0j0, ci1j0, -1);

				//store transposed matrix's location c[j*dimi+i]
				_mm256_storeu2_m128d(c+2*dimi+i, c+0*dimi+i, aki0);
				_mm256_storeu2_m128d(conv_addr_trans2normal(i,2), conv_addr_trans2normal(i,0), aki0);
				_mm256_storeu2_m128d(conv_addr_trans2normal(i,3), conv_addr_trans2normal(i,1), aki1);

				aki0 = _mm256_shuffle_pd(ci0j1, ci1j1, 0);
				aki1 = _mm256_shuffle_pd(ci0j1, ci1j1, -1);
				_mm256_storeu2_m128d(conv_addr_trans2normal(i,6), conv_addr_trans2normal(i,4), aki0);
				_mm256_storeu2_m128d(conv_addr_trans2normal(i,7), conv_addr_trans2normal(i,5), aki1);

				aki0 = _mm256_shuffle_pd(ci0j2, ci1j2, 0);
				aki1 = _mm256_shuffle_pd(ci0j2, ci1j2, -1);
				_mm256_storeu2_m128d(conv_addr_trans2normal(i,10), conv_addr_trans2normal(i,8), aki0);
				_mm256_storeu2_m128d(conv_addr_trans2normal(i,11), conv_addr_trans2normal(i,9), aki1);

				aki0 = _mm256_shuffle_pd(ci0j3, ci1j3, 0);
				aki1 = _mm256_shuffle_pd(ci0j3, ci1j3, -1);
				_mm256_storeu2_m128d(conv_addr_trans2normal(i,14), conv_addr_trans2normal(i,12), aki0);
				_mm256_storeu2_m128d(conv_addr_trans2normal(i,15), conv_addr_trans2normal(i,13), aki1);

				aki0 = _mm256_shuffle_pd(ci0j4, ci1j4, 0);
				aki1 = _mm256_shuffle_pd(ci0j4, ci1j4, -1);
				_mm256_storeu2_m128d(conv_addr_trans2normal(i,18), conv_addr_trans2normal(i,16), aki0);
				_mm256_storeu2_m128d(conv_addr_trans2normal(i,19), conv_addr_trans2normal(i,17), aki1);

				aki0 = _mm256_shuffle_pd(ci0j5, ci1j5, 0);
				aki1 = _mm256_shuffle_pd(ci0j5, ci1j5, -1);
				switch(numj){
					case 24:
						_mm256_storeu2_m128d(conv_addr_trans2normal(i,22), conv_addr_trans2normal(i,20), aki0);
						_mm256_storeu2_m128d(conv_addr_trans2normal(i,23), conv_addr_trans2normal(i,21), aki1);
						break;
					case 23:
						mask = _mm256_set_epi64x(0, 0, -1, -1);
						_mm256_storeu2_m128d(conv_addr_trans2normal(i,22), conv_addr_trans2normal(i,20), aki0);
						_mm256_maskstore_pd(conv_addr_trans2normal(i,21), mask, aki1);
						mask = _mm256_set_epi32( 0, 0,-1,-1,-1,-1,-1,-1);	//reset mask
						break;
					case 22:
						mask = _mm256_set_epi64x(0, 0, -1, -1);
						_mm256_maskstore_pd(conv_addr_trans2normal(i,20), mask, aki0);
						_mm256_maskstore_pd(conv_addr_trans2normal(i,21), mask, aki1);
						mask = _mm256_set_epi32( 0, 0, 0, 0,-1,-1,-1,-1);	//reset mask
						break;
					case 21:
						mask = _mm256_set_epi64x(0, 0, -1, -1);
						_mm256_maskstore_pd(conv_addr_trans2normal(i,20), mask, aki0);
						mask = _mm256_set_epi32( 0, 0, 0, 0, 0, 0,-1,-1);	//reset mask
						break;
				}
			}
		}
			
		if (numi&0x1) {
			const double* __restrict__ pbkj = b;
			const double* __restrict__ paki = a+numi-1;
			ci0j0 = _mm256_setzero_pd();
			ci0j1 = _mm256_setzero_pd();
			ci0j2 = _mm256_setzero_pd();
			ci0j3 = _mm256_setzero_pd();
			ci0j4 = _mm256_setzero_pd();
			ci0j5 = _mm256_setzero_pd();
			
			for (k=0; k<dimk; k++,pbkj+=dimj,paki+=dimi) {
				aki0 = _mm256_broadcast_sd(paki);
				
				bkj = _mm256_loadu_pd(pbkj   );
				ci0j0 = FMA(aki0, bkj, ci0j0);
				
				bkj = _mm256_loadu_pd(pbkj+ 4);
				ci0j1 = FMA(aki0, bkj, ci0j1);
				
				bkj = _mm256_loadu_pd(pbkj+ 8);
				ci0j2 = FMA(aki0, bkj, ci0j2);
				
				bkj = _mm256_loadu_pd(pbkj+12);
				ci0j3 = FMA(aki0, bkj, ci0j3);
				
				bkj = _mm256_loadu_pd(pbkj+16);
				ci0j4 = FMA(aki0, bkj, ci0j4);

				bkj = _mm256_maskload_pd(pbkj+20,mask);
				ci0j5 = FMA(aki0, bkj, ci0j5);
			}
			if(!is_trans){
				_mm256_storeu_pd(c+i*dimj   , ci0j0);
				_mm256_storeu_pd(c+i*dimj+ 4, ci0j1);
				_mm256_storeu_pd(c+i*dimj+ 8, ci0j2);
				_mm256_storeu_pd(c+i*dimj+12, ci0j3);
				_mm256_storeu_pd(c+i*dimj+16, ci0j4);
				_mm256_maskstore_pd(c+i*dimj+20, mask, ci0j5);
			}else{
				_mm256_storeu_pd(tmp, ci0j0);
				c[0 *dimi+i] = tmp[0];
				c[1 *dimi+i] = tmp[1];
				c[2 *dimi+i] = tmp[2];
				c[3 *dimi+i] = tmp[3];

				_mm256_storeu_pd(tmp, ci0j1);
				c[4 *dimi+i] = tmp[0];
				c[5 *dimi+i] = tmp[1];
				c[6 *dimi+i] = tmp[2];
				c[7 *dimi+i] = tmp[3];

				_mm256_storeu_pd(tmp, ci0j2);
				c[8 *dimi+i] = tmp[0];
				c[9 *dimi+i] = tmp[1];
				c[10*dimi+i] = tmp[2];
				c[11*dimi+i] = tmp[3];

				_mm256_storeu_pd(tmp, ci0j3);
				c[12*dimi+i] = tmp[0];
				c[13*dimi+i] = tmp[1];
				c[14*dimi+i] = tmp[2];
				c[15*dimi+i] = tmp[3];

				_mm256_storeu_pd(tmp, ci0j4);
				c[16*dimi+i] = tmp[0];
				c[17*dimi+i] = tmp[1];
				c[18*dimi+i] = tmp[2];
				c[19*dimi+i] = tmp[3];

				_mm256_storeu_pd(tmp, ci0j5);
				switch(numj){
				case 24:
					c[23*dimi+i] = tmp[3];
				case 23:
					c[22*dimi+i] = tmp[2];
				case 22:
					c[21*dimi+i] = tmp[1];
				case 21:
					c[20*dimi+i] = tmp[0];
				}
			}
		}

		break;

	case 20:
	case 19:
	case 18:
	case 17:
		if      (numj == 20) mask = _mm256_set_epi32(-1,-1,-1,-1,-1,-1,-1,-1);
		else if (numj == 19) mask = _mm256_set_epi32( 0, 0,-1,-1,-1,-1,-1,-1);
		else if (numj == 18) mask = _mm256_set_epi32( 0, 0, 0, 0,-1,-1,-1,-1);
		else if (numj == 17) mask = _mm256_set_epi32( 0, 0, 0, 0, 0, 0,-1,-1);

		for (i=0; i<dimi2; i+=2,ci+=dimj2) {
			const double* __restrict__ pbkj = b;
			const double* __restrict__ paki = a+i;
			ci0j0 = _mm256_setzero_pd();
			ci0j1 = _mm256_setzero_pd();
			ci0j2 = _mm256_setzero_pd();
			ci0j3 = _mm256_setzero_pd();
			ci0j4 = _mm256_setzero_pd();
			
			ci1j0 = _mm256_setzero_pd();
			ci1j1 = _mm256_setzero_pd();
			ci1j2 = _mm256_setzero_pd();
			ci1j3 = _mm256_setzero_pd();
			ci1j4 = _mm256_setzero_pd();
			
			for (k=0; k<dimk; k++,pbkj+=dimj,paki+=dimi) {
				aki0 = _mm256_broadcast_sd(paki);
				aki1 = _mm256_broadcast_sd(paki+1);
				
				bkj = _mm256_loadu_pd(pbkj   );
				ci0j0 = FMA(aki0, bkj, ci0j0);
				ci1j0 = FMA(aki1, bkj, ci1j0);
				
				bkj = _mm256_loadu_pd(pbkj+ 4);
				ci0j1 = FMA(aki0, bkj, ci0j1);
				ci1j1 = FMA(aki1, bkj, ci1j1);
				
				bkj = _mm256_loadu_pd(pbkj+ 8);
				ci0j2 = FMA(aki0, bkj, ci0j2);
				ci1j2 = FMA(aki1, bkj, ci1j2);
				
				bkj = _mm256_loadu_pd(pbkj+12);
				ci0j3 = FMA(aki0, bkj, ci0j3);
				ci1j3 = FMA(aki1, bkj, ci1j3);
				
				bkj = _mm256_maskload_pd(pbkj+16,mask);
				ci0j4 = FMA(aki0, bkj, ci0j4);
				ci1j4 = FMA(aki1, bkj, ci1j4);
			}
			if(!is_trans){
				_mm256_storeu_pd(c+i*dimj   , ci0j0);
				_mm256_storeu_pd(c+i*dimj+ 4, ci0j1);
				_mm256_storeu_pd(c+i*dimj+ 8, ci0j2);
				_mm256_storeu_pd(c+i*dimj+12, ci0j3);
				_mm256_maskstore_pd(c+i*dimj+16, mask, ci0j4);
				
				_mm256_storeu_pd(c+(i+1)*dimj   , ci1j0);
				_mm256_storeu_pd(c+(i+1)*dimj+ 4, ci1j1);
				_mm256_storeu_pd(c+(i+1)*dimj+ 8, ci1j2);
				_mm256_storeu_pd(c+(i+1)*dimj+12, ci1j3);
				_mm256_maskstore_pd(c+(i+1)*dimj+16, mask, ci1j4);
			}else{
				//temporary use aki0, aki1 and mask
				aki0 = _mm256_shuffle_pd(ci0j0, ci1j0, 0);
				aki1 = _mm256_shuffle_pd(ci0j0, ci1j0, -1);
				_mm256_storeu2_m128d(conv_addr_trans2normal(i,2), conv_addr_trans2normal(i,0), aki0);
				_mm256_storeu2_m128d(conv_addr_trans2normal(i,3), conv_addr_trans2normal(i,1), aki1);

				aki0 = _mm256_shuffle_pd(ci0j1, ci1j1, 0);
				aki1 = _mm256_shuffle_pd(ci0j1, ci1j1, -1);
				_mm256_storeu2_m128d(conv_addr_trans2normal(i,6), conv_addr_trans2normal(i,4), aki0);
				_mm256_storeu2_m128d(conv_addr_trans2normal(i,7), conv_addr_trans2normal(i,5), aki1);

				aki0 = _mm256_shuffle_pd(ci0j2, ci1j2, 0);
				aki1 = _mm256_shuffle_pd(ci0j2, ci1j2, -1);
				_mm256_storeu2_m128d(conv_addr_trans2normal(i,10), conv_addr_trans2normal(i,8), aki0);
				_mm256_storeu2_m128d(conv_addr_trans2normal(i,11), conv_addr_trans2normal(i,9), aki1);

				aki0 = _mm256_shuffle_pd(ci0j3, ci1j3, 0);
				aki1 = _mm256_shuffle_pd(ci0j3, ci1j3, -1);
				_mm256_storeu2_m128d(conv_addr_trans2normal(i,14), conv_addr_trans2normal(i,12), aki0);
				_mm256_storeu2_m128d(conv_addr_trans2normal(i,15), conv_addr_trans2normal(i,13), aki1);

				aki0 = _mm256_shuffle_pd(ci0j4, ci1j4, 0);
				aki1 = _mm256_shuffle_pd(ci0j4, ci1j4, -1);
				switch(numj){
					case 20:
						_mm256_storeu2_m128d(conv_addr_trans2normal(i,18), conv_addr_trans2normal(i,16), aki0);
						_mm256_storeu2_m128d(conv_addr_trans2normal(i,19), conv_addr_trans2normal(i,17), aki1);
						break;
					case 19:
						mask = _mm256_set_epi64x(0, 0, -1, -1);
						_mm256_storeu2_m128d(conv_addr_trans2normal(i,18), conv_addr_trans2normal(i,16), aki0);
						_mm256_maskstore_pd(conv_addr_trans2normal(i,17), mask, aki1);
						mask = _mm256_set_epi32( 0, 0,-1,-1,-1,-1,-1,-1);	//reset mask
						break;
					case 18:
						mask = _mm256_set_epi64x(0, 0, -1, -1);
						_mm256_maskstore_pd(conv_addr_trans2normal(i,16), mask, aki0);
						_mm256_maskstore_pd(conv_addr_trans2normal(i,17), mask, aki1);
						mask = _mm256_set_epi32( 0, 0, 0, 0,-1,-1,-1,-1);	//reset mask
						break;
					case 17:
						mask = _mm256_set_epi64x(0, 0, -1, -1);
						_mm256_maskstore_pd(conv_addr_trans2normal(i,16), mask, aki0);
						mask = _mm256_set_epi32( 0, 0, 0, 0, 0, 0,-1,-1);	//reset mask
						break;
				}
			}
		}
			
		if (numi&0x1) {
			const double* __restrict__ pbkj = b;
			const double* __restrict__ paki = a+numi-1;
			ci0j0 = _mm256_setzero_pd();
			ci0j1 = _mm256_setzero_pd();
			ci0j2 = _mm256_setzero_pd();
			ci0j3 = _mm256_setzero_pd();
			ci0j4 = _mm256_setzero_pd();
			
			for (k=0; k<dimk; k++,pbkj+=dimj,paki+=dimi) {
				aki0 = _mm256_broadcast_sd(paki);
				
				bkj = _mm256_loadu_pd(pbkj   );
				ci0j0 = FMA(aki0, bkj, ci0j0);
				
				bkj = _mm256_loadu_pd(pbkj+ 4);
				ci0j1 = FMA(aki0, bkj, ci0j1);
				
				bkj = _mm256_loadu_pd(pbkj+ 8);
				ci0j2 = FMA(aki0, bkj, ci0j2);
				
				bkj = _mm256_loadu_pd(pbkj+12);
				ci0j3 = FMA(aki0, bkj, ci0j3);
				
				bkj = _mm256_maskload_pd(pbkj+16,mask);
				ci0j4 = FMA(aki0, bkj, ci0j4);
			}

			if(!is_trans){
				_mm256_storeu_pd(c+i*dimj   , ci0j0);
				_mm256_storeu_pd(c+i*dimj+ 4, ci0j1);
				_mm256_storeu_pd(c+i*dimj+ 8, ci0j2);
				_mm256_storeu_pd(c+i*dimj+12, ci0j3);
				_mm256_maskstore_pd(c+i*dimj+16, mask, ci0j4);
			}else{
				_mm256_storeu_pd(tmp, ci0j0);
				c[0*dimi+i] = tmp[0];
				c[1*dimi+i] = tmp[1];
				c[2*dimi+i] = tmp[2];
				c[3*dimi+i] = tmp[3];

				_mm256_storeu_pd(tmp, ci0j1);
				c[4*dimi+i] = tmp[0];
				c[5*dimi+i] = tmp[1];
				c[6*dimi+i] = tmp[2];
				c[7*dimi+i] = tmp[3];

				_mm256_storeu_pd(tmp, ci0j2);
				c[8 *dimi+i] = tmp[0];
				c[9 *dimi+i] = tmp[1];
				c[10*dimi+i] = tmp[2];
				c[11*dimi+i] = tmp[3];

				_mm256_storeu_pd(tmp, ci0j3);
				c[12*dimi+i] = tmp[0];
				c[13*dimi+i] = tmp[1];
				c[14*dimi+i] = tmp[2];
				c[15*dimi+i] = tmp[3];

				_mm256_storeu_pd(tmp, ci0j4);
				switch(numj){
				case 20:
					c[19*dimi+i] = tmp[3];
				case 19:
					c[18*dimi+i] = tmp[2];
				case 18:
					c[17*dimi+i] = tmp[1];
				case 17:
					c[16*dimi+i] = tmp[0];
				}
			}
		}

		break;
		
	case 16:
	case 15:
	case 14:
	case 13:
		if      (numj == 16) mask = _mm256_set_epi32(-1,-1,-1,-1,-1,-1,-1,-1);
		else if (numj == 15) mask = _mm256_set_epi32( 0, 0,-1,-1,-1,-1,-1,-1);
		else if (numj == 14) mask = _mm256_set_epi32( 0, 0, 0, 0,-1,-1,-1,-1);
		else if (numj == 13) mask = _mm256_set_epi32( 0, 0, 0, 0, 0, 0,-1,-1);

		for (i=0; i<dimi2; i+=2,ci+=dimj2) {
			const double* __restrict__ pbkj = b;
			const double* __restrict__ paki = a+i;
			ci0j0 = _mm256_setzero_pd();
			ci0j1 = _mm256_setzero_pd();
			ci0j2 = _mm256_setzero_pd();
			ci0j3 = _mm256_setzero_pd();
			
			ci1j0 = _mm256_setzero_pd();
			ci1j1 = _mm256_setzero_pd();
			ci1j2 = _mm256_setzero_pd();
			ci1j3 = _mm256_setzero_pd();
			
			for (k=0; k<dimk; k++,pbkj+=dimj,paki+=dimi) {
				aki0 = _mm256_broadcast_sd(paki);
				aki1 = _mm256_broadcast_sd(paki+1);
				
				bkj = _mm256_loadu_pd(pbkj   );
				ci0j0 = FMA(aki0, bkj, ci0j0);
				ci1j0 = FMA(aki1, bkj, ci1j0);
				
				bkj = _mm256_loadu_pd(pbkj+ 4);
				ci0j1 = FMA(aki0, bkj, ci0j1);
				ci1j1 = FMA(aki1, bkj, ci1j1);
				
				bkj = _mm256_loadu_pd(pbkj+ 8);
				ci0j2 = FMA(aki0, bkj, ci0j2);
				ci1j2 = FMA(aki1, bkj, ci1j2);
				
				bkj = _mm256_maskload_pd(pbkj+12,mask);
				ci0j3 = FMA(aki0, bkj, ci0j3);
				ci1j3 = FMA(aki1, bkj, ci1j3);
			}
			if(!is_trans){
				_mm256_storeu_pd(c+i*dimj   , ci0j0);
				_mm256_storeu_pd(c+i*dimj+ 4, ci0j1);
				_mm256_storeu_pd(c+i*dimj+ 8, ci0j2);
				_mm256_maskstore_pd(c+i*dimj+12, mask, ci0j3); 
			   
				_mm256_storeu_pd(c+(i+1)*dimj   , ci1j0);
				_mm256_storeu_pd(c+(i+1)*dimj+ 4, ci1j1);
				_mm256_storeu_pd(c+(i+1)*dimj+ 8, ci1j2);
				_mm256_maskstore_pd(c+(i+1)*dimj+12, mask, ci1j3);
			}else{
				//temporary use aki0, aki1 and mask
				aki0 = _mm256_shuffle_pd(ci0j0, ci1j0, 0);
				aki1 = _mm256_shuffle_pd(ci0j0, ci1j0, -1);
				_mm256_storeu2_m128d(conv_addr_trans2normal(i,2), conv_addr_trans2normal(i,0), aki0);
				_mm256_storeu2_m128d(conv_addr_trans2normal(i,3), conv_addr_trans2normal(i,1), aki1);

				aki0 = _mm256_shuffle_pd(ci0j1, ci1j1, 0);
				aki1 = _mm256_shuffle_pd(ci0j1, ci1j1, -1);
				_mm256_storeu2_m128d(conv_addr_trans2normal(i,6), conv_addr_trans2normal(i,4), aki0);
				_mm256_storeu2_m128d(conv_addr_trans2normal(i,7), conv_addr_trans2normal(i,5), aki1);

				aki0 = _mm256_shuffle_pd(ci0j2, ci1j2, 0);
				aki1 = _mm256_shuffle_pd(ci0j2, ci1j2, -1);
				_mm256_storeu2_m128d(conv_addr_trans2normal(i,10), conv_addr_trans2normal(i,8), aki0);
				_mm256_storeu2_m128d(conv_addr_trans2normal(i,11), conv_addr_trans2normal(i,9), aki1);

				aki0 = _mm256_shuffle_pd(ci0j3, ci1j3, 0);
				aki1 = _mm256_shuffle_pd(ci0j3, ci1j3, -1);
				switch(numj){
					case 16:
						_mm256_storeu2_m128d(conv_addr_trans2normal(i,14), conv_addr_trans2normal(i,12), aki0);
						_mm256_storeu2_m128d(conv_addr_trans2normal(i,15), conv_addr_trans2normal(i,13), aki1);
						break;
					case 15:
						mask = _mm256_set_epi64x(0, 0, -1, -1);
						_mm256_storeu2_m128d(conv_addr_trans2normal(i,14), conv_addr_trans2normal(i,12), aki0);
						_mm256_maskstore_pd(conv_addr_trans2normal(i,13), mask, aki1);
						mask = _mm256_set_epi32( 0, 0,-1,-1,-1,-1,-1,-1);	//reset mask
						break;
					case 14:
						mask = _mm256_set_epi64x(0, 0, -1, -1);
						_mm256_maskstore_pd(conv_addr_trans2normal(i,12), mask, aki0);
						_mm256_maskstore_pd(conv_addr_trans2normal(i,13), mask, aki1);
						mask = _mm256_set_epi32( 0, 0, 0, 0,-1,-1,-1,-1);	//reset mask
						break;
					case 13:
						mask = _mm256_set_epi64x(0, 0, -1, -1);
						_mm256_maskstore_pd(conv_addr_trans2normal(i,12), mask, aki0);
						mask = _mm256_set_epi32( 0, 0, 0, 0, 0, 0,-1,-1);	//reset mask
						break;
				}
			}
		}
			
		if (numi&0x1) {
			const double* __restrict__ pbkj = b;
			const double* __restrict__ paki = a+numi-1;
			ci0j0 = _mm256_setzero_pd();
			ci0j1 = _mm256_setzero_pd();
			ci0j2 = _mm256_setzero_pd();
			ci0j3 = _mm256_setzero_pd();
			
			for (k=0; k<dimk; k++,pbkj+=dimj,paki+=dimi) {
				aki0 = _mm256_broadcast_sd(paki);
				
				bkj = _mm256_loadu_pd(pbkj   );
				ci0j0 = FMA(aki0, bkj, ci0j0);
				
				bkj = _mm256_loadu_pd(pbkj+ 4);
				ci0j1 = FMA(aki0, bkj, ci0j1);
				
				bkj = _mm256_loadu_pd(pbkj+ 8);
				ci0j2 = FMA(aki0, bkj, ci0j2);
				
				bkj = _mm256_maskload_pd(pbkj+12,mask);
				ci0j3 = FMA(aki0, bkj, ci0j3);
			}
			if(!is_trans){
				_mm256_storeu_pd(c+i*dimj   , ci0j0);
				_mm256_storeu_pd(c+i*dimj+ 4, ci0j1);
				_mm256_storeu_pd(c+i*dimj+ 8, ci0j2);
				_mm256_maskstore_pd(c+i*dimj+12, mask, ci0j3); 
			}else{
				_mm256_storeu_pd(tmp, ci0j0);
				c[0*dimi+i] = tmp[0];
				c[1*dimi+i] = tmp[1];
				c[2*dimi+i] = tmp[2];
				c[3*dimi+i] = tmp[3];

				_mm256_storeu_pd(tmp, ci0j1);
				c[ 4*dimi+i] = tmp[0];
				c[ 5*dimi+i] = tmp[1];
				c[ 6*dimi+i] = tmp[2];
				c[ 7*dimi+i] = tmp[3];

				_mm256_storeu_pd(tmp, ci0j2);
				c[ 8*dimi+i] = tmp[0];
				c[ 9*dimi+i] = tmp[1];
				c[10*dimi+i] = tmp[2];
				c[11*dimi+i] = tmp[3];

				_mm256_storeu_pd(tmp, ci0j3);
				switch(numj){
				case 16:
					c[15*dimi+i] = tmp[3];
				case 15:
					c[14*dimi+i] = tmp[2];
				case 14:
					c[13*dimi+i] = tmp[1];
				case 13:
					c[12*dimi+i] = tmp[0];
				}
			}
		}

		break;
		
	case 12:
	case 11:
	case 10:
	case  9:
		if      (numj == 12) mask = _mm256_set_epi32(-1,-1,-1,-1,-1,-1,-1,-1);
		else if (numj == 11) mask = _mm256_set_epi32( 0, 0,-1,-1,-1,-1,-1,-1);
		else if (numj == 10) mask = _mm256_set_epi32( 0, 0, 0, 0,-1,-1,-1,-1);
		else if (numj ==  9) mask = _mm256_set_epi32( 0, 0, 0, 0, 0, 0,-1,-1);

		for (i=0; i<dimi2; i+=2,ci+=dimj2) {
			const double* __restrict__ pbkj = b;
			const double* __restrict__ paki = a+i;
			ci0j0 = _mm256_setzero_pd();
			ci0j1 = _mm256_setzero_pd();
			ci0j2 = _mm256_setzero_pd();
			
			ci1j0 = _mm256_setzero_pd();
			ci1j1 = _mm256_setzero_pd();
			ci1j2 = _mm256_setzero_pd();
			
			for (k=0; k<dimk; k++,pbkj+=dimj,paki+=dimi) {
				aki0 = _mm256_broadcast_sd(paki);
				aki1 = _mm256_broadcast_sd(paki+1);
				
				bkj = _mm256_loadu_pd(pbkj   );
				ci0j0 = FMA(aki0, bkj, ci0j0);
				ci1j0 = FMA(aki1, bkj, ci1j0);
				
				bkj = _mm256_loadu_pd(pbkj+ 4);
				ci0j1 = FMA(aki0, bkj, ci0j1);
				ci1j1 = FMA(aki1, bkj, ci1j1);
				
				bkj = _mm256_maskload_pd(pbkj+8,mask);
				ci0j2 = FMA(aki0, bkj, ci0j2);
				ci1j2 = FMA(aki1, bkj, ci1j2);
			}
			if(!is_trans){
				_mm256_storeu_pd(c+i*dimj   , ci0j0);
				_mm256_storeu_pd(c+i*dimj+ 4, ci0j1);
				_mm256_maskstore_pd(c+i*dimj+8, mask, ci0j2); 
				
				_mm256_storeu_pd(c+(i+1)*dimj   , ci1j0);
				_mm256_storeu_pd(c+(i+1)*dimj+ 4, ci1j1);
				_mm256_maskstore_pd(c+(i+1)*dimj+8, mask, ci1j2); 
			}else{
				//temporary use aki0, aki1 and mask
				aki0 = _mm256_shuffle_pd(ci0j0, ci1j0, 0);
				aki1 = _mm256_shuffle_pd(ci0j0, ci1j0, -1);
				_mm256_storeu2_m128d(conv_addr_trans2normal(i,2), conv_addr_trans2normal(i,0), aki0);
				_mm256_storeu2_m128d(conv_addr_trans2normal(i,3), conv_addr_trans2normal(i,1), aki1);

				aki0 = _mm256_shuffle_pd(ci0j1, ci1j1, 0);
				aki1 = _mm256_shuffle_pd(ci0j1, ci1j1, -1);
				_mm256_storeu2_m128d(conv_addr_trans2normal(i,6), conv_addr_trans2normal(i,4), aki0);
				_mm256_storeu2_m128d(conv_addr_trans2normal(i,7), conv_addr_trans2normal(i,5), aki1);

				aki0 = _mm256_shuffle_pd(ci0j2, ci1j2, 0);
				aki1 = _mm256_shuffle_pd(ci0j2, ci1j2, -1);
				switch(numj){
					case 12:
						_mm256_storeu2_m128d(conv_addr_trans2normal(i,10), conv_addr_trans2normal(i,8), aki0);
						_mm256_storeu2_m128d(conv_addr_trans2normal(i,11), conv_addr_trans2normal(i,9), aki1);
						break;
					case 11:
						mask = _mm256_set_epi64x(0, 0, -1, -1);
						_mm256_storeu2_m128d(conv_addr_trans2normal(i,10), conv_addr_trans2normal(i,8), aki0);
						_mm256_maskstore_pd(conv_addr_trans2normal(i,9), mask, aki1);
						mask = _mm256_set_epi32( 0, 0,-1,-1,-1,-1,-1,-1);	//reset mask
						break;
					case 10:
						mask = _mm256_set_epi64x(0, 0, -1, -1);
						_mm256_maskstore_pd(conv_addr_trans2normal(i,8), mask, aki0);
						_mm256_maskstore_pd(conv_addr_trans2normal(i,9), mask, aki1);
						mask = _mm256_set_epi32( 0, 0, 0, 0,-1,-1,-1,-1);	//reset mask
						break;
					case  9:
						mask = _mm256_set_epi64x(0, 0, -1, -1);
						_mm256_maskstore_pd(conv_addr_trans2normal(i,8), mask, aki0);
						mask = _mm256_set_epi32( 0, 0, 0, 0, 0, 0,-1,-1);	//reset mask
						break;
				}
			}
		}
			
		if (numi&0x1) {
			const double* __restrict__ pbkj = b;
			const double* __restrict__ paki = a+numi-1;
			ci0j0 = _mm256_setzero_pd();
			ci0j1 = _mm256_setzero_pd();
			ci0j2 = _mm256_setzero_pd();
			
			for (k=0; k<dimk; k++,pbkj+=dimj,paki+=dimi) {
				aki0 = _mm256_broadcast_sd(paki);
				
				bkj = _mm256_loadu_pd(pbkj   );
				ci0j0 = FMA(aki0, bkj, ci0j0);
				
				bkj = _mm256_loadu_pd(pbkj+ 4);
				ci0j1 = FMA(aki0, bkj, ci0j1);
				
				bkj = _mm256_maskload_pd(pbkj+8,mask);
				ci0j2 = FMA(aki0, bkj, ci0j2);
			}
			if(!is_trans){
				_mm256_storeu_pd(c+i*dimj   , ci0j0);
				_mm256_storeu_pd(c+i*dimj+ 4, ci0j1);
				_mm256_maskstore_pd(c+i*dimj+8, mask, ci0j2); 
			}else{
				_mm256_storeu_pd(tmp, ci0j0);
				c[0*dimi+i] = tmp[0];
				c[1*dimi+i] = tmp[1];
				c[2*dimi+i] = tmp[2];
				c[3*dimi+i] = tmp[3];

				_mm256_storeu_pd(tmp, ci0j1);
				c[4*dimi+i] = tmp[0];
				c[5*dimi+i] = tmp[1];
				c[6*dimi+i] = tmp[2];
				c[7*dimi+i] = tmp[3];

				_mm256_storeu_pd(tmp, ci0j2);
				switch(numj){
				case 12:
					c[11*dimi+i] = tmp[3];
				case 11:
					c[10*dimi+i] = tmp[2];
				case 10:
					c[9*dimi+i] = tmp[1];
				case  9:
					c[8*dimi+i] = tmp[0];
				}
			}
		}

		break;
		
	case 8:
	case 7:
	case 6:
	case 5:
		if      (numj == 8) mask = _mm256_set_epi32(-1,-1,-1,-1,-1,-1,-1,-1);
		else if (numj == 7) mask = _mm256_set_epi32( 0, 0,-1,-1,-1,-1,-1,-1);
		else if (numj == 6) mask = _mm256_set_epi32( 0, 0, 0, 0,-1,-1,-1,-1);
		else if (numj == 5) mask = _mm256_set_epi32( 0, 0, 0, 0, 0, 0,-1,-1);
		for (i=0; i<dimi2; i+=2,ci+=dimj2) {
			const double* __restrict__ pbkj = b;
			const double* __restrict__ paki = a+i;
			ci0j0 = _mm256_setzero_pd();
			ci0j1 = _mm256_setzero_pd();
			
			ci1j0 = _mm256_setzero_pd();
			ci1j1 = _mm256_setzero_pd();
			
			for (k=0; k<dimk; k++,pbkj+=dimj,paki+=dimi) {
				aki0 = _mm256_broadcast_sd(paki);
				aki1 = _mm256_broadcast_sd(paki+1);
				
				bkj = _mm256_loadu_pd(pbkj   );
				ci0j0 = FMA(aki0, bkj, ci0j0);
				ci1j0 = FMA(aki1, bkj, ci1j0);
				
				bkj = _mm256_maskload_pd(pbkj+4,mask);
				ci0j1 = FMA(aki0, bkj, ci0j1);
				ci1j1 = FMA(aki1, bkj, ci1j1);
			}
			if(!is_trans){
				_mm256_storeu_pd(c+i*dimj   , ci0j0);
				_mm256_maskstore_pd(c+i*dimj+4, mask, ci0j1); 
				
				_mm256_storeu_pd(c+(i+1)*dimj   , ci1j0);
				_mm256_maskstore_pd(c+(i+1)*dimj+4, mask, ci1j1); 
			}else{
				//temporary use aki0, aki1 and mask
				aki0 = _mm256_shuffle_pd(ci0j0, ci1j0, 0);
				aki1 = _mm256_shuffle_pd(ci0j0, ci1j0, -1);
				_mm256_storeu2_m128d(conv_addr_trans2normal(i,2), conv_addr_trans2normal(i,0), aki0);
				_mm256_storeu2_m128d(conv_addr_trans2normal(i,3), conv_addr_trans2normal(i,1), aki1);

				aki0 = _mm256_shuffle_pd(ci0j1, ci1j1, 0);
				aki1 = _mm256_shuffle_pd(ci0j1, ci1j1, -1);
				switch(numj){
					case  8:
						_mm256_storeu2_m128d(conv_addr_trans2normal(i,6), conv_addr_trans2normal(i,4), aki0);
						_mm256_storeu2_m128d(conv_addr_trans2normal(i,7), conv_addr_trans2normal(i,5), aki1);
						mask = _mm256_set_epi32(-1,-1,-1,-1,-1,-1,-1,-1);	//reset mask
						break;
					case  7:
						mask = _mm256_set_epi64x(0, 0, -1, -1);
						_mm256_storeu2_m128d(conv_addr_trans2normal(i,6), conv_addr_trans2normal(i,4), aki0);
						_mm256_maskstore_pd(conv_addr_trans2normal(i,5), mask, aki1);
						mask = _mm256_set_epi32( 0, 0,-1,-1,-1,-1,-1,-1);	//reset mask
						break;
					case  6:
						mask = _mm256_set_epi64x(0, 0, -1, -1);
						_mm256_maskstore_pd(conv_addr_trans2normal(i,4), mask, aki0);
						_mm256_maskstore_pd(conv_addr_trans2normal(i,5), mask, aki1);
						mask = _mm256_set_epi32( 0, 0, 0, 0,-1,-1,-1,-1);	//reset mask
						break;
					case  5:
						mask = _mm256_set_epi64x(0, 0, -1, -1);
						_mm256_maskstore_pd(conv_addr_trans2normal(i,4), mask, aki0);
						mask = _mm256_set_epi32( 0, 0, 0, 0, 0, 0,-1,-1);	//reset mask
						break;
				}
			}

		}
			
		if (numi&0x1) {
			const double* __restrict__ pbkj = b;
			const double* __restrict__ paki = a+numi-1;
			ci0j0 = _mm256_setzero_pd();
			ci0j1 = _mm256_setzero_pd();
			
			for (k=0; k<dimk; k++,pbkj+=dimj,paki+=dimi) {
				aki0 = _mm256_broadcast_sd(paki);
				
				bkj = _mm256_loadu_pd(pbkj   );
				ci0j0 = FMA(aki0, bkj, ci0j0);

				bkj = _mm256_maskload_pd(pbkj+4,mask);
				ci0j1 = FMA(aki0, bkj, ci0j1);
			}

			if(!is_trans){
				_mm256_storeu_pd(c+i*dimj   , ci0j0);
				_mm256_maskstore_pd(c+i*dimj+4, mask, ci0j1); 
			}else{
				_mm256_storeu_pd(tmp, ci0j0);
				c[0*dimi+i] = tmp[0];
				c[1*dimi+i] = tmp[1];
				c[2*dimi+i] = tmp[2];
				c[3*dimi+i] = tmp[3];

				_mm256_storeu_pd(tmp, ci0j1);
				switch(numj){
				case 8:
					c[7*dimi+i] = tmp[3];
				case 7:
					c[6*dimi+i] = tmp[2];
				case 6:
					c[5*dimi+i] = tmp[1];
				case 5:
					c[4*dimi+i] = tmp[0];
				}
			}
		}

		break;
		
	case 4:
	case 3:
	case 2:
	case 1:
		if      (numj == 4) mask = _mm256_set_epi32(-1,-1,-1,-1,-1,-1,-1,-1);
		else if (numj == 3) mask = _mm256_set_epi32( 0, 0,-1,-1,-1,-1,-1,-1);
		else if (numj == 2) mask = _mm256_set_epi32( 0, 0, 0, 0,-1,-1,-1,-1);
		else if (numj == 1) mask = _mm256_set_epi32( 0, 0, 0, 0, 0, 0,-1,-1);

		for (i=0; i<dimi2; i+=2,ci+=dimj2) {
			const double* __restrict__ pbkj = b;
			const double* __restrict__ paki = a+i;
			ci0j0 = _mm256_setzero_pd();
			ci1j0 = _mm256_setzero_pd();
			
			for (k=0; k<dimk; k++,pbkj+=dimj,paki+=dimi) {
				aki0 = _mm256_broadcast_sd(paki);
				aki1 = _mm256_broadcast_sd(paki+1);
				
				bkj = _mm256_maskload_pd(pbkj, mask);
				ci0j0 = FMA(aki0, bkj, ci0j0);
				ci1j0 = FMA(aki1, bkj, ci1j0);
			}
			if(!is_trans){
				_mm256_maskstore_pd(c+i*dimj    , mask, ci0j0);
				_mm256_maskstore_pd(c+(i+1)*dimj, mask, ci1j0);
			}else{
				//temporary use aki0, aki1 and mask
				aki0 = _mm256_shuffle_pd(ci0j0, ci1j0, 0);
				aki1 = _mm256_shuffle_pd(ci0j0, ci1j0, -1);
				switch(numj){
					case  4:
						_mm256_storeu2_m128d(conv_addr_trans2normal(i,2), conv_addr_trans2normal(i,0), aki0);
						_mm256_storeu2_m128d(conv_addr_trans2normal(i,3), conv_addr_trans2normal(i,1), aki1);
						break;
					case  3:
						mask = _mm256_set_epi64x(0, 0, -1, -1);
						_mm256_storeu2_m128d(conv_addr_trans2normal(i,2), conv_addr_trans2normal(i,0), aki0);
						_mm256_maskstore_pd(conv_addr_trans2normal(i,1), mask, aki1);
						mask = _mm256_set_epi32( 0, 0,-1,-1,-1,-1,-1,-1);	//reset mask
						break;
					case  2:

						mask = _mm256_set_epi64x(0, 0, -1, -1);
						_mm256_maskstore_pd(conv_addr_trans2normal(i,0), mask, aki0);
						_mm256_maskstore_pd(conv_addr_trans2normal(i,1), mask, aki1);
						mask = _mm256_set_epi32( 0, 0, 0, 0,-1,-1,-1,-1);	//reset mask
						break;
					case  1:
						mask = _mm256_set_epi64x(0, 0, -1, -1);
						_mm256_maskstore_pd(conv_addr_trans2normal(i,0), mask, aki0);
						mask = _mm256_set_epi32( 0, 0, 0, 0, 0, 0,-1,-1);	//reset mask
						break;
				}
			}
		}
			
		if (numi&0x1) {
			const double* __restrict__ pbkj = b;
			const double* __restrict__ paki = a+i;
			ci0j0 = _mm256_setzero_pd();
			for (k=0; k<dimk; k++,pbkj+=dimj,paki+=dimi) {
				aki0 = _mm256_broadcast_sd(paki);
				bkj = _mm256_maskload_pd(pbkj, mask);
				ci0j0 = FMA(aki0, bkj, ci0j0);
			}
			if(!is_trans){
				_mm256_maskstore_pd(c+i*dimj    , mask, ci0j0);
			}else{
				_mm256_storeu_pd(tmp, ci0j0);
				switch(numj){
				case 4:
					c[3*dimi+i] = tmp[3];
				case 3:
					c[2*dimi+i] = tmp[2];
				case 2:
					c[1*dimi+i] = tmp[1];
				case 1:
					c[0*dimi+i] = tmp[0];
				}
			}
		}
		break;

	default:
		/* for (i=0; i<dimi; i++) { */
		/*     for (k=0; k<dimk; k++) { */
		/*         double aki = a[k*dimi+i]; */
		/*         for (j=0; j<numj; j++) { */
		/*             c[i*dimj+j] += aki*b[k*dimj+j]; */
		/*         } */
		/*     } */
		/* } */
		printf("HOW DID WE GET HERE?\n");
		break;
	}
}

void mTxmq_core(bool is_trans, long dimi, long dimj, long dimk,
           double_complex * __restrict__ c0, const double * __restrict__ a0, const double_complex * __restrict__ b0, long numi, long numj) {
	
    int i, k;
    int dimi2 = (numi>>1)<<1;
    int dimj2 = dimj<<1;
	double tmp[4];

    __m256d ci0j0, ci0j1, ci0j2, ci0j3, ci0j4, ci0j5;
    __m256d ci1j0, ci1j1, ci1j2, ci1j3, ci1j4, ci1j5;
    __m256d aki0, aki1, bkj;
    __m256i mask = _mm256_set_epi32(0,0,-1,-1,-1,-1,-1,-1);
    // __m256d tmp; //temporary from aki*bkj
    
    
	double* a = (double*)a0;
	double* b = (double*)b0;
	double* c = (double*)c0;
	double* __restrict__ ci = (double*)c0;

	//convert pointer 
	const auto conv_addr_trans2normal_comp = [dimi, c](long i, long j){return c + dimi * j + i*2;};

	switch (numj) {
	case 24:
	case 23:
	case 22:
	case 21:
		if      (numj == 24) mask = _mm256_set_epi32(-1,-1,-1,-1,-1,-1,-1,-1);
		else if (numj == 23) mask = _mm256_set_epi32( 0, 0,-1,-1,-1,-1,-1,-1);
		else if (numj == 22) mask = _mm256_set_epi32( 0, 0, 0, 0,-1,-1,-1,-1);
		else if (numj == 21) mask = _mm256_set_epi32( 0, 0, 0, 0, 0, 0,-1,-1);

		for (i=0; i<dimi2; i+=2,ci+=dimj2) {
			const double* __restrict__ pbkj = b;
			const double* __restrict__ paki = a+i;
			ci0j0 = _mm256_setzero_pd();
			ci0j1 = _mm256_setzero_pd();
			ci0j2 = _mm256_setzero_pd();
			ci0j3 = _mm256_setzero_pd();
			ci0j4 = _mm256_setzero_pd();
			ci0j5 = _mm256_setzero_pd();
			
			ci1j0 = _mm256_setzero_pd();
			ci1j1 = _mm256_setzero_pd();
			ci1j2 = _mm256_setzero_pd();
			ci1j3 = _mm256_setzero_pd();
			ci1j4 = _mm256_setzero_pd();
			ci1j5 = _mm256_setzero_pd();

			for (k=0; k<dimk; k++,pbkj+=dimj,paki+=dimi) {
				aki0 = _mm256_broadcast_sd(paki);
				aki1 = _mm256_broadcast_sd(paki+1);
				
				bkj = _mm256_loadu_pd(pbkj   );
				ci0j0 = FMA(aki0, bkj, ci0j0);
				ci1j0 = FMA(aki1, bkj, ci1j0);
				
				bkj = _mm256_loadu_pd(pbkj+ 4);
				ci0j1 = FMA(aki0, bkj, ci0j1);
				ci1j1 = FMA(aki1, bkj, ci1j1);
				
				bkj = _mm256_loadu_pd(pbkj+ 8);
				ci0j2 = FMA(aki0, bkj, ci0j2);
				ci1j2 = FMA(aki1, bkj, ci1j2);
				
				bkj = _mm256_loadu_pd(pbkj+12);
				ci0j3 = FMA(aki0, bkj, ci0j3);
				ci1j3 = FMA(aki1, bkj, ci1j3);
				
				bkj = _mm256_loadu_pd(pbkj+16);
				ci0j4 = FMA(aki0, bkj, ci0j4);
				ci1j4 = FMA(aki1, bkj, ci1j4);

				bkj = _mm256_maskload_pd(pbkj+20,mask);
				ci0j5 = FMA(aki0, bkj, ci0j5);
				ci1j5 = FMA(aki1, bkj, ci1j5);
			}
			if(!is_trans){
				//is_trans must be true
				printf("(real * complex) mtxmq_core:HOW DID WE GET HERE?\n");
			}else{
				//temporary use aki0, aki1 and mask
				aki0 = _mm256_permute2f128_pd(ci0j0, ci1j0, 0x20);
				aki1 = _mm256_permute2f128_pd(ci0j0, ci1j0, 0x31);
				_mm256_storeu_pd(conv_addr_trans2normal_comp(i, 4-2), aki1);
				_mm256_storeu_pd(conv_addr_trans2normal_comp(i, 2-2), aki0);

				aki0 = _mm256_permute2f128_pd(ci0j1, ci1j1, 0x20);
				aki1 = _mm256_permute2f128_pd(ci0j1, ci1j1, 0x31);
				_mm256_storeu_pd(conv_addr_trans2normal_comp(i, 8-2), aki1);
				_mm256_storeu_pd(conv_addr_trans2normal_comp(i, 6-2), aki0);

				aki0 = _mm256_permute2f128_pd(ci0j2, ci1j2, 0x20);
				aki1 = _mm256_permute2f128_pd(ci0j2, ci1j2, 0x31);
				_mm256_storeu_pd(conv_addr_trans2normal_comp(i, 12-2), aki1);
				_mm256_storeu_pd(conv_addr_trans2normal_comp(i, 10-2), aki0);

				aki0 = _mm256_permute2f128_pd(ci0j3, ci1j3, 0x20);
				aki1 = _mm256_permute2f128_pd(ci0j3, ci1j3, 0x31);
				_mm256_storeu_pd(conv_addr_trans2normal_comp(i, 16-2), aki1);
				_mm256_storeu_pd(conv_addr_trans2normal_comp(i, 14-2), aki0);

				aki0 = _mm256_permute2f128_pd(ci0j4, ci1j4, 0x20);
				aki1 = _mm256_permute2f128_pd(ci0j4, ci1j4, 0x31);
				_mm256_storeu_pd(conv_addr_trans2normal_comp(i, 20-2), aki1);
				_mm256_storeu_pd(conv_addr_trans2normal_comp(i, 18-2), aki0);

				aki0 = _mm256_permute2f128_pd(ci0j5, ci1j5, 0x20);
				aki1 = _mm256_permute2f128_pd(ci0j5, ci1j5, 0x31);
				switch(numj){
					case  24:
						_mm256_storeu_pd(conv_addr_trans2normal_comp(i, 24-2), aki1);
					case  22:
						_mm256_storeu_pd(conv_addr_trans2normal_comp(i, 22-2), aki0);
						break;
				}
			}
		}
			
		if (numi&0x1) {
			const double* __restrict__ pbkj = b;
			const double* __restrict__ paki = a+numi-1;
			ci0j0 = _mm256_setzero_pd();
			ci0j1 = _mm256_setzero_pd();
			ci0j2 = _mm256_setzero_pd();
			ci0j3 = _mm256_setzero_pd();
			ci0j4 = _mm256_setzero_pd();
			ci0j5 = _mm256_setzero_pd();
			
			for (k=0; k<dimk; k++,pbkj+=dimj,paki+=dimi) {
				aki0 = _mm256_broadcast_sd(paki);
				
				bkj = _mm256_loadu_pd(pbkj   );
				ci0j0 = FMA(aki0, bkj, ci0j0);
				
				bkj = _mm256_loadu_pd(pbkj+ 4);
				ci0j1 = FMA(aki0, bkj, ci0j1);
				
				bkj = _mm256_loadu_pd(pbkj+ 8);
				ci0j2 = FMA(aki0, bkj, ci0j2);
				
				bkj = _mm256_loadu_pd(pbkj+12);
				ci0j3 = FMA(aki0, bkj, ci0j3);
				
				bkj = _mm256_loadu_pd(pbkj+16);
				ci0j4 = FMA(aki0, bkj, ci0j4);

				bkj = _mm256_maskload_pd(pbkj+20,mask);
				ci0j5 = FMA(aki0, bkj, ci0j5);
			}
			if(!is_trans){
				//is_trans must be true
				printf("(real * complex) mtxmq_core:HOW DID WE GET HERE?\n");
			}else{
				_mm256_storeu2_m128d(conv_addr_trans2normal_comp(i,4-2), conv_addr_trans2normal_comp(i, 2-2), ci0j0);
				_mm256_storeu2_m128d(conv_addr_trans2normal_comp(i,8-2), conv_addr_trans2normal_comp(i, 6-2), ci0j1);
				_mm256_storeu2_m128d(conv_addr_trans2normal_comp(i,12-2), conv_addr_trans2normal_comp(i, 10-2), ci0j2);
				_mm256_storeu2_m128d(conv_addr_trans2normal_comp(i,16-2), conv_addr_trans2normal_comp(i, 14-2), ci0j3);
				_mm256_storeu2_m128d(conv_addr_trans2normal_comp(i,20-2), conv_addr_trans2normal_comp(i, 18-2), ci0j4);

				switch(numj){
					case 24:
						_mm256_storeu2_m128d(conv_addr_trans2normal_comp(i,24-2), conv_addr_trans2normal_comp(i, 22-2), ci0j5);
						break;
					case 22:
						_mm256_maskstore_pd(conv_addr_trans2normal_comp(i,22-2), mask, ci0j5);
						break;
				}
			}
		}

		break;

	case 20:
	case 19:
	case 18:
	case 17:
		if      (numj == 20) mask = _mm256_set_epi32(-1,-1,-1,-1,-1,-1,-1,-1);
		else if (numj == 19) mask = _mm256_set_epi32( 0, 0,-1,-1,-1,-1,-1,-1);
		else if (numj == 18) mask = _mm256_set_epi32( 0, 0, 0, 0,-1,-1,-1,-1);
		else if (numj == 17) mask = _mm256_set_epi32( 0, 0, 0, 0, 0, 0,-1,-1);

		for (i=0; i<dimi2; i+=2,ci+=dimj2) {
			const double* __restrict__ pbkj = b;
			const double* __restrict__ paki = a+i;
			ci0j0 = _mm256_setzero_pd();
			ci0j1 = _mm256_setzero_pd();
			ci0j2 = _mm256_setzero_pd();
			ci0j3 = _mm256_setzero_pd();
			ci0j4 = _mm256_setzero_pd();
			
			ci1j0 = _mm256_setzero_pd();
			ci1j1 = _mm256_setzero_pd();
			ci1j2 = _mm256_setzero_pd();
			ci1j3 = _mm256_setzero_pd();
			ci1j4 = _mm256_setzero_pd();
			
			for (k=0; k<dimk; k++,pbkj+=dimj,paki+=dimi) {
				aki0 = _mm256_broadcast_sd(paki);
				aki1 = _mm256_broadcast_sd(paki+1);
				
				bkj = _mm256_loadu_pd(pbkj   );
				ci0j0 = FMA(aki0, bkj, ci0j0);
				ci1j0 = FMA(aki1, bkj, ci1j0);
				
				bkj = _mm256_loadu_pd(pbkj+ 4);
				ci0j1 = FMA(aki0, bkj, ci0j1);
				ci1j1 = FMA(aki1, bkj, ci1j1);
				
				bkj = _mm256_loadu_pd(pbkj+ 8);
				ci0j2 = FMA(aki0, bkj, ci0j2);
				ci1j2 = FMA(aki1, bkj, ci1j2);
				
				bkj = _mm256_loadu_pd(pbkj+12);
				ci0j3 = FMA(aki0, bkj, ci0j3);
				ci1j3 = FMA(aki1, bkj, ci1j3);
				
				bkj = _mm256_maskload_pd(pbkj+16,mask);
				ci0j4 = FMA(aki0, bkj, ci0j4);
				ci1j4 = FMA(aki1, bkj, ci1j4);
			}
			if(!is_trans){
				//is_trans must be true
				printf("(real * complex) mtxmq_core:HOW DID WE GET HERE?\n");
			}else{
				//temporary use aki0, aki1 and mask
				aki0 = _mm256_permute2f128_pd(ci0j0, ci1j0, 0x20);
				aki1 = _mm256_permute2f128_pd(ci0j0, ci1j0, 0x31);
				_mm256_storeu_pd(conv_addr_trans2normal_comp(i, 4-2), aki1);
				_mm256_storeu_pd(conv_addr_trans2normal_comp(i, 2-2), aki0);

				aki0 = _mm256_permute2f128_pd(ci0j1, ci1j1, 0x20);
				aki1 = _mm256_permute2f128_pd(ci0j1, ci1j1, 0x31);
				_mm256_storeu_pd(conv_addr_trans2normal_comp(i, 8-2), aki1);
				_mm256_storeu_pd(conv_addr_trans2normal_comp(i, 6-2), aki0);

				aki0 = _mm256_permute2f128_pd(ci0j2, ci1j2, 0x20);
				aki1 = _mm256_permute2f128_pd(ci0j2, ci1j2, 0x31);
				_mm256_storeu_pd(conv_addr_trans2normal_comp(i, 12-2), aki1);
				_mm256_storeu_pd(conv_addr_trans2normal_comp(i, 10-2), aki0);

				aki0 = _mm256_permute2f128_pd(ci0j3, ci1j3, 0x20);
				aki1 = _mm256_permute2f128_pd(ci0j3, ci1j3, 0x31);
				_mm256_storeu_pd(conv_addr_trans2normal_comp(i, 16-2), aki1);
				_mm256_storeu_pd(conv_addr_trans2normal_comp(i, 14-2), aki0);

				aki0 = _mm256_permute2f128_pd(ci0j4, ci1j4, 0x20);
				aki1 = _mm256_permute2f128_pd(ci0j4, ci1j4, 0x31);
				switch(numj){
					case  20:
						_mm256_storeu_pd(conv_addr_trans2normal_comp(i, 20-2), aki1);
					case  18:
						_mm256_storeu_pd(conv_addr_trans2normal_comp(i, 18-2), aki0);
						break;
				}
			}
		}
			
		if (numi&0x1) {
			const double* __restrict__ pbkj = b;
			const double* __restrict__ paki = a+numi-1;
			ci0j0 = _mm256_setzero_pd();
			ci0j1 = _mm256_setzero_pd();
			ci0j2 = _mm256_setzero_pd();
			ci0j3 = _mm256_setzero_pd();
			ci0j4 = _mm256_setzero_pd();
			
			for (k=0; k<dimk; k++,pbkj+=dimj,paki+=dimi) {
				aki0 = _mm256_broadcast_sd(paki);
				
				bkj = _mm256_loadu_pd(pbkj   );
				ci0j0 = FMA(aki0, bkj, ci0j0);
				
				bkj = _mm256_loadu_pd(pbkj+ 4);
				ci0j1 = FMA(aki0, bkj, ci0j1);
				
				bkj = _mm256_loadu_pd(pbkj+ 8);
				ci0j2 = FMA(aki0, bkj, ci0j2);
				
				bkj = _mm256_loadu_pd(pbkj+12);
				ci0j3 = FMA(aki0, bkj, ci0j3);
				
				bkj = _mm256_maskload_pd(pbkj+16,mask);
				ci0j4 = FMA(aki0, bkj, ci0j4);
			}

			if(!is_trans){
				//is_trans must be true
				printf("(real * complex) mtxmq_core:HOW DID WE GET HERE?\n");
			}else{
				_mm256_storeu2_m128d(conv_addr_trans2normal_comp(i,4-2), conv_addr_trans2normal_comp(i, 2-2), ci0j0);
				_mm256_storeu2_m128d(conv_addr_trans2normal_comp(i,8-2), conv_addr_trans2normal_comp(i, 6-2), ci0j1);
				_mm256_storeu2_m128d(conv_addr_trans2normal_comp(i,12-2), conv_addr_trans2normal_comp(i, 10-2), ci0j2);
				_mm256_storeu2_m128d(conv_addr_trans2normal_comp(i,16-2), conv_addr_trans2normal_comp(i, 14-2), ci0j3);

				switch(numj){
					case 20:
						_mm256_storeu2_m128d(conv_addr_trans2normal_comp(i,20-2), conv_addr_trans2normal_comp(i, 18-2), ci0j4);
						break;
					case 18:
						_mm256_maskstore_pd(conv_addr_trans2normal_comp(i,18-2), mask, ci0j4);
						break;
				}
			}
		}

		break;
		
	case 16:
	case 15:
	case 14:
	case 13:
		if      (numj == 16) mask = _mm256_set_epi32(-1,-1,-1,-1,-1,-1,-1,-1);
		else if (numj == 15) mask = _mm256_set_epi32( 0, 0,-1,-1,-1,-1,-1,-1);
		else if (numj == 14) mask = _mm256_set_epi32( 0, 0, 0, 0,-1,-1,-1,-1);
		else if (numj == 13) mask = _mm256_set_epi32( 0, 0, 0, 0, 0, 0,-1,-1);

		for (i=0; i<dimi2; i+=2,ci+=dimj2) {
			const double* __restrict__ pbkj = b;
			const double* __restrict__ paki = a+i;
			ci0j0 = _mm256_setzero_pd();
			ci0j1 = _mm256_setzero_pd();
			ci0j2 = _mm256_setzero_pd();
			ci0j3 = _mm256_setzero_pd();
			
			ci1j0 = _mm256_setzero_pd();
			ci1j1 = _mm256_setzero_pd();
			ci1j2 = _mm256_setzero_pd();
			ci1j3 = _mm256_setzero_pd();
			
			for (k=0; k<dimk; k++,pbkj+=dimj,paki+=dimi) {
				aki0 = _mm256_broadcast_sd(paki);
				aki1 = _mm256_broadcast_sd(paki+1);
				
				bkj = _mm256_loadu_pd(pbkj   );
				ci0j0 = FMA(aki0, bkj, ci0j0);
				ci1j0 = FMA(aki1, bkj, ci1j0);
				
				bkj = _mm256_loadu_pd(pbkj+ 4);
				ci0j1 = FMA(aki0, bkj, ci0j1);
				ci1j1 = FMA(aki1, bkj, ci1j1);
				
				bkj = _mm256_loadu_pd(pbkj+ 8);
				ci0j2 = FMA(aki0, bkj, ci0j2);
				ci1j2 = FMA(aki1, bkj, ci1j2);
				
				bkj = _mm256_maskload_pd(pbkj+12,mask);
				ci0j3 = FMA(aki0, bkj, ci0j3);
				ci1j3 = FMA(aki1, bkj, ci1j3);
			}
			if(!is_trans){
				//is_trans must be true
				printf("(real * complex) mtxmq_core:HOW DID WE GET HERE?\n");
			}else{
				//temporary use aki0, aki1 and mask
				aki0 = _mm256_permute2f128_pd(ci0j0, ci1j0, 0x20);
				aki1 = _mm256_permute2f128_pd(ci0j0, ci1j0, 0x31);
				_mm256_storeu_pd(conv_addr_trans2normal_comp(i, 4-2), aki1);
				_mm256_storeu_pd(conv_addr_trans2normal_comp(i, 2-2), aki0);

				aki0 = _mm256_permute2f128_pd(ci0j1, ci1j1, 0x20);
				aki1 = _mm256_permute2f128_pd(ci0j1, ci1j1, 0x31);
				_mm256_storeu_pd(conv_addr_trans2normal_comp(i, 8-2), aki1);
				_mm256_storeu_pd(conv_addr_trans2normal_comp(i, 6-2), aki0);

				aki0 = _mm256_permute2f128_pd(ci0j2, ci1j2, 0x20);
				aki1 = _mm256_permute2f128_pd(ci0j2, ci1j2, 0x31);
				_mm256_storeu_pd(conv_addr_trans2normal_comp(i, 12-2), aki1);
				_mm256_storeu_pd(conv_addr_trans2normal_comp(i, 10-2), aki0);

				aki0 = _mm256_permute2f128_pd(ci0j3, ci1j3, 0x20);
				aki1 = _mm256_permute2f128_pd(ci0j3, ci1j3, 0x31);
				switch(numj){
					case  16:
						_mm256_storeu_pd(conv_addr_trans2normal_comp(i, 16-2), aki1);
					case  14:
						_mm256_storeu_pd(conv_addr_trans2normal_comp(i, 14-2), aki0);
						break;
				}
			}
		}
			
		if (numi&0x1) {
			const double* __restrict__ pbkj = b;
			const double* __restrict__ paki = a+numi-1;
			ci0j0 = _mm256_setzero_pd();
			ci0j1 = _mm256_setzero_pd();
			ci0j2 = _mm256_setzero_pd();
			ci0j3 = _mm256_setzero_pd();
			
			for (k=0; k<dimk; k++,pbkj+=dimj,paki+=dimi) {
				aki0 = _mm256_broadcast_sd(paki);
				
				bkj = _mm256_loadu_pd(pbkj   );
				ci0j0 = FMA(aki0, bkj, ci0j0);
				
				bkj = _mm256_loadu_pd(pbkj+ 4);
				ci0j1 = FMA(aki0, bkj, ci0j1);
				
				bkj = _mm256_loadu_pd(pbkj+ 8);
				ci0j2 = FMA(aki0, bkj, ci0j2);
				
				bkj = _mm256_maskload_pd(pbkj+12,mask);
				ci0j3 = FMA(aki0, bkj, ci0j3);
			}
			if(!is_trans){
				//is_trans must be true
				printf("(real * complex) mtxmq_core:HOW DID WE GET HERE?\n");
			}else{
				_mm256_storeu2_m128d(conv_addr_trans2normal_comp(i,4-2), conv_addr_trans2normal_comp(i, 2-2), ci0j0);
				_mm256_storeu2_m128d(conv_addr_trans2normal_comp(i,8-2), conv_addr_trans2normal_comp(i, 6-2), ci0j1);
				_mm256_storeu2_m128d(conv_addr_trans2normal_comp(i,12-2), conv_addr_trans2normal_comp(i, 10-2), ci0j2);

				switch(numj){
					case 16:
						_mm256_storeu2_m128d(conv_addr_trans2normal_comp(i,16-2), conv_addr_trans2normal_comp(i, 14-2), ci0j3);
						break;
					case 14:
						_mm256_maskstore_pd(conv_addr_trans2normal_comp(i,14-2), mask, ci0j3);
						break;
				}
			}
		}

		break;
		
	case 12:
	case 11:
	case 10:
	case  9:
		if      (numj == 12) mask = _mm256_set_epi32(-1,-1,-1,-1,-1,-1,-1,-1);
		else if (numj == 11) mask = _mm256_set_epi32( 0, 0,-1,-1,-1,-1,-1,-1);
		else if (numj == 10) mask = _mm256_set_epi32( 0, 0, 0, 0,-1,-1,-1,-1);
		else if (numj ==  9) mask = _mm256_set_epi32( 0, 0, 0, 0, 0, 0,-1,-1);

		for (i=0; i<dimi2; i+=2,ci+=dimj2) {
			const double* __restrict__ pbkj = b;
			const double* __restrict__ paki = a+i;
			ci0j0 = _mm256_setzero_pd();
			ci0j1 = _mm256_setzero_pd();
			ci0j2 = _mm256_setzero_pd();
			
			ci1j0 = _mm256_setzero_pd();
			ci1j1 = _mm256_setzero_pd();
			ci1j2 = _mm256_setzero_pd();
			
			for (k=0; k<dimk; k++,pbkj+=dimj,paki+=dimi) {
				aki0 = _mm256_broadcast_sd(paki);
				aki1 = _mm256_broadcast_sd(paki+1);
				
				bkj = _mm256_loadu_pd(pbkj   );
				ci0j0 = FMA(aki0, bkj, ci0j0);
				ci1j0 = FMA(aki1, bkj, ci1j0);
				
				bkj = _mm256_loadu_pd(pbkj+ 4);
				ci0j1 = FMA(aki0, bkj, ci0j1);
				ci1j1 = FMA(aki1, bkj, ci1j1);
				
				bkj = _mm256_maskload_pd(pbkj+8,mask);
				ci0j2 = FMA(aki0, bkj, ci0j2);
				ci1j2 = FMA(aki1, bkj, ci1j2);
			}
			if(!is_trans){
				//is_trans must be true
				printf("(real * complex) mtxmq_core:HOW DID WE GET HERE?\n");
			}else{
				//temporary use aki0, aki1 and mask
				aki0 = _mm256_permute2f128_pd(ci0j0, ci1j0, 0x20);
				aki1 = _mm256_permute2f128_pd(ci0j0, ci1j0, 0x31);
				_mm256_storeu_pd(conv_addr_trans2normal_comp(i, 4-2), aki1);
				_mm256_storeu_pd(conv_addr_trans2normal_comp(i, 2-2), aki0);

				aki0 = _mm256_permute2f128_pd(ci0j1, ci1j1, 0x20);
				aki1 = _mm256_permute2f128_pd(ci0j1, ci1j1, 0x31);
				_mm256_storeu_pd(conv_addr_trans2normal_comp(i, 8-2), aki1);
				_mm256_storeu_pd(conv_addr_trans2normal_comp(i, 6-2), aki0);

				aki0 = _mm256_permute2f128_pd(ci0j2, ci1j2, 0x20);
				aki1 = _mm256_permute2f128_pd(ci0j2, ci1j2, 0x31);
				switch(numj){
					case  12:
						_mm256_storeu_pd(conv_addr_trans2normal_comp(i, 12-2), aki1);
					case  10:
						_mm256_storeu_pd(conv_addr_trans2normal_comp(i, 10-2), aki0);
						break;
				}
			}
		}
			
		if (numi&0x1) {
			const double* __restrict__ pbkj = b;
			const double* __restrict__ paki = a+numi-1;
			ci0j0 = _mm256_setzero_pd();
			ci0j1 = _mm256_setzero_pd();
			ci0j2 = _mm256_setzero_pd();
			
			for (k=0; k<dimk; k++,pbkj+=dimj,paki+=dimi) {
				aki0 = _mm256_broadcast_sd(paki);
				
				bkj = _mm256_loadu_pd(pbkj   );
				ci0j0 = FMA(aki0, bkj, ci0j0);
				
				bkj = _mm256_loadu_pd(pbkj+ 4);
				ci0j1 = FMA(aki0, bkj, ci0j1);
				
				bkj = _mm256_maskload_pd(pbkj+8,mask);
				ci0j2 = FMA(aki0, bkj, ci0j2);
			}
			if(!is_trans){
				//is_trans must be true
				printf("(real * complex) mtxmq_core:HOW DID WE GET HERE?\n");
			}else{
				_mm256_storeu2_m128d(conv_addr_trans2normal_comp(i,4-2), conv_addr_trans2normal_comp(i, 2-2), ci0j0);
				_mm256_storeu2_m128d(conv_addr_trans2normal_comp(i,8-2), conv_addr_trans2normal_comp(i, 6-2), ci0j1);

				switch(numj){
					case 12:
						_mm256_storeu2_m128d(conv_addr_trans2normal_comp(i,12-2), conv_addr_trans2normal_comp(i, 10-2), ci0j2);
						break;
					case 10:
						_mm256_maskstore_pd(conv_addr_trans2normal_comp(i,10-2), mask, ci0j2);
						break;
				}
			}
		}

		break;
		
	case 8:
	case 7:
	case 6:
	case 5:
		if      (numj == 8) mask = _mm256_set_epi32(-1,-1,-1,-1,-1,-1,-1,-1);
		else if (numj == 7) mask = _mm256_set_epi32( 0, 0,-1,-1,-1,-1,-1,-1);
		else if (numj == 6) mask = _mm256_set_epi32( 0, 0, 0, 0,-1,-1,-1,-1);
		else if (numj == 5) mask = _mm256_set_epi32( 0, 0, 0, 0, 0, 0,-1,-1);
		for (i=0; i<dimi2; i+=2,ci+=dimj2) {
			const double* __restrict__ pbkj = b;
			const double* __restrict__ paki = a+i;
			ci0j0 = _mm256_setzero_pd();
			ci0j1 = _mm256_setzero_pd();
			
			ci1j0 = _mm256_setzero_pd();
			ci1j1 = _mm256_setzero_pd();
			
			for (k=0; k<dimk; k++,pbkj+=dimj,paki+=dimi) {
				aki0 = _mm256_broadcast_sd(paki);
				aki1 = _mm256_broadcast_sd(paki+1);
				
				bkj = _mm256_loadu_pd(pbkj   );
				ci0j0 = FMA(aki0, bkj, ci0j0);
				ci1j0 = FMA(aki1, bkj, ci1j0);
				
				bkj = _mm256_maskload_pd(pbkj+4,mask);
				ci0j1 = FMA(aki0, bkj, ci0j1);
				ci1j1 = FMA(aki1, bkj, ci1j1);
			}
			if(!is_trans){
				//is_trans must be true
				printf("(real * complex) mtxmq_core:HOW DID WE GET HERE?\n");
			}else{
				//temporary use aki0, aki1 and mask
				aki0 = _mm256_permute2f128_pd(ci0j0, ci1j0, 0x20);
				aki1 = _mm256_permute2f128_pd(ci0j0, ci1j0, 0x31);
				_mm256_storeu_pd(conv_addr_trans2normal_comp(i, 4-2), aki1);
				_mm256_storeu_pd(conv_addr_trans2normal_comp(i, 2-2), aki0);

				aki0 = _mm256_permute2f128_pd(ci0j1, ci1j1, 0x20);
				aki1 = _mm256_permute2f128_pd(ci0j1, ci1j1, 0x31);
				switch(numj){
					case  8:
						_mm256_storeu_pd(conv_addr_trans2normal_comp(i, 8-2), aki1);
					case  6:
						_mm256_storeu_pd(conv_addr_trans2normal_comp(i, 6-2), aki0);
						break;
				}
			}
		}
			
		if (numi&0x1) {
			const double* __restrict__ pbkj = b;
			const double* __restrict__ paki = a+numi-1;
			ci0j0 = _mm256_setzero_pd();
			ci0j1 = _mm256_setzero_pd();
			
			for (k=0; k<dimk; k++,pbkj+=dimj,paki+=dimi) {
				aki0 = _mm256_broadcast_sd(paki);
				
				bkj = _mm256_loadu_pd(pbkj   );
				ci0j0 = FMA(aki0, bkj, ci0j0);

				bkj = _mm256_maskload_pd(pbkj+4,mask);
				ci0j1 = FMA(aki0, bkj, ci0j1);
			}

			if(!is_trans){
				//is_trans must be true
				printf("(real * complex) mtxmq_core:HOW DID WE GET HERE?\n");
			}else{
				_mm256_storeu2_m128d(conv_addr_trans2normal_comp(i,4-2), conv_addr_trans2normal_comp(i, 2-2), ci0j0);

				switch(numj){
					case 8:
						_mm256_storeu2_m128d(conv_addr_trans2normal_comp(i,8-2), conv_addr_trans2normal_comp(i, 6-2), ci0j1);
						break;
					case 6:
						_mm256_maskstore_pd(conv_addr_trans2normal_comp(i,6-2), mask, ci0j1);
						break;
				}
			}
		}

		break;
		
	case 4:
	case 3:
	case 2:
	case 1:
		if      (numj == 4) mask = _mm256_set_epi32(-1,-1,-1,-1,-1,-1,-1,-1);
		else if (numj == 3) mask = _mm256_set_epi32( 0, 0,-1,-1,-1,-1,-1,-1);
		else if (numj == 2) mask = _mm256_set_epi32( 0, 0, 0, 0,-1,-1,-1,-1);
		else if (numj == 1) mask = _mm256_set_epi32( 0, 0, 0, 0, 0, 0,-1,-1);

		for (i=0; i<dimi2; i+=2,ci+=dimj2) {
			const double* __restrict__ pbkj = b;
			const double* __restrict__ paki = a+i;
			ci0j0 = _mm256_setzero_pd();
			ci1j0 = _mm256_setzero_pd();
			
			for (k=0; k<dimk; k++,pbkj+=dimj,paki+=dimi) {
				aki0 = _mm256_broadcast_sd(paki);
				aki1 = _mm256_broadcast_sd(paki+1);
				
				bkj = _mm256_maskload_pd(pbkj, mask);
				ci0j0 = FMA(aki0, bkj, ci0j0);
				ci1j0 = FMA(aki1, bkj, ci1j0);
			}
			if(!is_trans){
				//is_trans must be true
				printf("(real * complex) mtxmq_core:HOW DID WE GET HERE?\n");
			}else{
				//temporary use aki0, aki1 and mask
				aki0 = _mm256_permute2f128_pd(ci0j0, ci1j0, 0x20);
				aki1 = _mm256_permute2f128_pd(ci0j0, ci1j0, 0x31);
				switch(numj){
					case  4:
						_mm256_storeu_pd(conv_addr_trans2normal_comp(i, 4-2), aki1);
					case  2:
						_mm256_storeu_pd(conv_addr_trans2normal_comp(i, 2-2), aki0);
						break;
				}
			}
		}
			
		if (numi&0x1) {
			const double* __restrict__ pbkj = b;
			const double* __restrict__ paki = a+i;
			ci0j0 = _mm256_setzero_pd();
			for (k=0; k<dimk; k++,pbkj+=dimj,paki+=dimi) {
				aki0 = _mm256_broadcast_sd(paki);
				bkj = _mm256_maskload_pd(pbkj, mask);
				ci0j0 = FMA(aki0, bkj, ci0j0);
			}
			if(!is_trans){
				//is_trans must be true
				printf("(real * complex) mtxmq_core:HOW DID WE GET HERE?\n");
			}else{

				switch(numj){
				case 4:
					_mm256_storeu2_m128d(conv_addr_trans2normal_comp(i,4-2), conv_addr_trans2normal_comp(i, 2-2), ci0j0);
					break;
				case 2:
					_mm256_maskstore_pd(conv_addr_trans2normal_comp(i,2-2), mask, ci0j0);
					break;
				}
			}
		}
		break;

	default:
		/* for (i=0; i<dimi; i++) { */
		/*     for (k=0; k<dimk; k++) { */
		/*         double aki = a[k*dimi+i]; */
		/*         for (j=0; j<numj; j++) { */
		/*             c[i*dimj+j] += aki*b[k*dimj+j]; */
		/*         } */
		/*     } */
		/* } */
		printf("HOW DID WE GET HERE?\n");
		break;
	}
}

void mTxmq_core(bool is_trans, long dimi, long dimj, long dimk,
           double_complex * __restrict__ c0, const double_complex * __restrict__ a0, const double * __restrict__ b0, long numi, long numj) {
	//elements are complex number

    int i, k;
    int dimi2 = (numi>>1)<<1;
    int dimj2 = dimj<<1;
	double tmp[4];

    __m256d ci0j0, ci0j1, ci0j2, ci0j3, ci0j4, ci0j5;
    __m256d ci1j0, ci1j1, ci1j2, ci1j3, ci1j4, ci1j5;
    __m256d aki0, aki1, bkj;
    __m256i mask = _mm256_set_epi32(0,0,-1,-1,-1,-1,-1,-1);
    // __m256d tmp; //temporary from aki*bkj
    
    
	double* a = (double*)a0;
	double* b = (double*)b0;
	double* c = (double*)c0;
	double* __restrict__ ci = (double*)c0;

	switch (numj) {
	case 24:
	case 23:
	case 22:
	case 21:
		if      (numj == 24) mask = _mm256_set_epi32(-1,-1,-1,-1,-1,-1,-1,-1);
		else if (numj == 23) mask = _mm256_set_epi32( 0, 0,-1,-1,-1,-1,-1,-1);
		else if (numj == 22) mask = _mm256_set_epi32( 0, 0, 0, 0,-1,-1,-1,-1);
		else if (numj == 21) mask = _mm256_set_epi32( 0, 0, 0, 0, 0, 0,-1,-1);

		for (i=0; i<dimi2; i+=2,ci+=dimj2) {
			const double* __restrict__ pbkj = b;
			const double* __restrict__ paki = a+i;
			ci0j0 = _mm256_setzero_pd();
			ci0j1 = _mm256_setzero_pd();
			ci0j2 = _mm256_setzero_pd();
			ci0j3 = _mm256_setzero_pd();
			ci0j4 = _mm256_setzero_pd();
			ci0j5 = _mm256_setzero_pd();
			
			ci1j0 = _mm256_setzero_pd();
			ci1j1 = _mm256_setzero_pd();
			ci1j2 = _mm256_setzero_pd();
			ci1j3 = _mm256_setzero_pd();
			ci1j4 = _mm256_setzero_pd();
			ci1j5 = _mm256_setzero_pd();

			for (k=0; k<dimk; k++,pbkj+=dimj,paki+=dimi) {
				aki0 = _mm256_broadcast_sd(paki);
				aki1 = _mm256_broadcast_sd(paki+1);
				
				bkj = _mm256_loadu_pd(pbkj   );
				ci0j0 = FMA(aki0, bkj, ci0j0);
				ci1j0 = FMA(aki1, bkj, ci1j0);
				
				bkj = _mm256_loadu_pd(pbkj+ 4);
				ci0j1 = FMA(aki0, bkj, ci0j1);
				ci1j1 = FMA(aki1, bkj, ci1j1);
				
				bkj = _mm256_loadu_pd(pbkj+ 8);
				ci0j2 = FMA(aki0, bkj, ci0j2);
				ci1j2 = FMA(aki1, bkj, ci1j2);
				
				bkj = _mm256_loadu_pd(pbkj+12);
				ci0j3 = FMA(aki0, bkj, ci0j3);
				ci1j3 = FMA(aki1, bkj, ci1j3);
				
				bkj = _mm256_loadu_pd(pbkj+16);
				ci0j4 = FMA(aki0, bkj, ci0j4);
				ci1j4 = FMA(aki1, bkj, ci1j4);

				bkj = _mm256_maskload_pd(pbkj+20,mask);
				ci0j5 = FMA(aki0, bkj, ci0j5);
				ci1j5 = FMA(aki1, bkj, ci1j5);
			}
			if(!is_trans){
				bkj = _mm256_shuffle_pd(ci0j0, ci1j0, 0);
				aki1 = _mm256_shuffle_pd(ci0j0, ci1j0, -1);
				aki0 = _mm256_permute2f128_pd(bkj, aki1, 0x20);
				aki1 = _mm256_permute2f128_pd(bkj, aki1, 0x31);
				_mm256_storeu_pd(c+i*dimj   , aki0);
				_mm256_storeu_pd(c+i*dimj+ 4, aki1);

				bkj = _mm256_shuffle_pd(ci0j1, ci1j1, 0);
				aki1 = _mm256_shuffle_pd(ci0j1, ci1j1, -1);
				aki0 = _mm256_permute2f128_pd(bkj, aki1, 0x20);
				aki1 = _mm256_permute2f128_pd(bkj, aki1, 0x31);
				_mm256_storeu_pd(c+i*dimj+ 8, aki0);
				_mm256_storeu_pd(c+i*dimj+12, aki1);

				bkj = _mm256_shuffle_pd(ci0j2, ci1j2, 0);
				aki1 = _mm256_shuffle_pd(ci0j2, ci1j2, -1);
				aki0 = _mm256_permute2f128_pd(bkj, aki1, 0x20);
				aki1 = _mm256_permute2f128_pd(bkj, aki1, 0x31);
				_mm256_storeu_pd(c+i*dimj+16, aki0);
				_mm256_storeu_pd(c+i*dimj+20, aki1);

				bkj = _mm256_shuffle_pd(ci0j3, ci1j3, 0);
				aki1 = _mm256_shuffle_pd(ci0j3, ci1j3, -1);
				aki0 = _mm256_permute2f128_pd(bkj, aki1, 0x20);
				aki1 = _mm256_permute2f128_pd(bkj, aki1, 0x31);
				_mm256_storeu_pd(c+i*dimj+24, aki0);
				_mm256_storeu_pd(c+i*dimj+28, aki1);

				bkj = _mm256_shuffle_pd(ci0j4, ci1j4, 0);
				aki1 = _mm256_shuffle_pd(ci0j4, ci1j4, -1);
				aki0 = _mm256_permute2f128_pd(bkj, aki1, 0x20);
				aki1 = _mm256_permute2f128_pd(bkj, aki1, 0x31);
				_mm256_storeu_pd(c+i*dimj+32, aki0);
				_mm256_storeu_pd(c+i*dimj+36, aki1);

				bkj = _mm256_shuffle_pd(ci0j5, ci1j5, 0);
				aki1 = _mm256_shuffle_pd(ci0j5, ci1j5, -1);
				aki0 = _mm256_permute2f128_pd(bkj, aki1, 0x20);
				aki1 = _mm256_permute2f128_pd(bkj, aki1, 0x31);
				switch(numj){
					case 24:
						_mm256_storeu_pd(c+i*dimj+40 , aki0);
						_mm256_storeu_pd(c+i*dimj+44 , aki1);
						break;
					case 23:
						mask = _mm256_set_epi64x(0, 0, -1, -1);
						_mm256_storeu_pd(c+i*dimj+40 , aki0);
						_mm256_maskstore_pd(c+i*dimj+44, mask, aki1);
						mask = _mm256_set_epi32( 0, 0,-1,-1,-1,-1,-1,-1);	
						break;
					case 22:
						_mm256_storeu_pd(c+i*dimj+40 , aki0);
						break;
					case 21:
						mask = _mm256_set_epi64x(0, 0, -1, -1);
						_mm256_maskstore_pd(c+i*dimj+40, mask, aki0);
						mask = _mm256_set_epi32( 0, 0, 0, 0, 0, 0,-1,-1);	
						break;
				}
			}else{
				// is_trans must be true
				printf("(complex * real) mtxmq_core:HOW DID WE GET HERE?\n");
			}
		}
			
		break;

	case 20:
	case 19:
	case 18:
	case 17:
		if      (numj == 20) mask = _mm256_set_epi32(-1,-1,-1,-1,-1,-1,-1,-1);
		else if (numj == 19) mask = _mm256_set_epi32( 0, 0,-1,-1,-1,-1,-1,-1);
		else if (numj == 18) mask = _mm256_set_epi32( 0, 0, 0, 0,-1,-1,-1,-1);
		else if (numj == 17) mask = _mm256_set_epi32( 0, 0, 0, 0, 0, 0,-1,-1);

		for (i=0; i<dimi2; i+=2,ci+=dimj2) {
			const double* __restrict__ pbkj = b;
			const double* __restrict__ paki = a+i;
			ci0j0 = _mm256_setzero_pd();
			ci0j1 = _mm256_setzero_pd();
			ci0j2 = _mm256_setzero_pd();
			ci0j3 = _mm256_setzero_pd();
			ci0j4 = _mm256_setzero_pd();
			
			ci1j0 = _mm256_setzero_pd();
			ci1j1 = _mm256_setzero_pd();
			ci1j2 = _mm256_setzero_pd();
			ci1j3 = _mm256_setzero_pd();
			ci1j4 = _mm256_setzero_pd();
			
			for (k=0; k<dimk; k++,pbkj+=dimj,paki+=dimi) {
				aki0 = _mm256_broadcast_sd(paki);
				aki1 = _mm256_broadcast_sd(paki+1);
				
				bkj = _mm256_loadu_pd(pbkj   );
				ci0j0 = FMA(aki0, bkj, ci0j0);
				ci1j0 = FMA(aki1, bkj, ci1j0);
				
				bkj = _mm256_loadu_pd(pbkj+ 4);
				ci0j1 = FMA(aki0, bkj, ci0j1);
				ci1j1 = FMA(aki1, bkj, ci1j1);
				
				bkj = _mm256_loadu_pd(pbkj+ 8);
				ci0j2 = FMA(aki0, bkj, ci0j2);
				ci1j2 = FMA(aki1, bkj, ci1j2);
				
				bkj = _mm256_loadu_pd(pbkj+12);
				ci0j3 = FMA(aki0, bkj, ci0j3);
				ci1j3 = FMA(aki1, bkj, ci1j3);
				
				bkj = _mm256_maskload_pd(pbkj+16,mask);
				ci0j4 = FMA(aki0, bkj, ci0j4);
				ci1j4 = FMA(aki1, bkj, ci1j4);
			}
			if(!is_trans){
				bkj = _mm256_shuffle_pd(ci0j0, ci1j0, 0);
				aki1 = _mm256_shuffle_pd(ci0j0, ci1j0, -1);
				aki0 = _mm256_permute2f128_pd(bkj, aki1, 0x20);
				aki1 = _mm256_permute2f128_pd(bkj, aki1, 0x31);
				_mm256_storeu_pd(c+i*dimj   , aki0);
				_mm256_storeu_pd(c+i*dimj+ 4, aki1);

				bkj = _mm256_shuffle_pd(ci0j1, ci1j1, 0);
				aki1 = _mm256_shuffle_pd(ci0j1, ci1j1, -1);
				aki0 = _mm256_permute2f128_pd(bkj, aki1, 0x20);
				aki1 = _mm256_permute2f128_pd(bkj, aki1, 0x31);
				_mm256_storeu_pd(c+i*dimj+ 8, aki0);
				_mm256_storeu_pd(c+i*dimj+12, aki1);

				bkj = _mm256_shuffle_pd(ci0j2, ci1j2, 0);
				aki1 = _mm256_shuffle_pd(ci0j2, ci1j2, -1);
				aki0 = _mm256_permute2f128_pd(bkj, aki1, 0x20);
				aki1 = _mm256_permute2f128_pd(bkj, aki1, 0x31);
				_mm256_storeu_pd(c+i*dimj+16, aki0);
				_mm256_storeu_pd(c+i*dimj+20, aki1);

				bkj = _mm256_shuffle_pd(ci0j3, ci1j3, 0);
				aki1 = _mm256_shuffle_pd(ci0j3, ci1j3, -1);
				aki0 = _mm256_permute2f128_pd(bkj, aki1, 0x20);
				aki1 = _mm256_permute2f128_pd(bkj, aki1, 0x31);
				_mm256_storeu_pd(c+i*dimj+24, aki0);
				_mm256_storeu_pd(c+i*dimj+28, aki1);

				bkj = _mm256_shuffle_pd(ci0j4, ci1j4, 0);
				aki1 = _mm256_shuffle_pd(ci0j4, ci1j4, -1);
				aki0 = _mm256_permute2f128_pd(bkj, aki1, 0x20);
				aki1 = _mm256_permute2f128_pd(bkj, aki1, 0x31);

				switch(numj){
					case 20:
						_mm256_storeu_pd(c+i*dimj+32 , aki0);
						_mm256_storeu_pd(c+i*dimj+36 , aki1);
						break;
					case 19:
						mask = _mm256_set_epi64x(0, 0, -1, -1);
						_mm256_storeu_pd(c+i*dimj+32 , aki0);
						_mm256_maskstore_pd(c+i*dimj+36, mask, aki1);
						mask = _mm256_set_epi32( 0, 0,-1,-1,-1,-1,-1,-1);	
						break;
					case 18:
						_mm256_storeu_pd(c+i*dimj+32 , aki0);
						break;
					case 17:
						mask = _mm256_set_epi64x(0, 0, -1, -1);
						_mm256_maskstore_pd(c+i*dimj+32, mask, aki0);
						mask = _mm256_set_epi32( 0, 0, 0, 0, 0, 0,-1,-1);	
						break;
				}
			}else{
				// is_trans must be true
				printf("(complex * real) mtxmq_core:HOW DID WE GET HERE?\n");
			}
		}
			
		break;
		
	case 16:
	case 15:
	case 14:
	case 13:
		if      (numj == 16) mask = _mm256_set_epi32(-1,-1,-1,-1,-1,-1,-1,-1);
		else if (numj == 15) mask = _mm256_set_epi32( 0, 0,-1,-1,-1,-1,-1,-1);
		else if (numj == 14) mask = _mm256_set_epi32( 0, 0, 0, 0,-1,-1,-1,-1);
		else if (numj == 13) mask = _mm256_set_epi32( 0, 0, 0, 0, 0, 0,-1,-1);

		for (i=0; i<dimi2; i+=2,ci+=dimj2) {
			const double* __restrict__ pbkj = b;
			const double* __restrict__ paki = a+i;
			ci0j0 = _mm256_setzero_pd();
			ci0j1 = _mm256_setzero_pd();
			ci0j2 = _mm256_setzero_pd();
			ci0j3 = _mm256_setzero_pd();
			
			ci1j0 = _mm256_setzero_pd();
			ci1j1 = _mm256_setzero_pd();
			ci1j2 = _mm256_setzero_pd();
			ci1j3 = _mm256_setzero_pd();
			
			for (k=0; k<dimk; k++,pbkj+=dimj,paki+=dimi) {
				aki0 = _mm256_broadcast_sd(paki);
				aki1 = _mm256_broadcast_sd(paki+1);
				
				bkj = _mm256_loadu_pd(pbkj   );
				ci0j0 = FMA(aki0, bkj, ci0j0);
				ci1j0 = FMA(aki1, bkj, ci1j0);
				
				bkj = _mm256_loadu_pd(pbkj+ 4);
				ci0j1 = FMA(aki0, bkj, ci0j1);
				ci1j1 = FMA(aki1, bkj, ci1j1);
				
				bkj = _mm256_loadu_pd(pbkj+ 8);
				ci0j2 = FMA(aki0, bkj, ci0j2);
				ci1j2 = FMA(aki1, bkj, ci1j2);
				
				bkj = _mm256_maskload_pd(pbkj+12,mask);
				ci0j3 = FMA(aki0, bkj, ci0j3);
				ci1j3 = FMA(aki1, bkj, ci1j3);
			}
			if(!is_trans){
				bkj = _mm256_shuffle_pd(ci0j0, ci1j0, 0);
				aki1 = _mm256_shuffle_pd(ci0j0, ci1j0, -1);
				aki0 = _mm256_permute2f128_pd(bkj, aki1, 0x20);
				aki1 = _mm256_permute2f128_pd(bkj, aki1, 0x31);
				_mm256_storeu_pd(c+i*dimj   , aki0);
				_mm256_storeu_pd(c+i*dimj+ 4  , aki1);

				bkj = _mm256_shuffle_pd(ci0j1, ci1j1, 0);
				aki1 = _mm256_shuffle_pd(ci0j1, ci1j1, -1);
				aki0 = _mm256_permute2f128_pd(bkj, aki1, 0x20);
				aki1 = _mm256_permute2f128_pd(bkj, aki1, 0x31);
				_mm256_storeu_pd(c+i*dimj+ 8, aki0);
				_mm256_storeu_pd(c+i*dimj+12, aki1);

				bkj = _mm256_shuffle_pd(ci0j2, ci1j2, 0);
				aki1 = _mm256_shuffle_pd(ci0j2, ci1j2, -1);
				aki0 = _mm256_permute2f128_pd(bkj, aki1, 0x20);
				aki1 = _mm256_permute2f128_pd(bkj, aki1, 0x31);
				_mm256_storeu_pd(c+i*dimj+16, aki0);
				_mm256_storeu_pd(c+i*dimj+20, aki1);

				bkj = _mm256_shuffle_pd(ci0j3, ci1j3, 0);
				aki1 = _mm256_shuffle_pd(ci0j3, ci1j3, -1);
				aki0 = _mm256_permute2f128_pd(bkj, aki1, 0x20);
				aki1 = _mm256_permute2f128_pd(bkj, aki1, 0x31);

				switch(numj){
					case 16:
						_mm256_storeu_pd(c+i*dimj+24 , aki0);
						_mm256_storeu_pd(c+i*dimj+28 , aki1);
						break;
					case 15:
						mask = _mm256_set_epi64x(0, 0, -1, -1);
						_mm256_storeu_pd(c+i*dimj+24 , aki0);
						_mm256_maskstore_pd(c+i*dimj+28, mask, aki1);
						mask = _mm256_set_epi32( 0, 0,-1,-1,-1,-1,-1,-1);	
						break;
					case 14:
						_mm256_storeu_pd(c+i*dimj+24 , aki0);
						break;
					case 13:
						mask = _mm256_set_epi64x(0, 0, -1, -1);
						_mm256_maskstore_pd(c+i*dimj+24, mask, aki0);
						mask = _mm256_set_epi32( 0, 0, 0, 0, 0, 0,-1,-1);	
						break;
				}
			}else{
				// is_trans must be true
				printf("(complex * real) mtxmq_core:HOW DID WE GET HERE?\n");
			}
		}
			
		break;
		
	case 12:
	case 11:
	case 10:
	case  9:
		if      (numj == 12) mask = _mm256_set_epi32(-1,-1,-1,-1,-1,-1,-1,-1);
		else if (numj == 11) mask = _mm256_set_epi32( 0, 0,-1,-1,-1,-1,-1,-1);
		else if (numj == 10) mask = _mm256_set_epi32( 0, 0, 0, 0,-1,-1,-1,-1);
		else if (numj ==  9) mask = _mm256_set_epi32( 0, 0, 0, 0, 0, 0,-1,-1);

		for (i=0; i<dimi2; i+=2,ci+=dimj2) {
			const double* __restrict__ pbkj = b;
			const double* __restrict__ paki = a+i;
			ci0j0 = _mm256_setzero_pd();
			ci0j1 = _mm256_setzero_pd();
			ci0j2 = _mm256_setzero_pd();
			
			ci1j0 = _mm256_setzero_pd();
			ci1j1 = _mm256_setzero_pd();
			ci1j2 = _mm256_setzero_pd();
			
			for (k=0; k<dimk; k++,pbkj+=dimj,paki+=dimi) {
				aki0 = _mm256_broadcast_sd(paki);
				aki1 = _mm256_broadcast_sd(paki+1);
				
				bkj = _mm256_loadu_pd(pbkj   );
				ci0j0 = FMA(aki0, bkj, ci0j0);
				ci1j0 = FMA(aki1, bkj, ci1j0);
				
				bkj = _mm256_loadu_pd(pbkj+ 4);
				ci0j1 = FMA(aki0, bkj, ci0j1);
				ci1j1 = FMA(aki1, bkj, ci1j1);
				
				bkj = _mm256_maskload_pd(pbkj+8,mask);
				ci0j2 = FMA(aki0, bkj, ci0j2);
				ci1j2 = FMA(aki1, bkj, ci1j2);
			}
			if(!is_trans){
				bkj = _mm256_shuffle_pd(ci0j0, ci1j0, 0);
				aki1 = _mm256_shuffle_pd(ci0j0, ci1j0, -1);
				aki0 = _mm256_permute2f128_pd(bkj, aki1, 0x20);
				aki1 = _mm256_permute2f128_pd(bkj, aki1, 0x31);
				_mm256_storeu_pd(c+i*dimj   , aki0);
				_mm256_storeu_pd(c+i*dimj+ 4, aki1);

				bkj = _mm256_shuffle_pd(ci0j1, ci1j1, 0);
				aki1 = _mm256_shuffle_pd(ci0j1, ci1j1, -1);
				aki0 = _mm256_permute2f128_pd(bkj, aki1, 0x20);
				aki1 = _mm256_permute2f128_pd(bkj, aki1, 0x31);
				_mm256_storeu_pd(c+i*dimj+ 8, aki0);
				_mm256_storeu_pd(c+i*dimj+12, aki1);

				bkj = _mm256_shuffle_pd(ci0j2, ci1j2, 0);
				aki1 = _mm256_shuffle_pd(ci0j2, ci1j2, -1);
				aki0 = _mm256_permute2f128_pd(bkj, aki1, 0x20);
				aki1 = _mm256_permute2f128_pd(bkj, aki1, 0x31);

				switch(numj){
					case 12:
						_mm256_storeu_pd(c+i*dimj+16 , aki0);
						_mm256_storeu_pd(c+i*dimj+20 , aki1);
						break;
					case 11:
						mask = _mm256_set_epi64x(0, 0, -1, -1);
						_mm256_storeu_pd(c+i*dimj+16 , aki0);
						_mm256_maskstore_pd(c+i*dimj+20, mask, aki1);
						mask = _mm256_set_epi32( 0, 0,-1,-1,-1,-1,-1,-1);	
						break;
					case 10:
						_mm256_storeu_pd(c+i*dimj+16 , aki0);
						break;
					case  9:
						mask = _mm256_set_epi64x(0, 0, -1, -1);
						_mm256_maskstore_pd(c+i*dimj+16, mask, aki0);
						mask = _mm256_set_epi32( 0, 0, 0, 0, 0, 0,-1,-1);	
						break;
				}
			}else{
				// is_trans must be true
				printf("(complex * real) mtxmq_core:HOW DID WE GET HERE?\n");
			}
		}
			
		break;
		
	case 8:
	case 7:
	case 6:
	case 5:
		if      (numj == 8) mask = _mm256_set_epi32(-1,-1,-1,-1,-1,-1,-1,-1);
		else if (numj == 7) mask = _mm256_set_epi32( 0, 0,-1,-1,-1,-1,-1,-1);
		else if (numj == 6) mask = _mm256_set_epi32( 0, 0, 0, 0,-1,-1,-1,-1);
		else if (numj == 5) mask = _mm256_set_epi32( 0, 0, 0, 0, 0, 0,-1,-1);
		for (i=0; i<dimi2; i+=2,ci+=dimj2) {
			const double* __restrict__ pbkj = b;
			const double* __restrict__ paki = a+i;
			ci0j0 = _mm256_setzero_pd();
			ci0j1 = _mm256_setzero_pd();
			
			ci1j0 = _mm256_setzero_pd();
			ci1j1 = _mm256_setzero_pd();
			
			for (k=0; k<dimk; k++,pbkj+=dimj,paki+=dimi) {
				aki0 = _mm256_broadcast_sd(paki);
				aki1 = _mm256_broadcast_sd(paki+1);
				
				bkj = _mm256_loadu_pd(pbkj   );
				ci0j0 = FMA(aki0, bkj, ci0j0);
				ci1j0 = FMA(aki1, bkj, ci1j0);
				
				bkj = _mm256_maskload_pd(pbkj+4,mask);
				ci0j1 = FMA(aki0, bkj, ci0j1);
				ci1j1 = FMA(aki1, bkj, ci1j1);
			}
			if(!is_trans){

				bkj = _mm256_shuffle_pd(ci0j0, ci1j0, 0);
				aki1 = _mm256_shuffle_pd(ci0j0, ci1j0, -1);
				aki0 = _mm256_permute2f128_pd(bkj, aki1, 0x20);
				aki1 = _mm256_permute2f128_pd(bkj, aki1, 0x31);
				_mm256_storeu_pd(c+i*dimj   , aki0);
				_mm256_storeu_pd(c+i*dimj+4 , aki1);

				bkj = _mm256_shuffle_pd(ci0j1, ci1j1, 0);
				aki1 = _mm256_shuffle_pd(ci0j1, ci1j1, -1);
				aki0 = _mm256_permute2f128_pd(bkj, aki1, 0x20);
				aki1 = _mm256_permute2f128_pd(bkj, aki1, 0x31);

				switch(numj){
					case 8:
						_mm256_storeu_pd(c+i*dimj+ 8 , aki0);
						_mm256_storeu_pd(c+i*dimj+12 , aki1);
						break;
					case 7:
						mask = _mm256_set_epi64x(0, 0, -1, -1);
						_mm256_storeu_pd(c+i*dimj+ 8 , aki0);
						_mm256_maskstore_pd(c+i*dimj+12, mask, aki1);
						mask = _mm256_set_epi32( 0, 0,-1,-1,-1,-1,-1,-1);	
						break;
					case 6:
						_mm256_storeu_pd(c+i*dimj+ 8 , aki0);
						break;
					case 5:
						mask = _mm256_set_epi64x(0, 0, -1, -1);
						_mm256_maskstore_pd(c+i*dimj+ 8, mask, aki0);
						mask = _mm256_set_epi32( 0, 0, 0, 0, 0, 0,-1,-1);	
						break;
				}
			}else{
				// is_trans must be true
				printf("(complex * real) mtxmq_core:HOW DID WE GET HERE?\n");
			}

		}
			
		break;
		
	case 4:
	case 3:
	case 2:
	case 1:
		if      (numj == 4) mask = _mm256_set_epi32(-1,-1,-1,-1,-1,-1,-1,-1);
		else if (numj == 3) mask = _mm256_set_epi32( 0, 0,-1,-1,-1,-1,-1,-1);
		else if (numj == 2) mask = _mm256_set_epi32( 0, 0, 0, 0,-1,-1,-1,-1);
		else if (numj == 1) mask = _mm256_set_epi32( 0, 0, 0, 0, 0, 0,-1,-1);

		for (i=0; i<dimi2; i+=2,ci+=dimj2) {
			const double* __restrict__ pbkj = b;
			const double* __restrict__ paki = a+i;
			ci0j0 = _mm256_setzero_pd();
			ci1j0 = _mm256_setzero_pd();
			
			for (k=0; k<dimk; k++,pbkj+=dimj,paki+=dimi) {
				aki0 = _mm256_broadcast_sd(paki);
				aki1 = _mm256_broadcast_sd(paki+1);
				
				bkj = _mm256_maskload_pd(pbkj, mask);
				ci0j0 = FMA(aki0, bkj, ci0j0);
				ci1j0 = FMA(aki1, bkj, ci1j0);
			}
			if(!is_trans){
				bkj = _mm256_shuffle_pd(ci0j0, ci1j0, 0);
				aki1 = _mm256_shuffle_pd(ci0j0, ci1j0, -1);
				aki0 = _mm256_permute2f128_pd(bkj, aki1, 0x20);
				aki1 = _mm256_permute2f128_pd(bkj, aki1, 0x31);
				switch(numj){
					case 4:
						_mm256_storeu_pd(c+i*dimj   , aki0);
						_mm256_storeu_pd(c+i*dimj+4 , aki1);
						break;
					case 3:
						mask = _mm256_set_epi64x(0, 0, -1, -1);
						_mm256_storeu_pd(c+i*dimj   , aki0);
						_mm256_maskstore_pd(c+i*dimj+4, mask, aki1);
						mask = _mm256_set_epi32( 0, 0,-1,-1,-1,-1,-1,-1);	
						break;
					case 2:
						_mm256_storeu_pd(c+i*dimj   , aki0);
						break;
					case 1:
						mask = _mm256_set_epi64x(0, 0, -1, -1);
						_mm256_maskstore_pd(c+i*dimj, mask, aki0);
						mask = _mm256_set_epi32( 0, 0, 0, 0, 0, 0,-1,-1);	
						break;
				}
			}else{
				// is_trans must be true
				printf("(complex * real) mtxmq_core:HOW DID WE GET HERE?\n");
			}
		}
			
		break;

	default:
		/* for (i=0; i<dimi; i++) { */
		/*     for (k=0; k<dimk; k++) { */
		/*         double aki = a[k*dimi+i]; */
		/*         for (j=0; j<numj; j++) { */
		/*             c[i*dimj+j] += aki*b[k*dimj+j]; */
		/*         } */
		/*     } */
		/* } */
		printf("HOW DID WE GET HERE?\n");
		break;
	}
}

void mTxmq_core(bool is_trans, long dimi, long dimj, long dimk,
           double_complex * __restrict__ c0, const double_complex * __restrict__ a0, const double_complex * __restrict__ b0, long numi, long numj) {
	
    int i, k;
    int dimi2 = (numi>>1)<<1;
    int dimj2 = dimj<<1;
	double tmp[4];

    __m256d ci0j0, ci0j1, ci0j2, ci0j3;
    __m256d ci1j0, ci1j1, ci1j2, ci1j3;
    __m256d aki0, aki1, bkj;
	__m256d tmp00, tmp01, tmp10, tmp11;
    __m256i mask = _mm256_set_epi32(0,0,-1,-1,-1,-1,-1,-1);
    // __m256d tmp; //temporary from aki*bkj
    
	double* a = (double*)a0;
	double* b = (double*)b0;
	double* c = (double*)c0;
	double* __restrict__ ci = (double*)c0;

	//pointer converter 
	const auto conv_addr_trans2normal = [dimi, c](long i, long j){return c + dimi * j + i;};

	switch (numj) {
	case 16:
	case 15:
	case 14:
	case 13:
		if      (numj == 16) mask = _mm256_set_epi32(-1,-1,-1,-1,-1,-1,-1,-1);
		else if (numj == 15) mask = _mm256_set_epi32( 0, 0,-1,-1,-1,-1,-1,-1);
		else if (numj == 14) mask = _mm256_set_epi32( 0, 0, 0, 0,-1,-1,-1,-1);
		else if (numj == 13) mask = _mm256_set_epi32( 0, 0, 0, 0, 0, 0,-1,-1);

		for (i=0; i<dimi2; i+=2,ci+=dimj2) {
			const double* __restrict__ pbkj = b;
			const double* __restrict__ paki = a+i*2;
			ci0j0 = _mm256_setzero_pd();
			ci0j1 = _mm256_setzero_pd();
			ci0j2 = _mm256_setzero_pd();
			ci0j3 = _mm256_setzero_pd();
			
			ci1j0 = _mm256_setzero_pd();
			ci1j1 = _mm256_setzero_pd();
			ci1j2 = _mm256_setzero_pd();
			ci1j3 = _mm256_setzero_pd();
			
			for (k=0; k<dimk; k++,pbkj+=dimj*2,paki+=dimi*2) {
				//load:[a0.re, a0.im, a0.re, a0.im]
				aki0 = _mm256_set_pd(*(paki+1), *(paki+0), *(paki+1), *(paki+0));
				aki1 = _mm256_set_pd(*(paki+3), *(paki+2), *(paki+3), *(paki+2));
				
				bkj = _mm256_loadu_pd(pbkj   );
				//multiply complex vector 
				// [a0.re, a0.im, a0.re, a0.im] * [b0.re, b0.im, b1.re, b1.im]
				//(1). make only imaginary of aki
				// [a0.im, a0.im, a0.im, a0.im]
				tmp00 = _mm256_shuffle_pd(aki0, aki0, 0xF);
				tmp10 = _mm256_shuffle_pd(aki1, aki1, 0xF);
				//(2). swap real and imaginary of bkj 
				// [b0.im, b0.re, b1.im, b1.re]
				tmp01 = _mm256_shuffle_pd(bkj, bkj, 0x5);
				tmp11 = _mm256_shuffle_pd(bkj, bkj, 0x5);
				//(3). multiply (1) and (2)
				// [a0.im*b0.im, a0.im*b0.re, a0.im*b1.im, a0.im*b1.re]
				tmp00 = _mm256_mul_pd(tmp00, tmp01);
				tmp10 = _mm256_mul_pd(tmp10, tmp11);
				//(4). make only real of aki
				// [a0.re, a0.re, a0.re, a0.re]
				tmp01 = _mm256_shuffle_pd(aki0, aki0, 0x0);
				tmp11 = _mm256_shuffle_pd(aki1, aki1, 0x0);
				//(5). FMA: (4) * b -+ (3)
				tmp00 = _mm256_fmaddsub_pd(tmp01, bkj, tmp00);
				tmp01 = _mm256_fmaddsub_pd(tmp11, bkj, tmp10);
				ci0j0 = _mm256_add_pd(ci0j0, tmp00);
				ci1j0 = _mm256_add_pd(ci1j0, tmp01);

				bkj = _mm256_loadu_pd(pbkj+ 4);
				tmp00 = _mm256_shuffle_pd(aki0, aki0, 0xF);
				tmp10 = _mm256_shuffle_pd(aki1, aki1, 0xF);
				tmp01 = _mm256_shuffle_pd(bkj, bkj, 0x5);
				tmp11 = _mm256_shuffle_pd(bkj, bkj, 0x5);
				tmp00 = _mm256_mul_pd(tmp00, tmp01);
				tmp10 = _mm256_mul_pd(tmp10, tmp11);
				tmp01 = _mm256_shuffle_pd(aki0, aki0, 0x0);
				tmp11 = _mm256_shuffle_pd(aki1, aki1, 0x0);
				tmp00 = _mm256_fmaddsub_pd(tmp01, bkj, tmp00);
				tmp01 = _mm256_fmaddsub_pd(tmp11, bkj, tmp10);
				ci0j1 = _mm256_add_pd(ci0j1, tmp00);
				ci1j1 = _mm256_add_pd(ci1j1, tmp01);

				bkj = _mm256_loadu_pd(pbkj+ 8);
				tmp00 = _mm256_shuffle_pd(aki0, aki0, 0xF);
				tmp10 = _mm256_shuffle_pd(aki1, aki1, 0xF);
				tmp01 = _mm256_shuffle_pd(bkj, bkj, 0x5);
				tmp11 = _mm256_shuffle_pd(bkj, bkj, 0x5);
				tmp00 = _mm256_mul_pd(tmp00, tmp01);
				tmp10 = _mm256_mul_pd(tmp10, tmp11);
				tmp01 = _mm256_shuffle_pd(aki0, aki0, 0x0);
				tmp11 = _mm256_shuffle_pd(aki1, aki1, 0x0);
				tmp00 = _mm256_fmaddsub_pd(tmp01, bkj, tmp00);
				tmp01 = _mm256_fmaddsub_pd(tmp11, bkj, tmp10);
				ci0j2 = _mm256_add_pd(ci0j2, tmp00);
				ci1j2 = _mm256_add_pd(ci1j2, tmp01);
				
				bkj = _mm256_maskload_pd(pbkj+12,mask);
				tmp00 = _mm256_shuffle_pd(aki0, aki0, 0xF);
				tmp10 = _mm256_shuffle_pd(aki1, aki1, 0xF);
				tmp01 = _mm256_shuffle_pd(bkj, bkj, 0x5);
				tmp11 = _mm256_shuffle_pd(bkj, bkj, 0x5);
				tmp00 = _mm256_mul_pd(tmp00, tmp01);
				tmp10 = _mm256_mul_pd(tmp10, tmp11);
				tmp01 = _mm256_shuffle_pd(aki0, aki0, 0x0);
				tmp11 = _mm256_shuffle_pd(aki1, aki1, 0x0);
				tmp00 = _mm256_fmaddsub_pd(tmp01, bkj, tmp00);
				tmp01 = _mm256_fmaddsub_pd(tmp11, bkj, tmp10);
				ci0j3 = _mm256_add_pd(ci0j3, tmp00);
				ci1j3 = _mm256_add_pd(ci1j3, tmp01);
			}
			if(!is_trans){
				_mm256_storeu_pd(c+i*dimj*2   , ci0j0);
				_mm256_storeu_pd(c+i*dimj*2+ 4, ci0j1);
				_mm256_storeu_pd(c+i*dimj*2+ 8, ci0j2);
				_mm256_maskstore_pd(c+i*dimj*2+12, mask, ci0j3); 
			   
				_mm256_storeu_pd(c+(i+1)*dimj*2   , ci1j0);
				_mm256_storeu_pd(c+(i+1)*dimj*2+ 4, ci1j1);
				_mm256_storeu_pd(c+(i+1)*dimj*2+ 8, ci1j2);
				_mm256_maskstore_pd(c+(i+1)*dimj*2+12, mask, ci1j3);
			}else{
				//temporary use aki0, aki1 and mask
				aki0 = _mm256_shuffle_pd(ci0j0, ci1j0, 0);
				aki1 = _mm256_shuffle_pd(ci0j0, ci1j0, -1);
				_mm256_storeu2_m128d(conv_addr_trans2normal(i,2), conv_addr_trans2normal(i,0), aki0);
				_mm256_storeu2_m128d(conv_addr_trans2normal(i,3), conv_addr_trans2normal(i,1), aki1);

				aki0 = _mm256_shuffle_pd(ci0j1, ci1j1, 0);
				aki1 = _mm256_shuffle_pd(ci0j1, ci1j1, -1);
				_mm256_storeu2_m128d(conv_addr_trans2normal(i,6), conv_addr_trans2normal(i,4), aki0);
				_mm256_storeu2_m128d(conv_addr_trans2normal(i,7), conv_addr_trans2normal(i,5), aki1);

				aki0 = _mm256_shuffle_pd(ci0j2, ci1j2, 0);
				aki1 = _mm256_shuffle_pd(ci0j2, ci1j2, -1);
				_mm256_storeu2_m128d(conv_addr_trans2normal(i,10), conv_addr_trans2normal(i,8), aki0);
				_mm256_storeu2_m128d(conv_addr_trans2normal(i,11), conv_addr_trans2normal(i,9), aki1);

				aki0 = _mm256_shuffle_pd(ci0j3, ci1j3, 0);
				aki1 = _mm256_shuffle_pd(ci0j3, ci1j3, -1);
				switch(numj){
					case 16:
						_mm256_storeu2_m128d(conv_addr_trans2normal(i,14), conv_addr_trans2normal(i,12), aki0);
						_mm256_storeu2_m128d(conv_addr_trans2normal(i,15), conv_addr_trans2normal(i,13), aki1);
						break;
					case 15:
						mask = _mm256_set_epi64x(0, 0, -1, -1);
						_mm256_storeu2_m128d(conv_addr_trans2normal(i,14), conv_addr_trans2normal(i,12), aki0);
						_mm256_maskstore_pd(conv_addr_trans2normal(i,13), mask, aki1);
						mask = _mm256_set_epi32( 0, 0,-1,-1,-1,-1,-1,-1);	//reset mask
						break;
					case 14:
						mask = _mm256_set_epi64x(0, 0, -1, -1);
						_mm256_maskstore_pd(conv_addr_trans2normal(i,12), mask, aki0);
						_mm256_maskstore_pd(conv_addr_trans2normal(i,13), mask, aki1);
						mask = _mm256_set_epi32( 0, 0, 0, 0,-1,-1,-1,-1);	//reset mask
						break;
					case 13:
						mask = _mm256_set_epi64x(0, 0, -1, -1);
						_mm256_maskstore_pd(conv_addr_trans2normal(i,12), mask, aki0);
						mask = _mm256_set_epi32( 0, 0, 0, 0, 0, 0,-1,-1);	//reset mask
						break;
				}
			}
		}
			
		if (numi&0x1) {
			const double* __restrict__ pbkj = b;
			const double* __restrict__ paki = a+i*2;
			ci0j0 = _mm256_setzero_pd();
			ci0j1 = _mm256_setzero_pd();
			ci0j2 = _mm256_setzero_pd();
			ci0j3 = _mm256_setzero_pd();
			
			for (k=0; k<dimk; k++,pbkj+=dimj*2,paki+=dimi*2) {
				aki0 = _mm256_set_pd(*(paki+1), *(paki+0), *(paki+1), *(paki+0));
				
				bkj = _mm256_loadu_pd(pbkj   );
				tmp00 = _mm256_shuffle_pd(aki0, aki0, 0xF);
				tmp01 = _mm256_shuffle_pd(bkj, bkj, 0x5);
				tmp00 = _mm256_mul_pd(tmp00, tmp01);
				tmp01 = _mm256_shuffle_pd(aki0, aki0, 0x0);
				tmp00 = _mm256_fmaddsub_pd(tmp01, bkj, tmp00);
				ci0j0 = _mm256_add_pd(ci0j0, tmp00);
				
				bkj = _mm256_loadu_pd(pbkj+ 4);
				tmp00 = _mm256_shuffle_pd(aki0, aki0, 0xF);
				tmp01 = _mm256_shuffle_pd(bkj, bkj, 0x5);
				tmp00 = _mm256_mul_pd(tmp00, tmp01);
				tmp01 = _mm256_shuffle_pd(aki0, aki0, 0x0);
				tmp00 = _mm256_fmaddsub_pd(tmp01, bkj, tmp00);
				ci0j1 = _mm256_add_pd(ci0j1, tmp00);
				
				bkj = _mm256_loadu_pd(pbkj+ 8);
				tmp00 = _mm256_shuffle_pd(aki0, aki0, 0xF);
				tmp01 = _mm256_shuffle_pd(bkj, bkj, 0x5);
				tmp00 = _mm256_mul_pd(tmp00, tmp01);
				tmp01 = _mm256_shuffle_pd(aki0, aki0, 0x0);
				tmp00 = _mm256_fmaddsub_pd(tmp01, bkj, tmp00);
				ci0j2 = _mm256_add_pd(ci0j2, tmp00);
				
				bkj = _mm256_maskload_pd(pbkj+12,mask);
				tmp00 = _mm256_shuffle_pd(aki0, aki0, 0xF);
				tmp01 = _mm256_shuffle_pd(bkj, bkj, 0x5);
				tmp00 = _mm256_mul_pd(tmp00, tmp01);
				tmp01 = _mm256_shuffle_pd(aki0, aki0, 0x0);
				tmp00 = _mm256_fmaddsub_pd(tmp01, bkj, tmp00);
				ci0j3 = _mm256_add_pd(ci0j3, tmp00);
			}
			if(!is_trans){
				_mm256_storeu_pd(c+i*dimj*2   , ci0j0);
				_mm256_storeu_pd(c+i*dimj*2+ 4, ci0j1);
				_mm256_storeu_pd(c+i*dimj*2+ 8, ci0j2);
				_mm256_maskstore_pd(c+i*dimj*2+12, mask, ci0j3); 
			}else{
				_mm256_storeu_pd(tmp, ci0j0);
				c[0*dimi+i] = tmp[0];
				c[1*dimi+i] = tmp[1];
				c[2*dimi+i] = tmp[2];
				c[3*dimi+i] = tmp[3];

				_mm256_storeu_pd(tmp, ci0j1);
				c[ 4*dimi+i] = tmp[0];
				c[ 5*dimi+i] = tmp[1];
				c[ 6*dimi+i] = tmp[2];
				c[ 7*dimi+i] = tmp[3];

				_mm256_storeu_pd(tmp, ci0j2);
				c[ 8*dimi+i] = tmp[0];
				c[ 9*dimi+i] = tmp[1];
				c[10*dimi+i] = tmp[2];
				c[11*dimi+i] = tmp[3];

				_mm256_storeu_pd(tmp, ci0j3);
				switch(numj){
				case 16:
					c[15*dimi+i] = tmp[3];
				case 15:
					c[14*dimi+i] = tmp[2];
				case 14:
					c[13*dimi+i] = tmp[1];
				case 13:
					c[12*dimi+i] = tmp[0];
				}
			}
		}

		break;
		
	case 12:
	case 11:
	case 10:
	case  9:
		if      (numj == 12) mask = _mm256_set_epi32(-1,-1,-1,-1,-1,-1,-1,-1);
		else if (numj == 11) mask = _mm256_set_epi32( 0, 0,-1,-1,-1,-1,-1,-1);
		else if (numj == 10) mask = _mm256_set_epi32( 0, 0, 0, 0,-1,-1,-1,-1);
		else if (numj ==  9) mask = _mm256_set_epi32( 0, 0, 0, 0, 0, 0,-1,-1);

		for (i=0; i<dimi2; i+=2,ci+=dimj2) {
			const double* __restrict__ pbkj = b;
			const double* __restrict__ paki = a+i*2;
			ci0j0 = _mm256_setzero_pd();
			ci0j1 = _mm256_setzero_pd();
			ci0j2 = _mm256_setzero_pd();
			
			ci1j0 = _mm256_setzero_pd();
			ci1j1 = _mm256_setzero_pd();
			ci1j2 = _mm256_setzero_pd();
			
			for (k=0; k<dimk; k++,pbkj+=dimj*2,paki+=dimi*2) {
				aki0 = _mm256_set_pd(*(paki+1), *(paki+0), *(paki+1), *(paki+0));
				aki1 = _mm256_set_pd(*(paki+3), *(paki+2), *(paki+3), *(paki+2));
				
				bkj = _mm256_loadu_pd(pbkj);
				tmp00 = _mm256_shuffle_pd(aki0, aki0, 0xF);
				tmp10 = _mm256_shuffle_pd(aki1, aki1, 0xF);
				tmp01 = _mm256_shuffle_pd(bkj, bkj, 0x5);
				tmp11 = _mm256_shuffle_pd(bkj, bkj, 0x5);
				tmp00 = _mm256_mul_pd(tmp00, tmp01);
				tmp10 = _mm256_mul_pd(tmp10, tmp11);
				tmp01 = _mm256_shuffle_pd(aki0, aki0, 0x0);
				tmp11 = _mm256_shuffle_pd(aki1, aki1, 0x0);
				tmp00 = _mm256_fmaddsub_pd(tmp01, bkj, tmp00);
				tmp01 = _mm256_fmaddsub_pd(tmp11, bkj, tmp10);
				ci0j0 = _mm256_add_pd(ci0j0, tmp00);
				ci1j0 = _mm256_add_pd(ci1j0, tmp01);

				
				bkj = _mm256_loadu_pd(pbkj+ 4);
				tmp00 = _mm256_shuffle_pd(aki0, aki0, 0xF);
				tmp10 = _mm256_shuffle_pd(aki1, aki1, 0xF);
				tmp01 = _mm256_shuffle_pd(bkj, bkj, 0x5);
				tmp11 = _mm256_shuffle_pd(bkj, bkj, 0x5);
				tmp00 = _mm256_mul_pd(tmp00, tmp01);
				tmp10 = _mm256_mul_pd(tmp10, tmp11);
				tmp01 = _mm256_shuffle_pd(aki0, aki0, 0x0);
				tmp11 = _mm256_shuffle_pd(aki1, aki1, 0x0);
				tmp00 = _mm256_fmaddsub_pd(tmp01, bkj, tmp00);
				tmp01 = _mm256_fmaddsub_pd(tmp11, bkj, tmp10);
				ci0j1 = _mm256_add_pd(ci0j1, tmp00);
				ci1j1 = _mm256_add_pd(ci1j1, tmp01);
				
				bkj = _mm256_maskload_pd(pbkj+8,mask);
				tmp00 = _mm256_shuffle_pd(aki0, aki0, 0xF);
				tmp10 = _mm256_shuffle_pd(aki1, aki1, 0xF);
				tmp01 = _mm256_shuffle_pd(bkj, bkj, 0x5);
				tmp11 = _mm256_shuffle_pd(bkj, bkj, 0x5);
				tmp00 = _mm256_mul_pd(tmp00, tmp01);
				tmp10 = _mm256_mul_pd(tmp10, tmp11);
				tmp01 = _mm256_shuffle_pd(aki0, aki0, 0x0);
				tmp11 = _mm256_shuffle_pd(aki1, aki1, 0x0);
				tmp00 = _mm256_fmaddsub_pd(tmp01, bkj, tmp00);
				tmp01 = _mm256_fmaddsub_pd(tmp11, bkj, tmp10);
				ci0j2 = _mm256_add_pd(ci0j2, tmp00);
				ci1j2 = _mm256_add_pd(ci1j2, tmp01);
			}
			if(!is_trans){
				_mm256_storeu_pd(c+i*dimj*2   , ci0j0);
				_mm256_storeu_pd(c+i*dimj*2+ 4, ci0j1);
				_mm256_maskstore_pd(c+i*dimj*2+8, mask, ci0j2); 
				
				_mm256_storeu_pd(c+(i+1)*dimj*2   , ci1j0);
				_mm256_storeu_pd(c+(i+1)*dimj*2+ 4, ci1j1);
				_mm256_maskstore_pd(c+(i+1)*dimj*2+8, mask, ci1j2); 
			}else{
				//temporary use aki0, aki1 and mask
				aki0 = _mm256_shuffle_pd(ci0j0, ci1j0, 0);
				aki1 = _mm256_shuffle_pd(ci0j0, ci1j0, -1);
				_mm256_storeu2_m128d(conv_addr_trans2normal(i,2), conv_addr_trans2normal(i,0), aki0);
				_mm256_storeu2_m128d(conv_addr_trans2normal(i,3), conv_addr_trans2normal(i,1), aki1);

				aki0 = _mm256_shuffle_pd(ci0j1, ci1j1, 0);
				aki1 = _mm256_shuffle_pd(ci0j1, ci1j1, -1);
				_mm256_storeu2_m128d(conv_addr_trans2normal(i,6), conv_addr_trans2normal(i,4), aki0);
				_mm256_storeu2_m128d(conv_addr_trans2normal(i,7), conv_addr_trans2normal(i,5), aki1);

				aki0 = _mm256_shuffle_pd(ci0j2, ci1j2, 0);
				aki1 = _mm256_shuffle_pd(ci0j2, ci1j2, -1);
				switch(numj){
					case 12:
						_mm256_storeu2_m128d(conv_addr_trans2normal(i,10), conv_addr_trans2normal(i,8), aki0);
						_mm256_storeu2_m128d(conv_addr_trans2normal(i,11), conv_addr_trans2normal(i,9), aki1);
						break;
					case 11:
						mask = _mm256_set_epi64x(0, 0, -1, -1);
						_mm256_storeu2_m128d(conv_addr_trans2normal(i,10), conv_addr_trans2normal(i,8), aki0);
						_mm256_maskstore_pd(conv_addr_trans2normal(i,9), mask, aki1);
						mask = _mm256_set_epi32( 0, 0,-1,-1,-1,-1,-1,-1);	//reset mask
						break;
					case 10:
						mask = _mm256_set_epi64x(0, 0, -1, -1);
						_mm256_maskstore_pd(conv_addr_trans2normal(i,8), mask, aki0);
						_mm256_maskstore_pd(conv_addr_trans2normal(i,9), mask, aki1);
						mask = _mm256_set_epi32( 0, 0, 0, 0,-1,-1,-1,-1);	//reset mask
						break;
					case  9:
						mask = _mm256_set_epi64x(0, 0, -1, -1);
						_mm256_maskstore_pd(conv_addr_trans2normal(i,8), mask, aki0);
						mask = _mm256_set_epi32( 0, 0, 0, 0, 0, 0,-1,-1);	//reset mask
						break;
				}
			}
		}
			
		if (numi&0x1) {
			const double* __restrict__ pbkj = b;
			const double* __restrict__ paki = a+i*2;
			ci0j0 = _mm256_setzero_pd();
			ci0j1 = _mm256_setzero_pd();
			ci0j2 = _mm256_setzero_pd();
			
			for (k=0; k<dimk; k++,pbkj+=dimj*2,paki+=dimi*2) {
				aki0 = _mm256_set_pd(*(paki+1), *(paki+0), *(paki+1), *(paki+0));
				
				bkj = _mm256_loadu_pd(pbkj   );
				tmp00 = _mm256_shuffle_pd(aki0, aki0, 0xF);
				tmp01 = _mm256_shuffle_pd(bkj, bkj, 0x5);
				tmp00 = _mm256_mul_pd(tmp00, tmp01);
				tmp01 = _mm256_shuffle_pd(aki0, aki0, 0x0);
				tmp00 = _mm256_fmaddsub_pd(tmp01, bkj, tmp00);
				ci0j0 = _mm256_add_pd(ci0j0, tmp00);
				
				bkj = _mm256_loadu_pd(pbkj+ 4);
				tmp00 = _mm256_shuffle_pd(aki0, aki0, 0xF);
				tmp01 = _mm256_shuffle_pd(bkj, bkj, 0x5);
				tmp00 = _mm256_mul_pd(tmp00, tmp01);
				tmp01 = _mm256_shuffle_pd(aki0, aki0, 0x0);
				tmp00 = _mm256_fmaddsub_pd(tmp01, bkj, tmp00);
				ci0j1 = _mm256_add_pd(ci0j1, tmp00);
				
				bkj = _mm256_maskload_pd(pbkj+8,mask);
				tmp00 = _mm256_shuffle_pd(aki0, aki0, 0xF);
				tmp01 = _mm256_shuffle_pd(bkj, bkj, 0x5);
				tmp00 = _mm256_mul_pd(tmp00, tmp01);
				tmp01 = _mm256_shuffle_pd(aki0, aki0, 0x0);
				tmp00 = _mm256_fmaddsub_pd(tmp01, bkj, tmp00);
				ci0j2 = _mm256_add_pd(ci0j2, tmp00);
			}
			if(!is_trans){
				_mm256_storeu_pd(c+i*dimj*2   , ci0j0);
				_mm256_storeu_pd(c+i*dimj*2+ 4, ci0j1);
				_mm256_maskstore_pd(c+i*dimj*2+8, mask, ci0j2); 
			}else{
				_mm256_storeu_pd(tmp, ci0j0);
				c[0*dimi+i] = tmp[0];
				c[1*dimi+i] = tmp[1];
				c[2*dimi+i] = tmp[2];
				c[3*dimi+i] = tmp[3];

				_mm256_storeu_pd(tmp, ci0j1);
				c[4*dimi+i] = tmp[0];
				c[5*dimi+i] = tmp[1];
				c[6*dimi+i] = tmp[2];
				c[7*dimi+i] = tmp[3];

				_mm256_storeu_pd(tmp, ci0j2);
				switch(numj){
				case 12:
					c[11*dimi+i] = tmp[3];
				case 11:
					c[10*dimi+i] = tmp[2];
				case 10:
					c[9*dimi+i] = tmp[1];
				case  9:
					c[8*dimi+i] = tmp[0];
				}
			}
		}

		break;
		
	case 8:
	case 7:
	case 6:
	case 5:
		if      (numj == 8) mask = _mm256_set_epi32(-1,-1,-1,-1,-1,-1,-1,-1);
		else if (numj == 7) mask = _mm256_set_epi32( 0, 0,-1,-1,-1,-1,-1,-1);
		else if (numj == 6) mask = _mm256_set_epi32( 0, 0, 0, 0,-1,-1,-1,-1);
		else if (numj == 5) mask = _mm256_set_epi32( 0, 0, 0, 0, 0, 0,-1,-1);
		for (i=0; i<dimi2; i+=2,ci+=dimj2) {
			const double* __restrict__ pbkj = b;
			const double* __restrict__ paki = a+i*2;
			ci0j0 = _mm256_setzero_pd();
			ci0j1 = _mm256_setzero_pd();
			
			ci1j0 = _mm256_setzero_pd();
			ci1j1 = _mm256_setzero_pd();
			
			for (k=0; k<dimk; k++,pbkj+=dimj*2,paki+=dimi*2) {
				aki0 = _mm256_set_pd(*(paki+1), *(paki+0), *(paki+1), *(paki+0));
				aki1 = _mm256_set_pd(*(paki+3), *(paki+2), *(paki+3), *(paki+2));
				
				bkj = _mm256_loadu_pd(pbkj   );
				tmp00 = _mm256_shuffle_pd(aki0, aki0, 0xF);
				tmp10 = _mm256_shuffle_pd(aki1, aki1, 0xF);
				tmp01 = _mm256_shuffle_pd(bkj, bkj, 0x5);
				tmp11 = _mm256_shuffle_pd(bkj, bkj, 0x5);
				tmp00 = _mm256_mul_pd(tmp00, tmp01);
				tmp10 = _mm256_mul_pd(tmp10, tmp11);
				tmp01 = _mm256_shuffle_pd(aki0, aki0, 0x0);
				tmp11 = _mm256_shuffle_pd(aki1, aki1, 0x0);
				tmp00 = _mm256_fmaddsub_pd(tmp01, bkj, tmp00);
				tmp01 = _mm256_fmaddsub_pd(tmp11, bkj, tmp10);
				ci0j0 = _mm256_add_pd(ci0j0, tmp00);
				ci1j0 = _mm256_add_pd(ci1j0, tmp01);
				
				bkj = _mm256_maskload_pd(pbkj+4,mask);
				tmp00 = _mm256_shuffle_pd(aki0, aki0, 0xF);
				tmp10 = _mm256_shuffle_pd(aki1, aki1, 0xF);
				tmp01 = _mm256_shuffle_pd(bkj, bkj, 0x5);
				tmp11 = _mm256_shuffle_pd(bkj, bkj, 0x5);
				tmp00 = _mm256_mul_pd(tmp00, tmp01);
				tmp10 = _mm256_mul_pd(tmp10, tmp11);
				tmp01 = _mm256_shuffle_pd(aki0, aki0, 0x0);
				tmp11 = _mm256_shuffle_pd(aki1, aki1, 0x0);
				tmp00 = _mm256_fmaddsub_pd(tmp01, bkj, tmp00);
				tmp01 = _mm256_fmaddsub_pd(tmp11, bkj, tmp10);
				ci0j1 = _mm256_add_pd(ci0j1, tmp00);
				ci1j1 = _mm256_add_pd(ci1j1, tmp01);
			}
			if(!is_trans){
				_mm256_storeu_pd(c+i*dimj*2   , ci0j0);
				_mm256_maskstore_pd(c+i*dimj*2+4, mask, ci0j1); 
				
				_mm256_storeu_pd(c+(i+1)*dimj*2   , ci1j0);
				_mm256_maskstore_pd(c+(i+1)*dimj*2+4, mask, ci1j1); 
			}else{
				//temporary use aki0, aki1 and mask
				aki0 = _mm256_shuffle_pd(ci0j0, ci1j0, 0);
				aki1 = _mm256_shuffle_pd(ci0j0, ci1j0, -1);
				_mm256_storeu2_m128d(conv_addr_trans2normal(i,2), conv_addr_trans2normal(i,0), aki0);
				_mm256_storeu2_m128d(conv_addr_trans2normal(i,3), conv_addr_trans2normal(i,1), aki1);

				aki0 = _mm256_shuffle_pd(ci0j1, ci1j1, 0);
				aki1 = _mm256_shuffle_pd(ci0j1, ci1j1, -1);
				switch(numj){
					case  8:
						_mm256_storeu2_m128d(conv_addr_trans2normal(i,6), conv_addr_trans2normal(i,4), aki0);
						_mm256_storeu2_m128d(conv_addr_trans2normal(i,7), conv_addr_trans2normal(i,5), aki1);
						mask = _mm256_set_epi32(-1,-1,-1,-1,-1,-1,-1,-1);	//reset mask
						break;
					case  7:
						mask = _mm256_set_epi64x(0, 0, -1, -1);
						_mm256_storeu2_m128d(conv_addr_trans2normal(i,6), conv_addr_trans2normal(i,4), aki0);
						_mm256_maskstore_pd(conv_addr_trans2normal(i,5), mask, aki1);
						mask = _mm256_set_epi32( 0, 0,-1,-1,-1,-1,-1,-1);	//reset mask
						break;
					case  6:
						mask = _mm256_set_epi64x(0, 0, -1, -1);
						_mm256_maskstore_pd(conv_addr_trans2normal(i,4), mask, aki0);
						_mm256_maskstore_pd(conv_addr_trans2normal(i,5), mask, aki1);
						mask = _mm256_set_epi32( 0, 0, 0, 0,-1,-1,-1,-1);	//reset mask
						break;
					case  5:
						mask = _mm256_set_epi64x(0, 0, -1, -1);
						_mm256_maskstore_pd(conv_addr_trans2normal(i,4), mask, aki0);
						mask = _mm256_set_epi32( 0, 0, 0, 0, 0, 0,-1,-1);	//reset mask
						break;
				}
			}

		}
			
		if (numi&0x1) {
			const double* __restrict__ pbkj = b;
			const double* __restrict__ paki = a+i*2;
			ci0j0 = _mm256_setzero_pd();
			ci0j1 = _mm256_setzero_pd();
			
			for (k=0; k<dimk; k++,pbkj+=dimj*2,paki+=dimi*2) {
				aki0 = _mm256_set_pd(*(paki+1), *(paki+0), *(paki+1), *(paki+0));
				
				bkj = _mm256_loadu_pd(pbkj   );
				tmp00 = _mm256_shuffle_pd(aki0, aki0, 0xF);
				tmp01 = _mm256_shuffle_pd(bkj, bkj, 0x5);
				tmp00 = _mm256_mul_pd(tmp00, tmp01);
				tmp01 = _mm256_shuffle_pd(aki0, aki0, 0x0);
				tmp00 = _mm256_fmaddsub_pd(tmp01, bkj, tmp00);
				ci0j0 = _mm256_add_pd(ci0j0, tmp00);

				bkj = _mm256_maskload_pd(pbkj+4,mask);
				tmp00 = _mm256_shuffle_pd(aki0, aki0, 0xF);
				tmp01 = _mm256_shuffle_pd(bkj, bkj, 0x5);
				tmp00 = _mm256_mul_pd(tmp00, tmp01);
				tmp01 = _mm256_shuffle_pd(aki0, aki0, 0x0);
				tmp00 = _mm256_fmaddsub_pd(tmp01, bkj, tmp00);
				ci0j1 = _mm256_add_pd(ci0j1, tmp00);
			}

			if(!is_trans){
				_mm256_storeu_pd(c+i*dimj*2   , ci0j0);
				_mm256_maskstore_pd(c+i*dimj*2+4, mask, ci0j1); 
			}else{
				_mm256_storeu_pd(tmp, ci0j0);
				c[0*dimi+i] = tmp[0];
				c[1*dimi+i] = tmp[1];
				c[2*dimi+i] = tmp[2];
				c[3*dimi+i] = tmp[3];

				_mm256_storeu_pd(tmp, ci0j1);
				switch(numj){
				case 8:
					c[7*dimi+i] = tmp[3];
				case 7:
					c[6*dimi+i] = tmp[2];
				case 6:
					c[5*dimi+i] = tmp[1];
				case 5:
					c[4*dimi+i] = tmp[0];
				}
			}
		}

		break;
		
	case 4:
	case 3:
	case 2:
	case 1:
		if      (numj == 4) mask = _mm256_set_epi32(-1,-1,-1,-1,-1,-1,-1,-1);
		else if (numj == 3) mask = _mm256_set_epi32( 0, 0,-1,-1,-1,-1,-1,-1);
		else if (numj == 2) mask = _mm256_set_epi32( 0, 0, 0, 0,-1,-1,-1,-1);
		else if (numj == 1) mask = _mm256_set_epi32( 0, 0, 0, 0, 0, 0,-1,-1);

		for (i=0; i<dimi2; i+=2,ci+=dimj2) {
			const double* __restrict__ pbkj = b;
			const double* __restrict__ paki = a+i*2;
			ci0j0 = _mm256_setzero_pd();
			ci1j0 = _mm256_setzero_pd();
			
			for (k=0; k<dimk; k++,pbkj+=dimj*2,paki+=dimi*2) {
				aki0 = _mm256_set_pd(*(paki+1), *(paki+0), *(paki+1), *(paki+0));
				aki1 = _mm256_set_pd(*(paki+3), *(paki+2), *(paki+3), *(paki+2));
				
				bkj = _mm256_maskload_pd(pbkj, mask);
				tmp00 = _mm256_shuffle_pd(aki0, aki0, 0xF);
				tmp10 = _mm256_shuffle_pd(aki1, aki1, 0xF);
				tmp01 = _mm256_shuffle_pd(bkj, bkj, 0x5);
				tmp11 = _mm256_shuffle_pd(bkj, bkj, 0x5);
				tmp00 = _mm256_mul_pd(tmp00, tmp01);
				tmp10 = _mm256_mul_pd(tmp10, tmp11);
				tmp01 = _mm256_shuffle_pd(aki0, aki0, 0x0);
				tmp11 = _mm256_shuffle_pd(aki1, aki1, 0x0);
				tmp00 = _mm256_fmaddsub_pd(tmp01, bkj, tmp00);
				tmp01 = _mm256_fmaddsub_pd(tmp11, bkj, tmp10);
				ci0j0 = _mm256_add_pd(ci0j0, tmp00);
				ci1j0 = _mm256_add_pd(ci1j0, tmp01);
			}
			if(!is_trans){
				_mm256_maskstore_pd(c+i*dimj*2    , mask, ci0j0);
				_mm256_maskstore_pd(c+(i+1)*dimj*2, mask, ci1j0);
			}else{
				//temporary use aki0, aki1 and mask
				aki0 = _mm256_shuffle_pd(ci0j0, ci1j0, 0);
				aki1 = _mm256_shuffle_pd(ci0j0, ci1j0, -1);
				switch(numj){
					case  4:
						_mm256_storeu2_m128d(conv_addr_trans2normal(i,2), conv_addr_trans2normal(i,0), aki0);
						_mm256_storeu2_m128d(conv_addr_trans2normal(i,3), conv_addr_trans2normal(i,1), aki1);
						break;
					case  3:
						mask = _mm256_set_epi64x(0, 0, -1, -1);
						_mm256_storeu2_m128d(conv_addr_trans2normal(i,2), conv_addr_trans2normal(i,0), aki0);
						_mm256_maskstore_pd(conv_addr_trans2normal(i,1), mask, aki1);
						mask = _mm256_set_epi32( 0, 0,-1,-1,-1,-1,-1,-1);	//reset mask
						break;
					case  2:

						mask = _mm256_set_epi64x(0, 0, -1, -1);
						_mm256_maskstore_pd(conv_addr_trans2normal(i,0), mask, aki0);
						_mm256_maskstore_pd(conv_addr_trans2normal(i,1), mask, aki1);
						mask = _mm256_set_epi32( 0, 0, 0, 0,-1,-1,-1,-1);	//reset mask
						break;
					case  1:
						mask = _mm256_set_epi64x(0, 0, -1, -1);
						_mm256_maskstore_pd(conv_addr_trans2normal(i,0), mask, aki0);
						mask = _mm256_set_epi32( 0, 0, 0, 0, 0, 0,-1,-1);	//reset mask
						break;
				}
			}
		}
			
		if (numi&0x1) {
			const double* __restrict__ pbkj = b;
			const double* __restrict__ paki = a+i*2;
			ci0j0 = _mm256_setzero_pd();
			for (k=0; k<dimk; k++,pbkj+=dimj*2,paki+=dimi*2) {
				aki0 = _mm256_set_pd(*(paki+1), *(paki), *(paki+1), *(paki));

				bkj = _mm256_maskload_pd(pbkj, mask);
				tmp00 = _mm256_shuffle_pd(aki0, aki0, 0xF);
				tmp01 = _mm256_shuffle_pd(bkj, bkj, 0x5);
				tmp00 = _mm256_mul_pd(tmp00, tmp01);
				tmp01 = _mm256_shuffle_pd(aki0, aki0, 0x0);
				tmp00 = _mm256_fmaddsub_pd(tmp01, bkj, tmp00);
				ci0j0 = _mm256_add_pd(ci0j0, tmp00);
			}
			if(!is_trans){
				_mm256_maskstore_pd(c+i*dimj*2    , mask, ci0j0);
			}else{
				_mm256_storeu_pd(tmp, ci0j0);
				switch(numj){
				case 4:
					c[3*dimi+i] = tmp[3];
				case 3:
					c[2*dimi+i] = tmp[2];
				case 2:
					c[1*dimi+i] = tmp[1];
				case 1:
					c[0*dimi+i] = tmp[0];
				}
			}
		}
		break;

	default:
		/* for (i=0; i<dimi; i++) { */
		/*     for (k=0; k<dimk; k++) { */
		/*         double aki = a[k*dimi+i]; */
		/*         for (j=0; j<numj; j++) { */
		/*             c[i*dimj+j] += aki*b[k*dimj+j]; */
		/*         } */
		/*     } */
		/* } */
		printf("HOW DID WE GET HERE?\n");
		break;
	}
}

//Real x Real(AVX)
    template<>
void mTxmq(long dimi, long dimj, long dimk,
           double * __restrict__ c, const double * __restrict__ a, const double * __restrict__ b) {

	int nj = dimj;
	int ni = dimi;
	do{
        int numj = (nj>24) ? 24 : nj;
        int numi = (ni>24) ? 24 : ni;
		double* __restrict__ ci = c;

		if(dimj % 24 >= 12 || dimi % 24 >= 12){
			if((dimj-1) % 24 >= (dimi-1) % 24 && ((dimj % 4 == 0) || (dimi % 4 != 0))){
				mTxmq_core(false, dimi, dimj, dimk, ci, a, b, ni, numj);
				c += numj;
				b += numj;
				nj -= numj;
			}else{
				mTxmq_core(true, dimj, dimi, dimk, ci, b, a, nj, numi);
				c += dimj*numi;
				a += numi;
				ni -= numi;
			}
		}else{
			if((numj-1) % 24 >= (numi-1) % 24 && ((numj % 4 == 0) || (numi % 4 != 0))){
				mTxmq_core(false, dimi, dimj, dimk, ci, a, b, ni, numj);
				c += numj;
				b += numj;
				nj -= numj;
			}else{
				mTxmq_core(true, dimj, dimi, dimk, ci, b, a, nj, numi);
				c += dimj*numi;
				a += numi;
				ni -= numi;
			}
		}
	}while(nj && ni);
}

//Real x Complex(AVX)
    template<>
void mTxmq(long dimi, long dimj, long dimk,
           double_complex* __restrict__ c, const double* __restrict__ a, const double_complex* __restrict__ b) {

	mTxmq(dimi, dimj*2, dimk, (double*)c, a, (double*)b);
}

//Complex x Real(AVX)
    template<>
void mTxmq(long dimi, long dimj, long dimk,
           double_complex* __restrict__ c, const double_complex* __restrict__ a, const double* __restrict__ b) {

	dimi *= 2;	//a:complex matrix
	int nj = dimj;
	int ni = dimi;
	do{
        int numj = (nj>24) ? 24 : nj;
        int numi = (ni>24) ? 24 : ni;
		double_complex* __restrict__ ci = c;

		//a,c are double_complex*(not double*)
		if(dimj % 24 >= 12 || dimi % 24 >= 12){
			if((dimj-1) % 24 >= (dimi-1) % 24 && ((dimj % 4 == 0) || (dimi % 4 != 0))){
				mTxmq_core(false, dimi, dimj, dimk, ci, a, b, ni, numj);
				c += numj;
				b += numj;
				nj -= numj;
			}else{
				mTxmq_core(true, dimj, dimi, dimk, ci, b, a, nj, numi);
				c += dimj*(numi/2);
				a += numi/2;
				ni -= numi;
			}
		}else{
			if((numj-1) % 24 >= (numi-1) % 24 && ((numj % 4 == 0) || (numi % 4 != 0))){
				mTxmq_core(false, dimi, dimj, dimk, ci, a, b, ni, numj);
				c += numj;
				b += numj;
				nj -= numj;
			}else{
				mTxmq_core(true, dimj, dimi, dimk, ci, b, a, nj, numi);
				c += dimj*(numi/2);
				a += numi/2;
				ni -= numi;
			}

		}

	}while(nj && ni);
}

//Complex x Complex(AVX)
    template<>
void mTxmq(long dimi, long dimj, long dimk,
           double_complex * __restrict__ c, const double_complex * __restrict__ a, const double_complex * __restrict__ b) {

	int nj = dimj*2;
	int ni = dimi; //dimi*2?
	do{
        int numj = (nj>16) ? 16 : nj;
        int numi = (ni>16) ? 16 : ni;
		double_complex* __restrict__ ci = c;

		//if(dimj % 16 >= 12 || dimi % 16 >= 12){
		if(1){
			//if((dimj-1) % 16 >= (dimi-1) % 16 && ((dimj % 4 == 0) || (dimi % 4 != 0))){
			if(1){
				mTxmq_core(false, dimi, dimj, dimk, ci, a, b, ni, numj);
				c += numj/2;
				b += numj/2;
				nj -= numj;
			}else{
				mTxmq_core(true, dimj, dimi, dimk, ci, b, a, nj, numi);
				c += dimj*numi;
				a += numi;
				ni -= numi;
			}
		}else{
			if((numj-1) % 24 >= (numi-1) % 24 && ((numj % 4 == 0) || (numi % 4 != 0))){
				mTxmq_core(false, dimi, dimj, dimk, ci, a, b, ni, numj);
				c += numj;
				b += numj;
				nj -= numj;
			}else{
				mTxmq_core(true, dimj, dimi, dimk, ci, b, a, nj, numi);
				c += dimj*numi;
				a += numi;
				ni -= numi;
			}
		}
	}while(nj && ni);
}

    //template<>
    void mTxmqdjflkjsalkf(const long dimi, const long dimj, const long dimk,
               double* restrict c, const double* a, const double* b) {
        //PROFILE_BLOCK(mTxmq_double_asm);
        //std::cout << "IN DOUBLE ASM VERSION " << dimi << " " << dimj << " " << dimk << "\n";


        if (IS_ODD(dimi) || IS_ODD(dimj) || IS_ODD(dimk) ||
            IS_UNALIGNED(a) || IS_UNALIGNED(b) || IS_UNALIGNED(c)) {
            //std::cout << "slow\n";
            // CALL SLOW CODE
            for (long i=0; i<dimi; ++i,c+=dimj,++a) {
                for (long j=0; j<dimj; ++j) c[j] = 0.0;
                const double *ai = a;
                for (long k=0; k<dimk; ++k,ai+=dimi) {
                    double aki = *ai;
                    for (long j=0; j<dimj; ++j) {
                        c[j] += aki*b[k*dimj+j];
                    }
                }
            }
            return;
        }

        /*
           Choice is to unroll i or j
        */

#if   defined(AMD_QUADCORE_TUNE)
        bool test = dimj>=14 && dimj<=26;
#elif defined(OPTERON_TUNE)
        bool test = dimi <= dimj; /* Based on times from X86_64 Opteron ... an old one */
#elif defined(CORE_DUO_TUNE)
        bool test = true; /* Based on times from X86_32 Core Duo ... my old laptop */
#elif (defined(CORE2_TUNE) && defined(X86_32))
        bool test = false; /* Based on times from Core2 running in 32-bit mode ... a sad thing */
#elif (defined(CORE2_TUNE) && defined(X86_64))
        bool test = dimj > 12 || dimi <= dimj; /* Based on times from X86_64 Core2 */
#else
        bool test = dimj > 12 || dimi <= dimj; /* Based on times from X86_64 Core2 */
#endif
        if (test) {
            long nj = dimj;
            do {
#ifdef X86_64
                long numj = (nj>26) ? 26 : nj;
#else
                long numj = (nj>10) ? 10 : nj;
#endif

                switch (numj) {
#ifdef X86_64
                case 26:
                    TmTxm26(dimj, dimi, dimk, c, b, a) ;
                    break;

                case 24:
                    TmTxm24(dimj, dimi, dimk, c, b, a) ;
                    break;

                case 22:
                    TmTxm22(dimj, dimi, dimk, c, b, a) ;
                    break;

                case 20:
                    TmTxm20(dimj, dimi, dimk, c, b, a) ;
                    break;

                case 18:
                    TmTxm18(dimj, dimi, dimk, c, b, a) ;
                    break;

                case 16:
                    TmTxm16(dimj, dimi, dimk, c, b, a) ;
                    break;

                case 14:
                    TmTxm14(dimj, dimi, dimk, c, b, a) ;
                    break;

                case 12:
                    TmTxm12(dimj, dimi, dimk, c, b, a) ;
                    break;
#endif // X86_64

                case 10:
                    TmTxm10(dimj, dimi, dimk, c, b, a) ;
                    break;

                case 8:
                    TmTxm8(dimj, dimi, dimk, c, b, a) ;
                    break;

                case 6:
                    TmTxm6(dimj, dimi, dimk, c, b, a) ;
                    break;

                case 4:
                    TmTxm4(dimj, dimi, dimk, c, b, a) ;
                    break;

                case 2:
                    TmTxm2(dimj, dimi, dimk, c, b, a) ;
                    break;

                default:
                    throw "mtxmq_byj: should not be here";

                }
                nj -= numj;
                c += numj;
                b += numj;
            } while (nj);
        }
        else {
            long ni = dimi;
            do {
#ifdef X86_64
                long numi = (ni>26) ? 26 : ni;
#else
                long numi = (ni>10) ? 10 : ni;
#endif

                switch (numi) {
#ifdef X86_64
                case 26:
                    mTxm26(dimi, dimj, dimk, c, a, b) ;
                    break;

                case 24:
                    mTxm24(dimi, dimj, dimk, c, a, b) ;
                    break;

                case 22:
                    mTxm22(dimi, dimj, dimk, c, a, b) ;
                    break;

                case 20:
                    mTxm20(dimi, dimj, dimk, c, a, b) ;
                    break;

                case 18:
                    mTxm18(dimi, dimj, dimk, c, a, b) ;
                    break;

                case 16:
                    mTxm16(dimi, dimj, dimk, c, a, b) ;
                    break;

                case 14:
                    mTxm14(dimi, dimj, dimk, c, a, b) ;
                    break;

                case 12:
                    mTxm12(dimi, dimj, dimk, c, a, b) ;
                    break;
#endif // X86_64

                case 10:
                    mTxm10(dimi, dimj, dimk, c, a, b) ;
                    break;

                case 8:
                    mTxm8(dimi, dimj, dimk, c, a, b) ;
                    break;

                case 6:
                    mTxm6(dimi, dimj, dimk, c, a, b) ;
                    break;

                case 4:
                    mTxm4(dimi, dimj, dimk, c, a, b) ;
                    break;

                case 2:
                    mTxm2(dimi, dimj, dimk, c, a, b) ;
                    break;

                default:
                    throw "mtxmq: should not be here!";
                }
                ni -= numi;
                c += numi*dimj;
                a += numi;
            } while (ni);

        }
    }
}

#endif // defined(X86_32) || defined(X86_64)







