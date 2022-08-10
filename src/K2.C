#include "hip/hip_ext.h"
#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"
#define float_sw4 double
__launch_bounds__(256,2) 
__global__ void K2kernel(int start0, int N0, int start1, int N1, int start2, int N2,
			 float_sw4* __restrict__ a_u, float_sw4* __restrict__ a_mu,
			 float_sw4* __restrict__ a_lambda, float_sw4* __restrict__ a_met,
			 float_sw4* __restrict__ a_jac,float_sw4* __restrict__ a_lu,
			 float_sw4* __restrict__ a_strx, float_sw4* __restrict__ a_stry,
			 int ifirst,int ilast,int jfirst,int jlast,int kfirst, int klast,float_sw4 a1, float_sw4 sgn){

  const float_sw4 i6 = 1.0 / 6;
  const float_sw4 tf = 0.75;
  const float_sw4 c1 = 2.0 / 3;
  const float_sw4 c2 = -1.0 / 12;

  const int ni = ilast - ifirst + 1;
  const int nij = ni * (jlast - jfirst + 1);
  const int nijk = nij * (klast - kfirst + 1);
  const int base = -(ifirst + ni * jfirst + nij * kfirst);
  const int base3 = base - nijk;
  const int base4 = base - nijk;
  const int ifirst0 = ifirst;
  const int jfirst0 = jfirst;

#define mu(i, j, k) a_mu[base + (i) + ni * (j) + nij * (k)]
#define la(i, j, k) a_lambda[base + (i) + ni * (j) + nij * (k)]
#define jac(i, j, k) a_jac[base + (i) + ni * (j) + nij * (k)]
#define u(c, i, j, k) a_u[base3 + (i) + ni * (j) + nij * (k) + nijk * (c)]
#define lu(c, i, j, k) a_lu[base3 + (i) + ni * (j) + nij * (k) + nijk * (c)]
#define met(c, i, j, k) a_met[base4 + (i) + ni * (j) + nij * (k) + nijk * (c)]
#define strx(i) a_strx[i - ifirst0]
#define stry(j) a_stry[j - jfirst0]
  

  int i = start0 + threadIdx.x + blockIdx.x * blockDim.x;
  int j = start1 + threadIdx.y + blockIdx.y * blockDim.y;
  int ii=threadIdx.x+2;
  int jj=threadIdx.y+2;
  __shared__ float_sw4 su[20][20]; // Hardwired for 16x16 blocks PBUGS
  //  int k = start2 + threadIdx.z + blockIdx.z * blockDim.z;
  if ((i < N0) && (j < N1)) {
    for(int k=start2;k<N2;k++){
  
      su[threadIdx.x+2][threadIdx.y+2]=u(2,i,j,k);

      if (threadIdx.x==0){
	su[threadIdx.x][threadIdx.y+2]=u(2,i-2,j,k);
	su[threadIdx.x+1][threadIdx.y+2]=u(2,i-1,j,k);
	if (threadIdx.y==0){
	  su[threadIdx.x][threadIdx.y]=u(2,i-2,j-2,k);
	  su[threadIdx.x][threadIdx.y+1]=u(2,i-2,j-1,k);
	  su[threadIdx.x+1][threadIdx.y]=u(2,i-1,j-2,k);
	  su[threadIdx.x+1][threadIdx.y+1]=u(2,i-1,j-1,k);
	}
      }
      if (threadIdx.x==15){
	su[threadIdx.x+3][threadIdx.y+2]=u(2,i+1,j,k);
	su[threadIdx.x+4][threadIdx.y+2]=u(2,i+2,j,k);
	if (threadIdx.y==15){
	  su[threadIdx.x+3][threadIdx.y+3]=u(2,i+1,j+1,k);
	  su[threadIdx.x+4][threadIdx.y+4]=u(2,i+2,j+2,k);
	  su[threadIdx.x+3][threadIdx.y+4]=u(2,i+1,j+2,k);
	  su[threadIdx.x+4][threadIdx.y+3]=u(2,i+2,j+1,k);
	}
      }
      if (threadIdx.y==0){
	su[threadIdx.x+2][threadIdx.y]=  u(2,i,j-2,k);
	su[threadIdx.x+2][threadIdx.y+1]=u(2,i,j-1,k);
	if (threadIdx.x==15){
	  su[threadIdx.x+4][threadIdx.y]=u(2,i+2,j-2,k);
	  su[threadIdx.x+3][threadIdx.y+1] = u(2,i+1,j-1,k);
	  su[threadIdx.x+3][threadIdx.y]=u(2,i+1,j-2,k);
	  su[threadIdx.x+4][threadIdx.y+1] = u(2,i+2,j-1,k);
	}
      }
      if (threadIdx.y==15){
	su[threadIdx.x+2][threadIdx.y+3]=u(2,i,j+1,k);
	su[threadIdx.x+2][threadIdx.y+4]=u(2,i,j+2,k);
	if (threadIdx.x==0){
	  su[threadIdx.x][threadIdx.y+4]=u(2,i-2,j+2,k);
	  su[threadIdx.x+1][threadIdx.y+3]=u(2,i-1,j+1,k);
	  su[threadIdx.x+1][threadIdx.y+4]=u(2,i-1,j+2,k);
	  su[threadIdx.x][threadIdx.y+3]=u(2,i-2,j+1,k);
	}
      }
      int b1 = blockIdx.x;
      int b2 = blockIdx.y;
      __syncthreads();
      // Halos of partial blocks
      if (i==(N0-1)){
	su[threadIdx.x+3][threadIdx.y+2]=u(2,i+1,j,k);
	su[threadIdx.x+4][threadIdx.y+2]=u(2,i+2,j,k);
	if (j==(N1-1)){
	  su[threadIdx.x+3][threadIdx.y+3]=u(2,i+1,j+1,k);
	  su[threadIdx.x+4][threadIdx.y+4]=u(2,i+2,j+2,k);
	  su[threadIdx.x+3][threadIdx.y+4]=u(2,i+1,j+2,k);
	  su[threadIdx.x+4][threadIdx.y+3]=u(2,i+2,j+1,k);
	}
	if (threadIdx.y==0){
	  su[threadIdx.x+4][threadIdx.y]=    u(2,i+2,j-2,k);
	  su[threadIdx.x+3][threadIdx.y+1] = u(2,i+1,j-1,k);
	  su[threadIdx.x+3][threadIdx.y]=    u(2,i+1,j-2,k);
	  su[threadIdx.x+4][threadIdx.y+1] = u(2,i+2,j-1,k);
	}
	if (threadIdx.y==15){
	  su[threadIdx.x+3][threadIdx.y+3]=u(2,i+1,j+1,k);
	  su[threadIdx.x+4][threadIdx.y+4]=u(2,i+2,j+2,k);
	  su[threadIdx.x+3][threadIdx.y+4]=u(2,i+1,j+2,k);
	  su[threadIdx.x+4][threadIdx.y+3]=u(2,i+2,j+1,k);
	}
	  
      }
      if (j==(N1-1)){
	su[threadIdx.x+2][threadIdx.y+3]=u(2,i,j+1,k);
	su[threadIdx.x+2][threadIdx.y+4]=u(2,i,j+2,k);
	if (threadIdx.x==0){
	  //if (k==start2) printf("FIXING BLOCK %d %d -> %d %d %d\n",b1,b2,ii,jj,k);
	  su[threadIdx.x][threadIdx.y+4]=u(2,i-2,j+2,k);
	  su[threadIdx.x+1][threadIdx.y+3]=u(2,i-1,j+1,k);
	  su[threadIdx.x+1][threadIdx.y+4]=u(2,i-1,j+2,k);
	  su[threadIdx.x][threadIdx.y+3]=u(2,i-2,j+1,k);
	}
	if (threadIdx.x==15){
	  su[threadIdx.x+3][threadIdx.y+3]=u(2,i+1,j+1,k);
	  su[threadIdx.x+4][threadIdx.y+4]=u(2,i+2,j+2,k);
	  su[threadIdx.x+3][threadIdx.y+4]=u(2,i+1,j+2,k);
	  su[threadIdx.x+4][threadIdx.y+3]=u(2,i+2,j+1,k);
	}
      }
	
      __syncthreads();
  // #pragma ivdep
          // 	 for( int i=ifirst+2; i <= ilast-2 ; i++ )
          // 	 {
          // 5 ops
          float_sw4 ijac = strx(i) * stry(j) / jac(i, j, k);
          float_sw4 istry = 1 / (stry(j));
          float_sw4 istrx = 1 / (strx(i));
          float_sw4 istrxy = istry * istrx;

          float_sw4 r1 = 0;

          // pp derivative (u)
          // 53 ops, tot=58
          float_sw4 cof1 = (2 * mu(i - 2, j, k) + la(i - 2, j, k)) *
                           met(1, i - 2, j, k) * met(1, i - 2, j, k) *
                           strx(i - 2);
          float_sw4 cof2 = (2 * mu(i - 1, j, k) + la(i - 1, j, k)) *
                           met(1, i - 1, j, k) * met(1, i - 1, j, k) *
                           strx(i - 1);
          float_sw4 cof3 = (2 * mu(i, j, k) + la(i, j, k)) * met(1, i, j, k) *
                           met(1, i, j, k) * strx(i);
          float_sw4 cof4 = (2 * mu(i + 1, j, k) + la(i + 1, j, k)) *
                           met(1, i + 1, j, k) * met(1, i + 1, j, k) *
                           strx(i + 1);
          float_sw4 cof5 = (2 * mu(i + 2, j, k) + la(i + 2, j, k)) *
                           met(1, i + 2, j, k) * met(1, i + 2, j, k) *
                           strx(i + 2);
          float_sw4 mux1 = cof2 - tf * (cof3 + cof1);
          float_sw4 mux2 = cof1 + cof4 + 3 * (cof3 + cof2);
          float_sw4 mux3 = cof2 + cof5 + 3 * (cof4 + cof3);
          float_sw4 mux4 = cof4 - tf * (cof3 + cof5);

          r1 += i6 *
                (mux1 * (u(1, i - 2, j, k) - u(1, i, j, k)) +
                 mux2 * (u(1, i - 1, j, k) - u(1, i, j, k)) +
                 mux3 * (u(1, i + 1, j, k) - u(1, i, j, k)) +
                 mux4 * (u(1, i + 2, j, k) - u(1, i, j, k))) *
                istry;
          // qq derivative (u)
          // 43 ops, tot=101
          {
            float_sw4 cof1 = (mu(i, j - 2, k)) * met(1, i, j - 2, k) *
                             met(1, i, j - 2, k) * stry(j - 2);
            float_sw4 cof2 = (mu(i, j - 1, k)) * met(1, i, j - 1, k) *
                             met(1, i, j - 1, k) * stry(j - 1);
            float_sw4 cof3 =
                (mu(i, j, k)) * met(1, i, j, k) * met(1, i, j, k) * stry(j);
            float_sw4 cof4 = (mu(i, j + 1, k)) * met(1, i, j + 1, k) *
                             met(1, i, j + 1, k) * stry(j + 1);
            float_sw4 cof5 = (mu(i, j + 2, k)) * met(1, i, j + 2, k) *
                             met(1, i, j + 2, k) * stry(j + 2);
            float_sw4 mux1 = cof2 - tf * (cof3 + cof1);
            float_sw4 mux2 = cof1 + cof4 + 3 * (cof3 + cof2);
            float_sw4 mux3 = cof2 + cof5 + 3 * (cof4 + cof3);
            float_sw4 mux4 = cof4 - tf * (cof3 + cof5);

            r1 += i6 *
                  (mux1 * (u(1, i, j - 2, k) - u(1, i, j, k)) +
                   mux2 * (u(1, i, j - 1, k) - u(1, i, j, k)) +
                   mux3 * (u(1, i, j + 1, k) - u(1, i, j, k)) +
                   mux4 * (u(1, i, j + 2, k) - u(1, i, j, k))) *
                  istrx;
          }
          // rr derivative (u)
          // 5*11+14+14=83 ops, tot=184
          {
            float_sw4 cof1 =
                (2 * mu(i, j, k - 2) + la(i, j, k - 2)) * met(2, i, j, k - 2) *
                    strx(i) * met(2, i, j, k - 2) * strx(i) +
                mu(i, j, k - 2) * (met(3, i, j, k - 2) * stry(j) *
                                       met(3, i, j, k - 2) * stry(j) +
                                   met(4, i, j, k - 2) * met(4, i, j, k - 2));
            float_sw4 cof2 =
                (2 * mu(i, j, k - 1) + la(i, j, k - 1)) * met(2, i, j, k - 1) *
                    strx(i) * met(2, i, j, k - 1) * strx(i) +
                mu(i, j, k - 1) * (met(3, i, j, k - 1) * stry(j) *
                                       met(3, i, j, k - 1) * stry(j) +
                                   met(4, i, j, k - 1) * met(4, i, j, k - 1));
            float_sw4 cof3 = (2 * mu(i, j, k) + la(i, j, k)) * met(2, i, j, k) *
                                 strx(i) * met(2, i, j, k) * strx(i) +
                             mu(i, j, k) * (met(3, i, j, k) * stry(j) *
                                                met(3, i, j, k) * stry(j) +
                                            met(4, i, j, k) * met(4, i, j, k));
            float_sw4 cof4 =
                (2 * mu(i, j, k + 1) + la(i, j, k + 1)) * met(2, i, j, k + 1) *
                    strx(i) * met(2, i, j, k + 1) * strx(i) +
                mu(i, j, k + 1) * (met(3, i, j, k + 1) * stry(j) *
                                       met(3, i, j, k + 1) * stry(j) +
                                   met(4, i, j, k + 1) * met(4, i, j, k + 1));
            float_sw4 cof5 =
                (2 * mu(i, j, k + 2) + la(i, j, k + 2)) * met(2, i, j, k + 2) *
                    strx(i) * met(2, i, j, k + 2) * strx(i) +
                mu(i, j, k + 2) * (met(3, i, j, k + 2) * stry(j) *
                                       met(3, i, j, k + 2) * stry(j) +
                                   met(4, i, j, k + 2) * met(4, i, j, k + 2));

            float_sw4 mux1 = cof2 - tf * (cof3 + cof1);
            float_sw4 mux2 = cof1 + cof4 + 3 * (cof3 + cof2);
            float_sw4 mux3 = cof2 + cof5 + 3 * (cof4 + cof3);
            float_sw4 mux4 = cof4 - tf * (cof3 + cof5);

            r1 += i6 *
                  (mux1 * (u(1, i, j, k - 2) - u(1, i, j, k)) +
                   mux2 * (u(1, i, j, k - 1) - u(1, i, j, k)) +
                   mux3 * (u(1, i, j, k + 1) - u(1, i, j, k)) +
                   mux4 * (u(1, i, j, k + 2) - u(1, i, j, k))) *
                  istrxy;
          }
          // rr derivative (v)
          // 42 ops, tot=226
          cof1 = (mu(i, j, k - 2) + la(i, j, k - 2)) * met(2, i, j, k - 2) *
                 met(3, i, j, k - 2);
          cof2 = (mu(i, j, k - 1) + la(i, j, k - 1)) * met(2, i, j, k - 1) *
                 met(3, i, j, k - 1);
          cof3 =
              (mu(i, j, k) + la(i, j, k)) * met(2, i, j, k) * met(3, i, j, k);
          cof4 = (mu(i, j, k + 1) + la(i, j, k + 1)) * met(2, i, j, k + 1) *
                 met(3, i, j, k + 1);
          cof5 = (mu(i, j, k + 2) + la(i, j, k + 2)) * met(2, i, j, k + 2) *
                 met(3, i, j, k + 2);
          mux1 = cof2 - tf * (cof3 + cof1);
          mux2 = cof1 + cof4 + 3 * (cof3 + cof2);
          mux3 = cof2 + cof5 + 3 * (cof4 + cof3);
          mux4 = cof4 - tf * (cof3 + cof5);

          r1 += i6 * (mux1 * (u(2, i, j, k - 2) - su[ii][jj]) +
                      mux2 * (u(2, i, j, k - 1) - su[ii][jj]) +
                      mux3 * (u(2, i, j, k + 1) - su[ii][jj]) +
                      mux4 * (u(2, i, j, k + 2) - su[ii][jj]));

          // rr derivative (w)
          // 43 ops, tot=269
          cof1 = (mu(i, j, k - 2) + la(i, j, k - 2)) * met(2, i, j, k - 2) *
                 met(4, i, j, k - 2);
          cof2 = (mu(i, j, k - 1) + la(i, j, k - 1)) * met(2, i, j, k - 1) *
                 met(4, i, j, k - 1);
          cof3 =
              (mu(i, j, k) + la(i, j, k)) * met(2, i, j, k) * met(4, i, j, k);
          cof4 = (mu(i, j, k + 1) + la(i, j, k + 1)) * met(2, i, j, k + 1) *
                 met(4, i, j, k + 1);
          cof5 = (mu(i, j, k + 2) + la(i, j, k + 2)) * met(2, i, j, k + 2) *
                 met(4, i, j, k + 2);
          mux1 = cof2 - tf * (cof3 + cof1);
          mux2 = cof1 + cof4 + 3 * (cof3 + cof2);
          mux3 = cof2 + cof5 + 3 * (cof4 + cof3);
          mux4 = cof4 - tf * (cof3 + cof5);

          r1 += i6 *
                (mux1 * (u(3, i, j, k - 2) - u(3, i, j, k)) +
                 mux2 * (u(3, i, j, k - 1) - u(3, i, j, k)) +
                 mux3 * (u(3, i, j, k + 1) - u(3, i, j, k)) +
                 mux4 * (u(3, i, j, k + 2) - u(3, i, j, k))) *
                istry;

          // pq-derivatives

          // 38 ops, tot=307
	  //#define DEBUG 1
#ifdef DEBUG
	  if (k==start2){
	    
	    float_sw4 diff=(u(2,i-2,j-2,k)-su[ii-2][jj-2]);
	    if (diff!=0.0)
	      printf(" BOOM BUD = (%d , %d) (%d %d, %d, %d, %g\n",b1,b2,i,j,ii,jj,diff);
	  }
#endif
          r1 +=
              c2 *
                  (mu(i, j + 2, k) * met(1, i, j + 2, k) * met(1, i, j + 2, k) *
		   (c2 * (su[ii+2][jj+2] - su[ii-2][jj+2]) +
                        c1 * (su[ii + 1][ jj + 2] - su[ii - 1][ jj + 2])) -
                   mu(i, j - 2, k) * met(1, i, j - 2, k) * met(1, i, j - 2, k) *
                       (c2 * (su[ii + 2][jj - 2] - su[ii - 2][ jj - 2]) +
                        c1 * (su[ii + 1][ jj - 2] - su[ii - 1][ jj - 2]))) +
              c1 *
                  (mu(i, j + 1, k) * met(1, i, j + 1, k) * met(1, i, j + 1, k) *
                       (c2 * (su[ii + 2][ jj + 1] - su[ii - 2][ jj + 1]) +
                        c1 * (su[ii + 1][jj + 1] - su[ii - 1][jj + 1])) -
                   mu(i, j - 1, k) * met(1, i, j - 1, k) * met(1, i, j - 1, k) *
                       (c2 * (su[ii + 2][jj - 1] - su[ii - 2][ jj - 1]) +
                        c1 * (su[ii + 1][ jj - 1] - su[ii - 1][ jj - 1])));

          // qp-derivatives
          // 38 ops, tot=345
          r1 +=
              c2 *
                  (la(i + 2, j, k) * met(1, i + 2, j, k) * met(1, i + 2, j, k) *
		   (c2 * (su[ii + 2][ jj + 2] - su[ii + 2][jj - 2]) +
                        c1 * (su[ii + 2][jj + 1] - su[ii + 2][jj - 1])) -
                   la(i - 2, j, k) * met(1, i - 2, j, k) * met(1, i - 2, j, k) *
                       (c2 * (su[ii - 2][jj + 2] - su[ii - 2][jj - 2]) +
                        c1 * (su[ii - 2][jj + 1] - su[ii - 2][jj - 1]))) +
              c1 *
                  (la(i + 1, j, k) * met(1, i + 1, j, k) * met(1, i + 1, j, k) *
                       (c2 * (u(2, i + 1, j + 2, k) - u(2, i + 1, j - 2, k)) +
                        c1 * (u(2, i + 1, j + 1, k) - u(2, i + 1, j - 1, k))) -
                   la(i - 1, j, k) * met(1, i - 1, j, k) * met(1, i - 1, j, k) *
                       (c2 * (u(2, i - 1, j + 2, k) - u(2, i - 1, j - 2, k)) +
                        c1 * (u(2, i - 1, j + 1, k) - u(2, i - 1, j - 1, k))));

          // pr-derivatives
          // 130 ops., tot=475
          r1 +=
              c2 *
                  ((2 * mu(i, j, k + 2) + la(i, j, k + 2)) *
                       met(2, i, j, k + 2) * met(1, i, j, k + 2) *
                       (c2 * (u(1, i + 2, j, k + 2) - u(1, i - 2, j, k + 2)) +
                        c1 * (u(1, i + 1, j, k + 2) - u(1, i - 1, j, k + 2))) *
                       strx(i) * istry +
                   mu(i, j, k + 2) * met(3, i, j, k + 2) * met(1, i, j, k + 2) *
                       (c2 * (u(2, i + 2, j, k + 2) - u(2, i - 2, j, k + 2)) +
                        c1 * (u(2, i + 1, j, k + 2) - u(2, i - 1, j, k + 2))) +
                   mu(i, j, k + 2) * met(4, i, j, k + 2) * met(1, i, j, k + 2) *
                       (c2 * (u(3, i + 2, j, k + 2) - u(3, i - 2, j, k + 2)) +
                        c1 * (u(3, i + 1, j, k + 2) - u(3, i - 1, j, k + 2))) *
                       istry -
                   ((2 * mu(i, j, k - 2) + la(i, j, k - 2)) *
                        met(2, i, j, k - 2) * met(1, i, j, k - 2) *
                        (c2 * (u(1, i + 2, j, k - 2) - u(1, i - 2, j, k - 2)) +
                         c1 * (u(1, i + 1, j, k - 2) - u(1, i - 1, j, k - 2))) *
                        strx(i) * istry +
                    mu(i, j, k - 2) * met(3, i, j, k - 2) *
                        met(1, i, j, k - 2) *
                        (c2 * (u(2, i + 2, j, k - 2) - u(2, i - 2, j, k - 2)) +
                         c1 * (u(2, i + 1, j, k - 2) - u(2, i - 1, j, k - 2))) +
                    mu(i, j, k - 2) * met(4, i, j, k - 2) *
                        met(1, i, j, k - 2) *
                        (c2 * (u(3, i + 2, j, k - 2) - u(3, i - 2, j, k - 2)) +
                         c1 * (u(3, i + 1, j, k - 2) - u(3, i - 1, j, k - 2))) *
                        istry)) +
              c1 *
                  ((2 * mu(i, j, k + 1) + la(i, j, k + 1)) *
                       met(2, i, j, k + 1) * met(1, i, j, k + 1) *
                       (c2 * (u(1, i + 2, j, k + 1) - u(1, i - 2, j, k + 1)) +
                        c1 * (u(1, i + 1, j, k + 1) - u(1, i - 1, j, k + 1))) *
                       strx(i) * istry +
                   mu(i, j, k + 1) * met(3, i, j, k + 1) * met(1, i, j, k + 1) *
                       (c2 * (u(2, i + 2, j, k + 1) - u(2, i - 2, j, k + 1)) +
                        c1 * (u(2, i + 1, j, k + 1) - u(2, i - 1, j, k + 1))) +
                   mu(i, j, k + 1) * met(4, i, j, k + 1) * met(1, i, j, k + 1) *
                       (c2 * (u(3, i + 2, j, k + 1) - u(3, i - 2, j, k + 1)) +
                        c1 * (u(3, i + 1, j, k + 1) - u(3, i - 1, j, k + 1))) *
                       istry -
                   ((2 * mu(i, j, k - 1) + la(i, j, k - 1)) *
                        met(2, i, j, k - 1) * met(1, i, j, k - 1) *
                        (c2 * (u(1, i + 2, j, k - 1) - u(1, i - 2, j, k - 1)) +
                         c1 * (u(1, i + 1, j, k - 1) - u(1, i - 1, j, k - 1))) *
                        strx(i) * istry +
                    mu(i, j, k - 1) * met(3, i, j, k - 1) *
                        met(1, i, j, k - 1) *
                        (c2 * (u(2, i + 2, j, k - 1) - u(2, i - 2, j, k - 1)) +
                         c1 * (u(2, i + 1, j, k - 1) - u(2, i - 1, j, k - 1))) +
                    mu(i, j, k - 1) * met(4, i, j, k - 1) *
                        met(1, i, j, k - 1) *
                        (c2 * (u(3, i + 2, j, k - 1) - u(3, i - 2, j, k - 1)) +
                         c1 * (u(3, i + 1, j, k - 1) - u(3, i - 1, j, k - 1))) *
                        istry));

          // rp derivatives
          // 130 ops, tot=605
          r1 +=
              (c2 *
                   ((2 * mu(i + 2, j, k) + la(i + 2, j, k)) *
                        met(2, i + 2, j, k) * met(1, i + 2, j, k) *
                        (c2 * (u(1, i + 2, j, k + 2) - u(1, i + 2, j, k - 2)) +
                         c1 * (u(1, i + 2, j, k + 1) - u(1, i + 2, j, k - 1))) *
                        strx(i + 2) +
                    la(i + 2, j, k) * met(3, i + 2, j, k) *
                        met(1, i + 2, j, k) *
                        (c2 * (u(2, i + 2, j, k + 2) - u(2, i + 2, j, k - 2)) +
                         c1 * (u(2, i + 2, j, k + 1) - u(2, i + 2, j, k - 1))) *
                        stry(j) +
                    la(i + 2, j, k) * met(4, i + 2, j, k) *
                        met(1, i + 2, j, k) *
                        (c2 * (u(3, i + 2, j, k + 2) - u(3, i + 2, j, k - 2)) +
                         c1 * (u(3, i + 2, j, k + 1) - u(3, i + 2, j, k - 1))) -
                    ((2 * mu(i - 2, j, k) + la(i - 2, j, k)) *
                         met(2, i - 2, j, k) * met(1, i - 2, j, k) *
                         (c2 * (u(1, i - 2, j, k + 2) - u(1, i - 2, j, k - 2)) +
                          c1 *
                              (u(1, i - 2, j, k + 1) - u(1, i - 2, j, k - 1))) *
                         strx(i - 2) +
                     la(i - 2, j, k) * met(3, i - 2, j, k) *
                         met(1, i - 2, j, k) *
                         (c2 * (u(2, i - 2, j, k + 2) - u(2, i - 2, j, k - 2)) +
                          c1 *
                              (u(2, i - 2, j, k + 1) - u(2, i - 2, j, k - 1))) *
                         stry(j) +
                     la(i - 2, j, k) * met(4, i - 2, j, k) *
                         met(1, i - 2, j, k) *
                         (c2 * (u(3, i - 2, j, k + 2) - u(3, i - 2, j, k - 2)) +
                          c1 * (u(3, i - 2, j, k + 1) -
                                u(3, i - 2, j, k - 1))))) +
               c1 *
                   ((2 * mu(i + 1, j, k) + la(i + 1, j, k)) *
                        met(2, i + 1, j, k) * met(1, i + 1, j, k) *
                        (c2 * (u(1, i + 1, j, k + 2) - u(1, i + 1, j, k - 2)) +
                         c1 * (u(1, i + 1, j, k + 1) - u(1, i + 1, j, k - 1))) *
                        strx(i + 1) +
                    la(i + 1, j, k) * met(3, i + 1, j, k) *
                        met(1, i + 1, j, k) *
                        (c2 * (u(2, i + 1, j, k + 2) - u(2, i + 1, j, k - 2)) +
                         c1 * (u(2, i + 1, j, k + 1) - u(2, i + 1, j, k - 1))) *
                        stry(j) +
                    la(i + 1, j, k) * met(4, i + 1, j, k) *
                        met(1, i + 1, j, k) *
                        (c2 * (u(3, i + 1, j, k + 2) - u(3, i + 1, j, k - 2)) +
                         c1 * (u(3, i + 1, j, k + 1) - u(3, i + 1, j, k - 1))) -
                    ((2 * mu(i - 1, j, k) + la(i - 1, j, k)) *
                         met(2, i - 1, j, k) * met(1, i - 1, j, k) *
                         (c2 * (u(1, i - 1, j, k + 2) - u(1, i - 1, j, k - 2)) +
                          c1 *
                              (u(1, i - 1, j, k + 1) - u(1, i - 1, j, k - 1))) *
                         strx(i - 1) +
                     la(i - 1, j, k) * met(3, i - 1, j, k) *
                         met(1, i - 1, j, k) *
                         (c2 * (u(2, i - 1, j, k + 2) - u(2, i - 1, j, k - 2)) +
                          c1 *
                              (u(2, i - 1, j, k + 1) - u(2, i - 1, j, k - 1))) *
                         stry(j) +
                     la(i - 1, j, k) * met(4, i - 1, j, k) *
                         met(1, i - 1, j, k) *
                         (c2 * (u(3, i - 1, j, k + 2) - u(3, i - 1, j, k - 2)) +
                          c1 * (u(3, i - 1, j, k + 1) -
                                u(3, i - 1, j, k - 1)))))) *
              istry;

          // qr derivatives
          // 82 ops, tot=687
          r1 +=
              c2 *
                  (mu(i, j, k + 2) * met(3, i, j, k + 2) * met(1, i, j, k + 2) *
                       (c2 * (u(1, i, j + 2, k + 2) - u(1, i, j - 2, k + 2)) +
                        c1 * (u(1, i, j + 1, k + 2) - u(1, i, j - 1, k + 2))) *
                       stry(j) * istrx +
                   la(i, j, k + 2) * met(2, i, j, k + 2) * met(1, i, j, k + 2) *
                       (c2 * (u(2, i, j + 2, k + 2) - u(2, i, j - 2, k + 2)) +
                        c1 * (u(2, i, j + 1, k + 2) - u(2, i, j - 1, k + 2))) -
                   (mu(i, j, k - 2) * met(3, i, j, k - 2) *
                        met(1, i, j, k - 2) *
                        (c2 * (u(1, i, j + 2, k - 2) - u(1, i, j - 2, k - 2)) +
                         c1 * (u(1, i, j + 1, k - 2) - u(1, i, j - 1, k - 2))) *
                        stry(j) * istrx +
                    la(i, j, k - 2) * met(2, i, j, k - 2) *
                        met(1, i, j, k - 2) *
                        (c2 * (u(2, i, j + 2, k - 2) - u(2, i, j - 2, k - 2)) +
                         c1 * (u(2, i, j + 1, k - 2) -
                               u(2, i, j - 1, k - 2))))) +
              c1 *
                  (mu(i, j, k + 1) * met(3, i, j, k + 1) * met(1, i, j, k + 1) *
                       (c2 * (u(1, i, j + 2, k + 1) - u(1, i, j - 2, k + 1)) +
                        c1 * (u(1, i, j + 1, k + 1) - u(1, i, j - 1, k + 1))) *
                       stry(j) * istrx +
                   la(i, j, k + 1) * met(2, i, j, k + 1) * met(1, i, j, k + 1) *
                       (c2 * (u(2, i, j + 2, k + 1) - u(2, i, j - 2, k + 1)) +
                        c1 * (u(2, i, j + 1, k + 1) - u(2, i, j - 1, k + 1))) -
                   (mu(i, j, k - 1) * met(3, i, j, k - 1) *
                        met(1, i, j, k - 1) *
                        (c2 * (u(1, i, j + 2, k - 1) - u(1, i, j - 2, k - 1)) +
                         c1 * (u(1, i, j + 1, k - 1) - u(1, i, j - 1, k - 1))) *
                        stry(j) * istrx +
                    la(i, j, k - 1) * met(2, i, j, k - 1) *
                        met(1, i, j, k - 1) *
                        (c2 * (u(2, i, j + 2, k - 1) - u(2, i, j - 2, k - 1)) +
                         c1 *
                             (u(2, i, j + 1, k - 1) - u(2, i, j - 1, k - 1)))));

          // rq derivatives
          // 82 ops, tot=769
          r1 +=
              c2 *
                  (mu(i, j + 2, k) * met(3, i, j + 2, k) * met(1, i, j + 2, k) *
                       (c2 * (u(1, i, j + 2, k + 2) - u(1, i, j + 2, k - 2)) +
                        c1 * (u(1, i, j + 2, k + 1) - u(1, i, j + 2, k - 1))) *
                       stry(j + 2) * istrx +
                   mu(i, j + 2, k) * met(2, i, j + 2, k) * met(1, i, j + 2, k) *
                       (c2 * (u(2, i, j + 2, k + 2) - u(2, i, j + 2, k - 2)) +
                        c1 * (u(2, i, j + 2, k + 1) - u(2, i, j + 2, k - 1))) -
                   (mu(i, j - 2, k) * met(3, i, j - 2, k) *
                        met(1, i, j - 2, k) *
                        (c2 * (u(1, i, j - 2, k + 2) - u(1, i, j - 2, k - 2)) +
                         c1 * (u(1, i, j - 2, k + 1) - u(1, i, j - 2, k - 1))) *
                        stry(j - 2) * istrx +
                    mu(i, j - 2, k) * met(2, i, j - 2, k) *
                        met(1, i, j - 2, k) *
                        (c2 * (u(2, i, j - 2, k + 2) - u(2, i, j - 2, k - 2)) +
                         c1 * (u(2, i, j - 2, k + 1) -
                               u(2, i, j - 2, k - 1))))) +
              c1 *
                  (mu(i, j + 1, k) * met(3, i, j + 1, k) * met(1, i, j + 1, k) *
                       (c2 * (u(1, i, j + 1, k + 2) - u(1, i, j + 1, k - 2)) +
                        c1 * (u(1, i, j + 1, k + 1) - u(1, i, j + 1, k - 1))) *
                       stry(j + 1) * istrx +
                   mu(i, j + 1, k) * met(2, i, j + 1, k) * met(1, i, j + 1, k) *
                       (c2 * (u(2, i, j + 1, k + 2) - u(2, i, j + 1, k - 2)) +
                        c1 * (u(2, i, j + 1, k + 1) - u(2, i, j + 1, k - 1))) -
                   (mu(i, j - 1, k) * met(3, i, j - 1, k) *
                        met(1, i, j - 1, k) *
                        (c2 * (u(1, i, j - 1, k + 2) - u(1, i, j - 1, k - 2)) +
                         c1 * (u(1, i, j - 1, k + 1) - u(1, i, j - 1, k - 1))) *
                        stry(j - 1) * istrx +
                    mu(i, j - 1, k) * met(2, i, j - 1, k) *
                        met(1, i, j - 1, k) *
                        (c2 * (u(2, i, j - 1, k + 2) - u(2, i, j - 1, k - 2)) +
                         c1 *
                             (u(2, i, j - 1, k + 1) - u(2, i, j - 1, k - 1)))));

          // 4 ops, tot=773
          lu(1, i, j, k) = a1 * lu(1, i, j, k) + sgn * r1 * ijac;
  }
  }
}
