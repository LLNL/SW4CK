#ifdef ENABLE_HIP
#include "hip/hip_ext.h"
#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"
#endif
#define float_sw4 double
template<int NX, int NY,int NZ>
#ifdef ENABLE_HIP
__launch_bounds__(256,2) 
#endif
__global__ void K2kernel(int start0, int N0, int start1, int N1, int start2, int N2,
			 const float_sw4* const __restrict__ a_u, float_sw4* __restrict__ a_mu,
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
  

  const int i = start0 + threadIdx.x + blockIdx.x * blockDim.x;
  const int j = start1 + threadIdx.y + blockIdx.y * blockDim.y;
  const int k = start2 + threadIdx.z + blockIdx.z * blockDim.z;

  // Check is making these const ints makes a difference
#define TX threadIdx.x
#define TY threadIdx.y
#define TZ threadIdx.z

#define TX2 (threadIdx.x+2)
#define TY2 (threadIdx.y+2)
#define TZ2 (threadIdx.z+2)

  //define NX 64
    //#define NY 2
  //#define NZ 2

  __shared__ float_sw4 su[NX+4][NY+4][NZ+4];

  // Block stride loops ?

#ifdef TEST_METHOD
  if (i>(N0+3)) return;
  int ic=0;
  if (((blockIdx.y+blockIdx.z+threadIdx.y + threadIdx.z)==0)&&(blockIdx.x==0)){
    int tid=TX;
    int b = blockIdx.x;
    //int bd=blockDim.x;
    for(int ii=TX;ii<NX+4;ii+=blockDim.x){
      int I=start0 + threadIdx.x + (blockIdx.x+ic)* blockDim.x-2;
      if (I<(N0+2)) printf("DUMP %d SARRAY INDEX = %d(%d) -> TID = %d I = %d END = %d\n",b,ii,ic,tid,I,N0);
      ic=ic+1;
    }
}
return
#else
  int ic=0;
  for(int ii=TX;ii<NX+4;ii+=NX){
    int I=start0 + threadIdx.x + (blockIdx.x+ic)* blockDim.x-2;
    int jc=0;
    for (int jj=TY;jj<NY+4;jj+=NY){
      int J=start1+threadIdx.y + (blockIdx.y+jc)* blockDim.y-2;
      int kc=0;
      for(int kk=TZ;kk<NZ+4;kk+=NZ){
	int K=start2+threadIdx.z + (blockIdx.z+kc)* blockDim.z-2;
	kc++;
	if ((I>(N0+1))||(J>(N1+1))||(K>(N2+1))) continue;
	su[ii][jj][kk]=u(2,I,J,K);

      }
      jc++;
    }
    ic++;
  }
#endif
  
  __syncthreads();

  if ((i < N0) && (j < N1) && (k < N2)) {


  
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

#ifdef DEBUG
	  int t1= TX2;
	  int t2 = TY2;
	  int t3 = TZ2;

	  double diff = u(2, i, j, k)-su[TX2][TY2][TZ2] ;
	  int bb = blockIdx.x;
	  if ((diff!=0.0)&&((blockIdx.x+blockIdx.y+blockIdx.z)==0)) printf("COMP %d %d %d -> %d %d %d = %g\n",i,j,k,t1,t2,t3,diff);
#endif

          r1 += i6 * (mux1 * (su[TX2][TY2][TZ] - su[TX2][TY2][TZ2]) +
                      mux2 * (su[TX2][TY2][TZ+1] - su[TX2][TY2][TZ2]) +
                      mux3 * (su[TX2][TY2][TZ2+1] - su[TX2][TY2][TZ2]) +
                      mux4 * (su[TX2][TY2][TZ2+2] - su[TX2][TY2][TZ2]));

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
          r1 +=
              c2 *
                  (mu(i, j + 2, k) * met(1, i, j + 2, k) * met(1, i, j + 2, k) *
                       (c2 * (su[TX2+2][TY2+2][TZ2] - su[TX][TY2+2][TZ2]) +
                        c1 * (su[TX2+1][TY2+2][TZ2] - su[TX+1][TY2+2][TZ2])) -
                   mu(i, j - 2, k) * met(1, i, j - 2, k) * met(1, i, j - 2, k) *
                       (c2 * (su[TX2+2][TY][TZ2] - su[TX][TY][TZ2]) +
                        c1 * (su[TX2+1][TY][TZ2] - su[TX+1][TY][TZ2]))) +
              c1 *
                  (mu(i, j + 1, k) * met(1, i, j + 1, k) * met(1, i, j + 1, k) *
                       (c2 * (su[TX2+2][TY2+1][TZ2] - su[TX][TY2+1][TZ2]) +
                        c1 * (su[TX2+1][TY2+1][TZ2] - su[TX+1][TY2+1][TZ2])) -
                   mu(i, j - 1, k) * met(1, i, j - 1, k) * met(1, i, j - 1, k) *
                       (c2 * (su[TX2+2][TY+1][TZ2] - su[TX][TY+1][TZ2]) +
                        c1 * (su[TX2+1][TY+1][TZ2] - su[TX+1][TY+1][TZ2])));

          // qp-derivatives
          // 38 ops, tot=345
          r1 +=
              c2 *
                  (la(i + 2, j, k) * met(1, i + 2, j, k) * met(1, i + 2, j, k) *
                       (c2 * (su[TX2+2][TY2+2][TZ2] - su[TX2+2][TY][TZ2]) +
	      c1 * (su[TX2+2][TY2+1][TZ2] - su[TX2+2][TY+1][TZ2])) - // Time jumped from 1.89 to 2.33 with the 2nd convertion in this line
                   la(i - 2, j, k) * met(1, i - 2, j, k) * met(1, i - 2, j, k) *
                       (c2 * (su[TX][TY2+2][TZ2] - su[TX][TY][TZ2]) +
                        c1 * (su[TX][TY2+1][TZ2] - su[TX][TY+1][TZ2]))) +
              c1 *
                  (la(i + 1, j, k) * met(1, i + 1, j, k) * met(1, i + 1, j, k) *
                       (c2 * (su[TX2+1][TY2+2][TZ2] - su[TX2+1][TY][TZ2]) +
                        c1 * (su[TX2+1][TY2+1][TZ2] - su[TX2+1][TY+1][TZ2])) -
                   la(i - 1, j, k) * met(1, i - 1, j, k) * met(1, i - 1, j, k) *
		   (c2 * (su[TX+1][TY2+2][TZ2] - su[TX+1][TY][TZ2]) +
                        c1 * (su[TX+1][TY2+1][TZ2] - su[TX+1][TY+1][TZ2])));

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
                       (c2 * (su[TX2+2][TY2][TZ2+2] - su[TX][TY2][TZ2+2]) +
                        c1 * (su[TX2+1][TY2][TZ2+2] - su[TX+1][TY2][TZ2+2])) +
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
                        (c2 * (su[TX2+2][TY2][TZ] - su[TX][TY2][TZ]) +
                         c1 * (su[TX2+1][TY2][TZ] - su[TX+1][TY2][TZ])) +
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
                       (c2 * (su[TX2+2][TY2][TZ2+1] - su[TX][TY2][TZ2+1]) +
                        c1 * (su[TX2+1][TY2][TZ2+1] - su[TX+1][TY2][TZ2+1])) +
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
                        (c2 * (su[TX2+2][TY2][TZ+1] - su[TX][TY2][TZ+1]) +
                         c1 * (su[TX2+1][TY2][TZ+1] - su[TX+1][TY2][TZ+1])) +
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
                        (c2 * (su[TX2+2][TY2][TZ2+2] - su[TX2+2][TY2][TZ]) +
                         c1 * (su[TX2+2][TY2][TZ2+1] - su[TX2+2][TY2][TZ+1])) *
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
                         (c2 * (su[TX][TY2][TZ2+2] - su[TX][TY2][TZ]) +
                          c1 *
                              (su[TX][TY2][TZ2+1] - su[TX][TY2][TZ+1])) *
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
                        (c2 * (su[TX2+1][TY2][TZ2+2] - su[TX2+1][TY2][TZ]) +
                         c1 * (su[TX2+1][TY2][TZ2+1] - su[TX2+1][TY2][TZ+1])) *
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
                         (c2 * (su[TX+1][TY2][TZ2+2] - su[TX+1][TY2][TZ]) +
                          c1 *
                              (su[TX+1][TY2][TZ2+1] - su[TX+1][TY2][TZ+1])) *
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
                       (c2 * (su[TX2][TY2+2][TZ2+2] - su[TX2][TY][TZ2+2]) +
                        c1 * (su[TX2][TY2+1][TZ2+2] - su[TX2][TY+1][TZ2+2])) -
                   (mu(i, j, k - 2) * met(3, i, j, k - 2) *
                        met(1, i, j, k - 2) *
                        (c2 * (u(1, i, j + 2, k - 2) - u(1, i, j - 2, k - 2)) +
                         c1 * (u(1, i, j + 1, k - 2) - u(1, i, j - 1, k - 2))) *
                        stry(j) * istrx +
                    la(i, j, k - 2) * met(2, i, j, k - 2) *
                        met(1, i, j, k - 2) *
                        (c2 * (su[TX2][TY2+2][TZ] - su[TX2][TY][TZ]) +
                         c1 * (su[TX2][TY2+1][TZ] -
                               su[TX2][TY+1][TZ])))) +
              c1 *
                  (mu(i, j, k + 1) * met(3, i, j, k + 1) * met(1, i, j, k + 1) *
                       (c2 * (u(1, i, j + 2, k + 1) - u(1, i, j - 2, k + 1)) +
                        c1 * (u(1, i, j + 1, k + 1) - u(1, i, j - 1, k + 1))) *
                       stry(j) * istrx +
                   la(i, j, k + 1) * met(2, i, j, k + 1) * met(1, i, j, k + 1) *
                       (c2 * (su[TX2][TY2+2][TZ2+1] - su[TX2][TY][TZ2+1]) +
                        c1 * (su[TX2][TY2+1][TZ2+1] - su[TX2][TY+1][TZ2+1])) -
                   (mu(i, j, k - 1) * met(3, i, j, k - 1) *
                        met(1, i, j, k - 1) *
                        (c2 * (u(1, i, j + 2, k - 1) - u(1, i, j - 2, k - 1)) +
                         c1 * (u(1, i, j + 1, k - 1) - u(1, i, j - 1, k - 1))) *
                        stry(j) * istrx +
                    la(i, j, k - 1) * met(2, i, j, k - 1) *
                        met(1, i, j, k - 1) *
                        (c2 * (su[TX2][TY2+2][TZ+1]- su[TX2][TY][TZ+1]) +
                         c1 *
                             (su[TX2][TY2+1][TZ+1] - su[TX2][TY+1][TZ+1]))));

          // rq derivatives
          // 82 ops, tot=769
          r1 +=
              c2 *
                  (mu(i, j + 2, k) * met(3, i, j + 2, k) * met(1, i, j + 2, k) *
                       (c2 * (u(1, i, j + 2, k + 2) - u(1, i, j + 2, k - 2)) +
                        c1 * (u(1, i, j + 2, k + 1) - u(1, i, j + 2, k - 1))) *
                       stry(j + 2) * istrx +
                   mu(i, j + 2, k) * met(2, i, j + 2, k) * met(1, i, j + 2, k) *
                       (c2 * (su[TX2][TY2+2][TZ2+2] - su[TX2][TY2+2][TZ]) +
                        c1 * (su[TX2][TY2+2][TZ2+1] - su[TX2][TY2+2][TZ+1])) -
                   (mu(i, j - 2, k) * met(3, i, j - 2, k) *
                        met(1, i, j - 2, k) *
                        (c2 * (u(1, i, j - 2, k + 2) - u(1, i, j - 2, k - 2)) +
                         c1 * (u(1, i, j - 2, k + 1) - u(1, i, j - 2, k - 1))) *
                        stry(j - 2) * istrx +
                    mu(i, j - 2, k) * met(2, i, j - 2, k) *
                        met(1, i, j - 2, k) *
                        (c2 * (su[TX2][TY][TZ2+2] - su[TX2][TY][TZ]) +
                         c1 * (su[TX2][TY][TZ2+1] -
                               su[TX2][TY][TZ+1])))) +
              c1 *
                  (mu(i, j + 1, k) * met(3, i, j + 1, k) * met(1, i, j + 1, k) *
                       (c2 * (u(1, i, j + 1, k + 2) - u(1, i, j + 1, k - 2)) +
                        c1 * (u(1, i, j + 1, k + 1) - u(1, i, j + 1, k - 1))) *
                       stry(j + 1) * istrx +
                   mu(i, j + 1, k) * met(2, i, j + 1, k) * met(1, i, j + 1, k) *
                       (c2 * (su[TX2][TY2+1][TZ2+2] - su[TX2][TY2+1][TZ]) +
                        c1 * (su[TX2][TY2+1][TZ2+1] - su[TX2][TY2+1][TZ+1])) -
                   (mu(i, j - 1, k) * met(3, i, j - 1, k) *
                        met(1, i, j - 1, k) *
                        (c2 * (u(1, i, j - 1, k + 2) - u(1, i, j - 1, k - 2)) +
                         c1 * (u(1, i, j - 1, k + 1) - u(1, i, j - 1, k - 1))) *
                        stry(j - 1) * istrx +
                    mu(i, j - 1, k) * met(2, i, j - 1, k) *
                        met(1, i, j - 1, k) *
                        (c2 * (su[TX2][TY+1][TZ2+2] - su[TX2][TY+1][TZ]) +
                         c1 *
                             (su[TX2][TY+1][TZ2+1] - su[TX2][TY+1][TZ+1]))));

          // 4 ops, tot=773
          lu(1, i, j, k) = a1 * lu(1, i, j, k) + sgn * r1 * ijac;
  }
}
