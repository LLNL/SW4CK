//////////////////////////////////////////////////////////////////////////////
//// Copyright (c) 2021, Lawrence Livermore National Security, LLC and SW4CK
//// project contributors. See the COPYRIGHT file for details.
////
//// SPDX-License-Identifier: GPL-2.0-only
////////////////////////////////////////////////////////////////////////////////
#ifndef __OMP_FORALLS_H__
#define __OMP_FORALLS_H__

#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>

std::vector<int> factors(int N);
std::vector<int> factors(int N, int start);
#ifdef ENABLE_OPENMP
template <typename Func>
void forallkernel(int start, int N, Func f) {
#pragma omp parallel for 
  for(int tid=start;tid<N;tid++)
  f(tid);
}
template <typename LoopBody>
void forall(int start, int end, LoopBody &&body) {
	forallkernel(start,end,body);
}
template <typename LoopBody>
void forallasync(int start, int end, LoopBody &&body) {
	forallkernel(start,end,body);
}

template <int N, typename LoopBody>
void forall(int start, int end, LoopBody &&body) {
	forallkernel(start,end,body);
}



template <int N>
class Range {
 public:
  Range(int istart, int iend) : start(istart), end(iend), tpb(N) {
    blocks = (end - start) / N;
    blocks = ((end - start) % N == 0) ? blocks : blocks + 1;
    invalid = false;
    if (blocks <= 0) invalid = true;
  };
  int start;
  int end;
  int blocks;
  int tpb;
  bool invalid;
};

template <int N, int M>
class RangeGS {
 public:
  RangeGS(int istart, int iend) : start(istart), end(iend), tpb(N), blocks(M){};
  int start;
  int end;
  int tpb;
  int blocks;
};

template <typename Func>
void forall3kernel(const int start0, const int N0, const int start1,
                              const int N1, const int start2, const int N2,
                              Func f) {
  int tid1,tid2;
#pragma omp parallel for private(tid1,tid2) collapse(3)
  for(int tid0=start0;tid0<N0;tid0++){
    for(tid1=start1;tid1<N1;tid1++){
      for(tid2=start2;tid2<N2;tid2++){
	f(tid0, tid1, tid2);
      }
    }
  }
}




template <typename LoopBody>
void forall3(int start0, int end0, int start1, int end1, int start2, int end2,
             LoopBody &&body) {
 
  forall3kernel(start0, end0, start1,
                     end1, start2, end2, body);

}

template <typename T1, typename T2, typename T3, typename LoopBody>
void forall3asyncnotimer(T1 &irange, T2 &jrange, T3 &krange, LoopBody &&body) {

  forall3kernel(
                        irange.start, irange.end, jrange.start, jrange.end,
                        krange.start, krange.end, body);
}

template <typename T1, typename T2, typename T3, typename LoopBody>
void forall3async(T1 &irange, T2 &jrange, T3 &krange, LoopBody &&body) {

  forall3kernel(
                        irange.start, irange.end, jrange.start, jrange.end,
                        krange.start, krange.end, body);
  
  float ms;
  
  std::cout << "Kernel runtime " << ms << " us\n";
}
template <typename T1, typename T2, typename T3, typename LoopBody>
void forall3(T1 &irange, T2 &jrange, T3 &krange, LoopBody &&body) {
  // dim3 tpb(irange.tpb,jrange.tpb,krange.tpb);
  // dim3 blocks(irange.blocks,jrange.blocks,krange.blocks);

  // forall3kernel<<<blocks,tpb>>>(irange.start,irange.end,jrange.start,jrange.end,krange.start,krange.end,body);
  forall3async(irange, jrange, krange, body);

}


// Forall2





template <int N>
class Tclass {
 public:
  Tclass(float best_in = 0.0) {
    value = N;
    best = best_in;
  }
  int value;
  float best;
};

template <int N, typename Tag, typename Func>
void forall3kernel(Tag t, const int start0, const int N0,
                              const int start1, const int N1, const int start2,
                              const int N2, Func f) {

  int tid1,tid2;
#pragma omp parallel for private(tid1,tid2) collapse(3)
  for(int tid0=start0;tid0<N0;tid0++){
    for(tid1=start1;tid1<N1;tid1++){
      for(tid2=start2;tid2<N2;tid2++){
	f(t, tid0, tid1, tid2);
      }
    }
  }
}


#ifdef COMPARE_KERNEL_TIMES
template <int N, typename Tag, typename T1, typename T2, typename T3,
          typename LoopBody>
void forall3async(Tag &t, T1 &irange, T2 &jrange, T3 &krange, LoopBody &&body) {
#ifdef VERBOSE
  std::cout << "forall launch tpb " << irange.tpb << " " << jrange.tpb << "  "
            << krange.tpb << "\n";
  std::cout << "forall launch blocks " << irange.blocks << "  " << jrange.blocks
            << " " << krange.blocks << "="
            << irange.blocks * jrange.blocks * krange.blocks << " "
            << irange.blocks * jrange.blocks * krange.blocks / 120.0 << "\n";
#endif

  forall3kernel<N>(t,
                        irange.start, irange.end, jrange.start, jrange.end,
                        krange.start, krange.end, body);
}
#else
template <int N, typename Tag, typename T1, typename T2, typename T3,
          typename LoopBody>
void forall3async(Tag &t, T1 &irange, T2 &jrange, T3 &krange, LoopBody &&body) {
  forall3kernel<N,Tag,LoopBody>(t,
                        irange.start, irange.end, jrange.start, jrange.end,
                        krange.start, krange.end, body);
   }
#endif

#endif

#endif  // Guards
