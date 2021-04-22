//////////////////////////////////////////////////////////////////////////////
//// Copyright (c) 2021, Lawrence Livermore National Security, LLC and SW4CK
//// project contributors. See the COPYRIGHT file for details.
////
//// SPDX-License-Identifier: GPL-2.0-only
////////////////////////////////////////////////////////////////////////////////
#ifndef __SW4_FORALL_H__
#define __SW4_FORALL_H__
void promo_version();
#if defined(ENABLE_CUDA)
#include <cuda.h>
#include "cuda_foralls.h"
#endif

#if defined(ENABLE_HIP)
#include "hip_foralls.h"
#include <hip/hip_version.h>

#endif

#endif
