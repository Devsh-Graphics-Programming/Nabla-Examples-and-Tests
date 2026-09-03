#ifndef _ENVMAP_IMPORTANCE_SAMPLING_SEARCH_H_INCLUDED_
#define _ENVMAP_IMPORTANCE_SAMPLING_SEARCH_H_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat/basic.h>
#include <nbl/builtin/hlsl/cpp_compat/vector.hlsl>

using namespace nbl;
using namespace nbl::hlsl;

NBL_CONSTEXPR uint32_t WorkgroupSize = 128;

struct STestPushConstants
{
  float32_t eps;
  uint64_t outputAddress;
  uint32_t2 warpResolution;
  float32_t avgLuma;
};

struct TestOutput
{
  float32_t3 L;
  float32_t2 uv;
  float32_t jacobian;
  float32_t pdf;
  float32_t deferredPdf;
};

struct TestSample
{
  TestOutput directOutput;
  TestOutput cachedOutput;
  float32_t2 xi;
};

using test_sample_t = TestSample;

#endif // _COOPERATIVE_BINARY_SEARCH_H_INCLUDED_
