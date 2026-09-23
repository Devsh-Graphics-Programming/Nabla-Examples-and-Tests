#include "common.hlsl"

#include <nbl/builtin/hlsl/sampling/hierarchical_image.hlsl>
#include <nbl/builtin/hlsl/random/pcg.hlsl>
#include <nbl/builtin/hlsl/random/dim_adaptor_recursive.hlsl>
#include <nbl/builtin/hlsl/tgmath.hlsl>
#include <nbl/builtin/hlsl/sampling/warps/spherical.hlsl>
#include <nbl/builtin/hlsl/cpp_compat.hlsl>

[[vk::push_constant]] STestPushConstants pc;

[[vk::combinedImageSampler]][[vk::binding(0, 0)]] Texture2D<float32_t> lumaMap;
[[vk::combinedImageSampler]][[vk::binding(0, 0)]] SamplerState lumaSampler;

[[vk::combinedImageSampler]][[vk::binding(1, 0)]] Texture2D<float32_t2> warpMap;
[[vk::combinedImageSampler]][[vk::binding(1, 0)]] SamplerState warpSampler;

using namespace nbl::hlsl::sampling::hierarchical_image;


struct LuminanceAccessor
{
  template <typename ValT, typename IndexT 
    NBL_FUNC_REQUIRES(
      concepts::same_as<IndexT, float32_t2> && 
      concepts::same_as<ValT, float32_t>
    )
  void get(IndexT index, NBL_REF_ARG(ValT) val)
  {
    val = lumaMap.SampleLevel(lumaSampler, index, 0);
  }

  float32_t texelFetch(uint32_t2 coord, uint32_t level)
  {
    return lumaMap.Load(uint32_t3(coord, level));
  }

  float32_t4 texelGather(uint32_t2 coord, uint32_t level)
  {
    return float32_t4(
      lumaMap.Load(uint32_t3(coord, level), uint32_t2(0, 1)),
      lumaMap.Load(uint32_t3(coord, level), uint32_t2(1, 1)),
      lumaMap.Load(uint32_t3(coord, level), uint32_t2(1, 0)),
      lumaMap.Load(uint32_t3(coord, level), uint32_t2(0, 0))
    );
  }
};

struct WarpAccessor
{
     matrix<float, 4, 2> sampleUvs(uint32_t2 sampleCoord) NBL_CONST_MEMBER_FUNC
     {
        const float32_t2 dir0 = warpMap.Load(int32_t3(sampleCoord + uint32_t2(0, 1), 0));
        const float32_t2 dir1 = warpMap.Load(int32_t3(sampleCoord + uint32_t2(1, 1), 0));
        const float32_t2 dir2 = warpMap.Load(int32_t3(sampleCoord + uint32_t2(1, 0), 0));
        const float32_t2 dir3 = warpMap.Load(int32_t3(sampleCoord, 0));
        return matrix<float, 4, 2>(
          dir0,
          dir1,
          dir2,
          dir3
        );
     }
};


template <typename HierarchicalImageT>
TestOutput GenerateTestOutput(NBL_CONST_REF_ARG(HierarchicalImageT) hImage, float32_t2 xi)
{
  float pdf;
  float32_t2 uv;

  const float3 L = hImage.generate_and_pdf(pdf, uv, xi);

  float eps_x = pc.eps;
  float eps_y = pc.eps;

  float32_t2 d_uv;
  float32_t d_pdf;
  const float3 L_plus_du = hImage.generate_and_pdf(d_pdf, d_uv, xi + float32_t2(0.5f * eps_x, 0));
  const float3 L_plus_dv = hImage.generate_and_pdf(d_pdf, d_uv, xi + float32_t2(0, 0.5f * eps_y));

  const float3 L_minus_du = hImage.generate_and_pdf(d_pdf, d_uv, xi - float32_t2(0.5f * eps_x, 0));
  const float3 L_minus_dv = hImage.generate_and_pdf(d_pdf, d_uv, xi - float32_t2(0, 0.5f * eps_y));

  float jacobian = length(cross(L_plus_du - L_minus_du, L_plus_dv - L_minus_dv)) / (eps_x * eps_y);

  TestOutput testOutput;
  testOutput.uv = uv;
  testOutput.L = L;
  testOutput.jacobian = jacobian;
  testOutput.pdf = pdf;
  testOutput.deferredPdf = hImage.deferredPdf(L);
  return testOutput;
}

float32_t2 convertToFloat01(uint32_t2 xi_uint)
{
  return float32_t2(xi_uint) / promote<float32_t2>(float32_t(numeric_limits<uint32_t>::max));
}

[numthreads(WorkgroupSize, 1, 1)]
[shader("compute")]
void main(uint32_t3 threadID : SV_DispatchThreadID)
{
  const LuminanceAccessor luminanceAccessor;
  const WarpAccessor warpAccessor;
  using luminance_sampler_type = nbl::hlsl::sampling::LuminanceMapSampler<float32_t, LuminanceAccessor>;

  using direct_hierarchical_image_type = sampling::HierarchicalImage<float, LuminanceAccessor, luminance_sampler_type, sampling::warp::Spherical<float> >;

  const luminance_sampler_type luminanceSampler = luminance_sampler_type::create(luminanceAccessor, pc.warpResolution, true, pc.warpResolution);

  float32_t eps = pc.eps;

  random::PCG32 pcg = random::PCG32::construct(threadID.x);
  uint32_t2 xi_uint = random::DimAdaptorRecursive<random::PCG32, 2>::__call(pcg);
  
  float32_t2 xi = convertToFloat01(xi_uint);

  xi.x = hlsl::clamp(xi.x, eps, 1.f - eps);
  xi.y = hlsl::clamp(xi.y, eps, 1.f - eps);

  test_sample_t testSample;
  testSample.xi = xi;

  const direct_hierarchical_image_type directHImage = direct_hierarchical_image_type::create(luminanceAccessor, luminanceSampler, pc.warpResolution, pc.avgLuma);
  testSample.directOutput = GenerateTestOutput(directHImage, xi);

  using cached_hierarchical_image_type = sampling::HierarchicalImage<float, LuminanceAccessor, WarpAccessor, sampling::warp::Spherical<float> >;
  const cached_hierarchical_image_type cachedHImage = cached_hierarchical_image_type::create(luminanceAccessor, warpAccessor, pc.warpResolution, pc.avgLuma);
  testSample.cachedOutput = GenerateTestOutput(cachedHImage, xi);
  vk::RawBufferStore<test_sample_t>(pc.outputAddress + threadID.x * sizeof(test_sample_t), testSample);
}
