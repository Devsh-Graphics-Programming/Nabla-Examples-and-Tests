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

  float32_t load(uint32_t2 coord, uint32_t level)
  {
    return lumaMap.Load(uint32_t3(coord, level));
  }

};

struct WarpmapAccessor
{
  void gatherUv(float32_t2 xi, NBL_REF_ARG(matrix<float, 4, 2>) uvs, NBL_REF_ARG(float32_t2) interpolant) NBL_CONST_MEMBER_FUNC
  {
    float32_t2 texelCoord = xi * float32_t2(pc.warpWidth - 1, pc.warpHeight - 1);
    interpolant = frac(texelCoord);
    uint32_t2 uv = texelCoord / float32_t2(pc.warpWidth, pc.warpHeight);
    const float32_t4 reds = warpMap.GatherRed(warpSampler, uv);
    const float32_t4 greens = warpMap.GatherGreen(warpSampler, uv);

    uvs = transpose(matrix<float, 2, 4>(
      reds, 
      greens
    ));

  }
};


template <typename WarpSamplerT>
TestOutput GenerateTestOutput(NBL_CONST_REF_ARG(WarpSamplerT) hImage, float32_t2 xi)
{
  using sample_type = typename WarpSamplerT::sample_type;

  const sample_type sample = hImage.generate(xi);
  const float3 L = sample.value();

  float eps_x = pc.eps;
  float eps_y = pc.eps;

  const sample_type sample_plus_du = hImage.generate(xi + float32_t2(0.5f * eps_x, 0));
  const float3 L_plus_du = sample_plus_du.value();
  const sample_type sample_plus_dv = hImage.generate(xi + float32_t2(0, 0.5f * eps_y));
  const float3 L_plus_dv = sample_plus_dv.value();

  const sample_type sample_minus_du = hImage.generate(xi - float32_t2(0.5f * eps_x, 0));
  const float3 L_minus_du = sample_minus_du.value();
  const sample_type sample_minus_dv = hImage.generate(xi - float32_t2(0, 0.5f * eps_y));
  const float3 L_minus_dv = sample_minus_dv.value();

  float jacobian = length(cross(L_plus_du - L_minus_du, L_plus_dv - L_minus_dv)) / (eps_x * eps_y);

  TestOutput testOutput;
  testOutput.L = L;
  testOutput.jacobian = jacobian;
  testOutput.pdf = sample.pdf();
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
  const WarpmapAccessor warpmapAccessor;

  using direct_hierarchical_image_type = sampling::HierarchicalWarpSampler<float, LuminanceAccessor, sampling::warp::Spherical<float> >;

  float32_t eps = pc.eps;

  random::PCG32 pcg = random::PCG32::construct(threadID.x);
  uint32_t2 xi_uint = random::DimAdaptorRecursive<random::PCG32, 2>::__call(pcg);
  
  float32_t2 xi = convertToFloat01(xi_uint);

  xi.x = hlsl::clamp(xi.x, eps, 1.f - eps);
  xi.y = hlsl::clamp(xi.y, eps, 1.f - eps);

  test_sample_t testSample;
  testSample.xi = xi;

  uint32_t2 warpResolution = { pc.warpWidth, pc.warpHeight };
  const direct_hierarchical_image_type directHImage = direct_hierarchical_image_type::create(luminanceAccessor, pc.avgLuma, warpResolution, warpResolution.x != warpResolution.y);

  testSample.directOutput = GenerateTestOutput(directHImage, xi);

  using cached_hierarchical_image_type = sampling::WarpmapSampler<float, LuminanceAccessor, WarpmapAccessor, sampling::warp::Spherical<float> >;
  const cached_hierarchical_image_type cachedHImage = cached_hierarchical_image_type::create(luminanceAccessor, warpmapAccessor, warpResolution, pc.avgLuma);
  testSample.cachedOutput = GenerateTestOutput(cachedHImage, xi);
  vk::RawBufferStore<test_sample_t>(pc.outputAddress + threadID.x * sizeof(test_sample_t), testSample);
}
