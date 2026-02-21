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

using hierarchical_image_type = sampling::HierarchicalImage<float, LuminanceAccessor, WarpAccessor, sampling::warp::Spherical<float> >;
static const LuminanceAccessor luminanceAccessor;
static const WarpAccessor warpAccessor;
static const hierarchical_image_type hImage = hierarchical_image_type::create(luminanceAccessor, warpAccessor, pc.warpResolution, pc.avgLuma);

float32_t2 convertToFloat01(uint32_t2 xi_uint)
{
  return float32_t2(xi_uint) / promote<float32_t2>(float32_t(numeric_limits<uint32_t>::max));
}

[numthreads(WorkgroupSize, 1, 1)]
[shader("compute")]
void main(uint32_t3 threadID : SV_DispatchThreadID)
{
  float32_t eps = pc.eps;

  random::PCG32 pcg = random::PCG32::construct(threadID.x);
  random::DimAdaptorRecursive<random::PCG32, 2> rng = random::DimAdaptorRecursive<random::PCG32, 2>::construct(pcg);
  uint32_t2 xi_uint = rng();
  
  float32_t2 xi = convertToFloat01(xi_uint);

  // uint32_t2 xi_uint = (threadID.x / 1000, threadID.x % 1000);
  //
  // float32_t2 xi = float32_t2(xi_uint) / float32_t2(1000, 1000);


  xi.x = hlsl::clamp(xi.x, eps, 1.f - eps);
  xi.y = hlsl::clamp(xi.y, eps, 1.f - eps);

  float pdf;
  float32_t2 uv;

  const float3 L = hImage.generate_and_pdf(pdf, uv, xi);

  float eps_x = eps;
  float eps_y = eps;

  float32_t2 d_uv;
  float32_t d_pdf;
  const float3 L_plus_du = hImage.generate_and_pdf(d_pdf, d_uv, xi + float32_t2(0.5f * eps_x, 0));
  const float3 L_plus_dv = hImage.generate_and_pdf(d_pdf, d_uv, xi + float32_t2(0, 0.5f * eps_y));

  const float3 L_minus_du = hImage.generate_and_pdf(d_pdf, d_uv, xi - float32_t2(0.5f * eps_x, 0));
  const float3 L_minus_dv = hImage.generate_and_pdf(d_pdf, d_uv, xi - float32_t2(0, 0.5f * eps_y));

  float jacobian = length(cross(L_plus_du - L_minus_du, L_plus_dv - L_minus_dv)) / (eps_x * eps_y);

  test_sample_t testSample;
  testSample.xi = xi;
  testSample.uv = uv;
  testSample.L = L;
  testSample.jacobian = jacobian;
  testSample.pdf = pdf;
  testSample.deferredPdf = hImage.deferredPdf(L);
  vk::RawBufferStore<test_sample_t>(pc.outputAddress + threadID.x * sizeof(test_sample_t), testSample);
}
