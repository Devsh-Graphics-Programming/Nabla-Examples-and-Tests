#include <nbl/builtin/hlsl/rwmc/resolve.hlsl>
#include "resolve_common.hlsl"
#include "rwmc_global_settings_common.hlsl"
#ifdef PERSISTENT_WORKGROUPS
#include "nbl/builtin/hlsl/math/morton.hlsl"
#endif

[[vk::image_format("rgba16f")]] [[vk::binding(0)]] RWTexture2DArray<float32_t4> outImage;
[[vk::image_format("rgba16f")]] [[vk::binding(1)]] RWTexture2DArray<float32_t4> cascade;
[[vk::push_constant]] ResolvePushConstants pc;

using namespace nbl;
using namespace hlsl;

template<typename OutputScalar>
struct ResolveAccessorAdaptor
{
	using output_scalar_type = OutputScalar;
	using output_type = vector<OutputScalar, 4>;
	NBL_CONSTEXPR int32_t image_dimension = 2;

	float32_t calcLuma(NBL_REF_ARG(float32_t3) col)
	{
		return hlsl::dot<float32_t3>(colorspace::scRGB::ToXYZ()[1], col);
	}

	template<typename OutputScalarType, int32_t Dimension>
	output_type get(vector<uint16_t, 2> uv, uint16_t layer)
	{
		uint32_t imgWidth, imgHeight, layers;
		cascade.GetDimensions(imgWidth, imgHeight, layers);
		int16_t2 cascadeImageDimension = int16_t2(imgWidth, imgHeight);

		if (any(uv < int16_t2(0, 0)) || any(uv > cascadeImageDimension))
			return vector<OutputScalar, 4>(0, 0, 0, 0);

		return cascade.Load(int32_t3(uv, int32_t(layer)));
	}
};

int32_t2 getImageExtents()
{
    uint32_t width, height, imageArraySize;
    outImage.GetDimensions(width, height, imageArraySize);
    return int32_t2(width, height);
}

[numthreads(ResolveWorkgroupSizeX, ResolveWorkgroupSizeY, 1)]
void main(uint32_t3 threadID : SV_DispatchThreadID)
{
    const int32_t2 coords = int32_t2(threadID.x, threadID.y);
    const int32_t2 imageExtents = getImageExtents();
    if (coords.x >= imageExtents.x || coords.y >= imageExtents.y)
        return;

    using ResolveAccessorAdaptorType = ResolveAccessorAdaptor<float>;
    using ResolverType = rwmc::Resolver<ResolveAccessorAdaptorType, float32_t3>;
    ResolveAccessorAdaptorType accessor;
    ResolverType resolve = ResolverType::create(pc.resolveParameters);

    float32_t3 color = resolve(accessor, int16_t2(coords.x, coords.y));

    //float32_t3 color = rwmc::reweight<ResolveAccessorAdaptor<float> >(pc.resolveParameters, cascade, coords);

    outImage[uint3(coords.x, coords.y, 0)] = float32_t4(color, 1.0f);
}
