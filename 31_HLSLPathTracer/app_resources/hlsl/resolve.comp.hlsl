#include <nbl/builtin/hlsl/rwmc/resolve.hlsl>
#include "resolve_common.hlsl"
#include "rwmc_global_settings_common.hlsl"
#ifdef PERSISTENT_WORKGROUPS
#include "nbl/builtin/hlsl/math/morton.hlsl"
#endif
#include "rwmc_global_settings_common.hlsl"

[[vk::image_format("rgba16f")]] [[vk::binding(0)]] RWTexture2DArray<float32_t4> outImage;
[[vk::image_format("rgba16f")]] [[vk::binding(1)]] RWTexture2DArray<float32_t4> cascade;
[[vk::push_constant]] ResolvePushConstants pc;

using namespace nbl;
using namespace hlsl;

struct SCascadeAccessor
{
    using output_scalar_t = float32_t;
    NBL_CONSTEXPR_STATIC_INLINE int32_t Components = 4;
    using output_t = vector<output_scalar_t, Components>;
    NBL_CONSTEXPR_STATIC_INLINE int32_t image_dimension = 2;

    static SCascadeAccessor create()
    {
        SCascadeAccessor retval;
        uint32_t imgWidth, imgHeight, layers;
        cascade.GetDimensions(imgWidth, imgHeight, layers);
        retval.cascadeImageDimension = int16_t2(imgWidth, imgHeight);
        return retval;
    }

    template<typename OutputScalarType, int32_t Dimension>
    void get(NBL_REF_ARG(output_t) value, vector<uint16_t, 2> uv, uint16_t layer, uint16_t level)
    {
        if (any(uv < int16_t2(0, 0)) || any(uv >= cascadeImageDimension))
        {
            value = promote<output_t, output_scalar_t>(0);
            return;
        }

        value = cascade.Load(int32_t3(uv, int32_t(layer)));
    }

    int16_t2 cascadeImageDimension;
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

    using SResolveAccessorAdaptorType = rwmc::SResolveAccessorAdaptor<SCascadeAccessor, float32_t>;
    using SResolverType = rwmc::SResolver<SResolveAccessorAdaptorType, CascadeCount>;
    SResolveAccessorAdaptorType accessor = { SCascadeAccessor::create() };
    SResolverType resolve = SResolverType::create(pc.resolveParameters);

    float32_t3 color = resolve(accessor, int16_t2(coords.x, coords.y));

    outImage[uint3(coords.x, coords.y, 0)] = float32_t4(color, 1.0f);
}
