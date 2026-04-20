#include "nbl/builtin/hlsl/rwmc/resolve.hlsl"

NBL_CONSTEXPR_INLINE_NSPC_SCOPE_VAR uint32_t SessionDSIndex = 0;
#include "renderer/shaders/session.hlsl"


using namespace nbl;
using namespace nbl::hlsl;
using namespace nbl::this_example;

[[vk::push_constant]] SResolveConstants pc;

struct SCascadeAccessor
{
	using output_scalar_t = float32_t;
	NBL_CONSTEXPR_STATIC_INLINE int32_t Components = 3;
	using output_t = vector<output_scalar_t,Components>;
	NBL_CONSTEXPR_STATIC_INLINE int32_t image_dimension = 2;

	static SCascadeAccessor create(const uint32_t outputLayer, const uint32_t cascadeCount)
	{
		SCascadeAccessor retval;
		uint32_t width, height, layers;
		gRWMCCascades.GetDimensions(width,height,layers);
		retval.cascadeImageDimension = uint32_t2(width,height);
		retval.outputLayer = outputLayer;
		retval.cascadeCount = cascadeCount;
		retval.totalLayers = layers;
		return retval;
	}

	template<typename OutputScalarType, int32_t Dimension>
	void get(NBL_REF_ARG(output_t) value, vector<uint16_t,2> uv, uint16_t layer, uint16_t level)
	{
		const uint32_t cascadeLayer = outputLayer*cascadeCount+uint32_t(layer);
		if (any(uint32_t2(uv)>=cascadeImageDimension) || uint32_t(layer)>=cascadeCount || cascadeLayer>=totalLayers)
		{
			value = promote<output_t,output_scalar_t>(0.f);
			return;
		}

		const uint32_t2 packed = gRWMCCascades.Load(int32_t3(_static_cast<int32_t2>(uv),int32_t(cascadeLayer)));
		const uint16_t4 data = bit_cast<uint16_t4>(packed);
		value = float32_t3(bit_cast<float16_t3>(data.xyz));
	}

	uint32_t2 cascadeImageDimension;
	uint32_t outputLayer;
	uint32_t cascadeCount;
	uint32_t totalLayers;
};

uint32_t3 getBeautyExtents()
{
	uint32_t width, height, imageArraySize;
	gBeauty.GetDimensions(width,height,imageArraySize);
	return uint32_t3(width,height,imageArraySize);
}

template<uint16_t CascadeCount>
float32_t3 resolveCascades(NBL_REF_ARG(SCascadeAccessor) cascadeAccessor, const uint16_t2 coords)
{
	using adaptor_t = rwmc::SResolveAccessorAdaptor<SCascadeAccessor,float32_t>;
	using resolver_t = rwmc::SResolver<adaptor_t,CascadeCount>;
	adaptor_t accessor = {cascadeAccessor};
	resolver_t resolver = resolver_t::create(pc.resolveParameters);
	return resolver(accessor,int16_t2(coords));
}

float32_t3 resolveCascades(NBL_REF_ARG(SCascadeAccessor) cascadeAccessor, const uint16_t2 coords)
{
	switch (pc.cascadeCount)
	{
		case 1u:
			return resolveCascades<1u>(cascadeAccessor,coords);
		case 2u:
			return resolveCascades<2u>(cascadeAccessor,coords);
		case 3u:
			return resolveCascades<3u>(cascadeAccessor,coords);
		case 4u:
			return resolveCascades<4u>(cascadeAccessor,coords);
		case 5u:
			return resolveCascades<5u>(cascadeAccessor,coords);
		case 6u:
			return resolveCascades<6u>(cascadeAccessor,coords);
		case 7u:
			return resolveCascades<7u>(cascadeAccessor,coords);
		default:
			return resolveCascades<8u>(cascadeAccessor,coords);
	}
}

[numthreads(ResolveWorkgroupSizeX,ResolveWorkgroupSizeY,1)]
[shader("compute")]
void resolve(uint32_t3 threadID : SV_DispatchThreadID)
{
	const uint32_t3 imageExtents = getBeautyExtents();
	if (any(threadID>=imageExtents) || pc.cascadeCount==0u || pc.cascadeCount>MaxCascadeCount)
		return;

	SCascadeAccessor accessor = SCascadeAccessor::create(threadID.z,pc.cascadeCount);
	const float32_t3 color = max(resolveCascades(accessor,uint16_t2(threadID.xy)),float32_t3(0.f,0.f,0.f));

	gBeauty[threadID] = float32_t4(color,1.f);
}
