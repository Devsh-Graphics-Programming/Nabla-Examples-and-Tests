#include "app_resources/common.hlsl"
#include "nbl/builtin/hlsl/bda/legacy_bda_accessor.hlsl"

[[vk::push_constant]] PushConstantData pushConstants;

using namespace nbl::hlsl;

[numthreads(bufferSize, 1, 1)]
void main(uint32_t3 ID : SV_DispatchThreadID)
{
	LegacyBdaAccessor<unsigned_scalar_t> accessor = LegacyBdaAccessor<unsigned_scalar_t>::create(pushConstants.deviceBufferAddress);
	
	morton::code<int32_t, 2> foo = morton::code<int32_t, 2>::create(vector<int32_t, 2>(-32768, -1));

	accessor.set(0, foo.value);
}