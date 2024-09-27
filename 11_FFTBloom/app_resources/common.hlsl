#include "nbl/builtin/hlsl/cpp_compat.hlsl"

struct PushConstantData
{
	uint64_t inputAddress;
	uint64_t outputAddress;
	uint32_t dataElementCount;
};