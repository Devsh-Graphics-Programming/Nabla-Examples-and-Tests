//#include "nbl/builtin/hlsl/morton.hlsl"
#include "nbl/builtin/hlsl/cpp_compat.hlsl"

NBL_CONSTEXPR uint32_t bufferSize = 256;

// Proper coverage would require writing tests for ALL possible sign, dimensions and width configurations
//using morton_t2 = nbl::hlsl::morton::code<true, 8, 2>; // Fits in an int16_t
using vector_t2 = nbl::hlsl::vector<int16_t, 3>;

struct PushConstantData
{
	uint64_t deviceBufferAddress;
};