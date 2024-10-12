#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/complex.hlsl"
#include "nbl/builtin/hlsl/workgroup/fft.hlsl"

using namespace nbl::hlsl;

#define CHANNELS 3

struct PushConstantData
{
	uint64_t colMajorBufferAddress;
	uint64_t rowMajorBufferAddress;
	uint32_t dataElementCount;
	float32_t2 kernelHalfPixelSize;
};
