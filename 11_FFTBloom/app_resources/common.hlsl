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

// Util to unpack two values from the packed FFT X + iY - get outputs in the same input arguments, storing x to lo and y to hi
template<typename Scalar>
void unpack(NBL_CONST_REF_ARG(complex_t<Scalar>) lo, NBL_CONST_REF_ARG(complex_t<Scalar>) hi)
{
	complex_t<Scalar> x = (lo + conj(hi)) * Scalar(0.5);
	hi = rotateRight<Scalar>(lo - conj(hi)) * 0.5;
	lo = x;
}
