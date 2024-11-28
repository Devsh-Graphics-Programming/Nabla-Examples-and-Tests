#include "nbl/builtin/hlsl/cpp_compat.hlsl"

NBL_CONSTEXPR_STATIC_INLINE uint32_t Channels = 3;

using namespace nbl::hlsl;

struct PushConstantData
{
	// After running FFT along a column, we want to store the result in column major order for coalesced writes, and similarly after running an FFT in row major order
	// All FFTs that read from a buffer and write to a buffer (kernel second FFT, image second FFT + conv + IFFT) read in one order and write in another. 
	// Since workgroups start and finish at arbitrary times it's not possible to pickup the data from a buffer in one layout and write it back in another layout due to
	// possibility of writing over data that still hasn't been read by some workgroups.
	uint64_t colMajorBufferAddress;
	uint64_t rowMajorBufferAddress;
	// To save some work, we don't mirror the image along both directions when doing the FFT. This means that when doing the FFT along the second axis, we do an FFT of length
	// `RoundUpToPoT(imageRowLength + kernelPadding)` where `imageRowLength` is the actual length of the image along the second axis. We need it to keep track of the image's original dimension.
	// The following three fields being push constants allow dynamic resizing of the image without recompiling shaders (limited by the FFT length)
	int32_t imageRowLength; 
	int32_t imageHalfRowLength;
	// Used by IFFT to tell if an index belongs to an image or is in the padding
	uint32_t imageColumnLength;
	int32_t padding;
	float32_t2 imageHalfPixelSize;
	float32_t2 imagePixelSize;
	float32_t imageTwoPixelSize_x;
	float32_t interpolatingFactor;
};

#ifdef __HLSL_VERSION

#include "nbl/builtin/hlsl/bda/legacy_bda_accessor.hlsl"
#include "nbl/builtin/hlsl/workgroup/fft.hlsl"

[[vk::push_constant]] PushConstantData pushConstants;

using FFTParameters = ConstevalParameters::FFTParameters;
using scalar_t = FFTParameters::scalar_t;

using FFTIndexingUtils = workgroup::fft::FFTIndexingUtils<FFTParameters::ElementsPerInvocationLog2, FFTParameters::WorkgroupSizeLog2>;

// Users MUST define this method for FFT to work - workgroup::SubgroupContiguousIndex also needs it defined
uint32_t3 glsl::gl_WorkGroupSize() { return uint32_t3(uint32_t(FFTParameters::WorkgroupSize), 1, 1); }

#endif
