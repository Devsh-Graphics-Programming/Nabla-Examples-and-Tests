#ifndef FFT_BLOOM_UTILS
#define FFT_BLOOM_UTILS


using namespace nbl;
using namespace core;
using namespace system;
using namespace asset;
using namespace video;


static inline asset::VkExtent3D padDimensions(asset::VkExtent3D dimension)
{
	for (auto i = 0u; i < 3u; i++)
	{
		auto& coord = (&dimension.width)[i];
		if (coord <= 1u)
			continue;
		coord = core::roundUpToPoT(coord);
	}
	return dimension;
}

static inline size_t getOutputBufferSize(const asset::VkExtent3D& inputDimensions, uint32_t numChannels, bool halfFloats = true)
{
	auto paddedDims = padDimensions(inputDimensions);
	size_t numberOfComplexElements = paddedDims.width * paddedDims.height * paddedDims.depth * numChannels;
	// We would multiply the number below by 2 usually (2 scalars per complex) but since we're doing real FFT packing we're only keeping half of the first axis so they cancel out
	return numberOfComplexElements * (halfFloats ? sizeof(uint16_t) : sizeof(uint32_t)); 
}



#endif