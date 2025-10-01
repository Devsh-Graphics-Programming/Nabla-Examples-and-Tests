#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include <nbl/builtin/hlsl/colorspace/encodeCIEXYZ.hlsl>

struct SPushConstants
{
	uint32_t cascadeCount;
	float base;
	uint32_t sampleCount;
	float minReliableLuma;
	float kappa;
};

[[vk::push_constant]] SPushConstants pc;
[[vk::image_format("rgba16f")]] [[vk::binding(0, 0)]] RWTexture2D<float32_t4> outImage;
[[vk::image_format("rgba16f")]] [[vk::binding(1, 0)]] RWTexture2DArray<float32_t4> cascade;

using namespace nbl;
using namespace hlsl;

NBL_CONSTEXPR uint32_t WorkgroupSize = 512;
NBL_CONSTEXPR uint32_t MAX_DEPTH_LOG2 = 4;
NBL_CONSTEXPR uint32_t MAX_SAMPLES_LOG2 = 10;

struct RWMCReweightingParameters
{
    uint32_t lastCascadeIndex;
    float initialEmin; // a minimum image brightness that we always consider reliable
    float reciprocalBase;
    float reciprocalN;
    float reciprocalKappa;
    float colorReliabilityFactor;
    float NOverKappa;
};

RWMCReweightingParameters computeReweightingParameters(uint32_t cascadeCount, float base, uint32_t sampleCount, float minReliableLuma, float kappa)
{
    RWMCReweightingParameters retval;
    retval.lastCascadeIndex = cascadeCount - 1u;
    retval.initialEmin = minReliableLuma;
    retval.reciprocalBase = 1.f / base;
    const float N = float(sampleCount);
    retval.reciprocalN = 1.f / N;
    retval.reciprocalKappa = 1.f / kappa;
    // if not interested in exact expected value estimation (kappa!=1.f), can usually accept a bit more variance relative to the image brightness we already have
    // allow up to ~<cascadeBase> more energy in one sample to lessen bias in some cases
    retval.colorReliabilityFactor = base + (1.f - base) * retval.reciprocalKappa;
    retval.NOverKappa = N * retval.reciprocalKappa;

    return retval;
}

struct RWMCCascadeSample
{
	float32_t3 centerValue;
	float normalizedCenterLuma;
	float normalizedNeighbourhoodAverageLuma;
};

// TODO: figure out what values should pixels outside have, 0.0f is incorrect
float32_t3 RWMCsampleCascadeTexel(int32_t2 currentCoord, int32_t2 offset, uint32_t cascadeIndex)
{
	const int32_t2 texelCoord = currentCoord + offset;
	if (any(texelCoord < int32_t2(0, 0)))
		return float32_t3(0.0f, 0.0f, 0.0f);

    float32_t4 output = cascade.Load(int32_t3(texelCoord, int32_t(cascadeIndex)));
    return float32_t3(output.r, output.g, output.b);
}

float32_t calcLuma(in float32_t3 col)
{
    return hlsl::dot<float32_t3>(hlsl::transpose(colorspace::scRGBtoXYZ)[1], col);
}

RWMCCascadeSample RWMCSampleCascade(in int32_t2 coord, in uint cascadeIndex, in float reciprocalBaseI)
{
	float32_t3 neighbourhood[9];
	neighbourhood[0] = RWMCsampleCascadeTexel(coord, int32_t2(-1, -1), cascadeIndex);
    neighbourhood[1] = RWMCsampleCascadeTexel(coord, int32_t2(0, -1), cascadeIndex);
    neighbourhood[2] = RWMCsampleCascadeTexel(coord, int32_t2(1, -1), cascadeIndex);
    neighbourhood[3] = RWMCsampleCascadeTexel(coord, int32_t2(-1, 0), cascadeIndex);
    neighbourhood[4] = RWMCsampleCascadeTexel(coord, int32_t2(0, 0), cascadeIndex);
    neighbourhood[5] = RWMCsampleCascadeTexel(coord, int32_t2(1, 0), cascadeIndex);
    neighbourhood[6] = RWMCsampleCascadeTexel(coord, int32_t2(-1, 1), cascadeIndex);
    neighbourhood[7] = RWMCsampleCascadeTexel(coord, int32_t2(0, 1), cascadeIndex);
    neighbourhood[8] = RWMCsampleCascadeTexel(coord, int32_t2(1, 1), cascadeIndex);

	// numerical robustness
	float32_t3 excl_hood_sum = ((neighbourhood[0] + neighbourhood[1]) + (neighbourhood[2] + neighbourhood[3])) +
		((neighbourhood[5] + neighbourhood[6]) + (neighbourhood[7] + neighbourhood[8]));

	RWMCCascadeSample retval;
	retval.centerValue = neighbourhood[4];
	retval.normalizedNeighbourhoodAverageLuma = retval.normalizedCenterLuma = calcLuma(neighbourhood[4]) * reciprocalBaseI;
	retval.normalizedNeighbourhoodAverageLuma = (calcLuma(excl_hood_sum) * reciprocalBaseI + retval.normalizedNeighbourhoodAverageLuma) / 9.f;
	return retval;
}

float32_t3 RWMCReweight(in RWMCReweightingParameters params, in int32_t2 coord)
{
	float reciprocalBaseI = 1.f;
	RWMCCascadeSample curr = RWMCSampleCascade(coord, 0u, reciprocalBaseI);

	float32_t3 accumulation = float32_t3(0.0f, 0.0f, 0.0f);
	float Emin = params.initialEmin;

	float prevNormalizedCenterLuma, prevNormalizedNeighbourhoodAverageLuma;
	for (uint i = 0u; i <= params.lastCascadeIndex; i++)
	{
		const bool notFirstCascade = i != 0u;
		const bool notLastCascade = i != params.lastCascadeIndex;

		RWMCCascadeSample next;
		if (notLastCascade)
		{
			reciprocalBaseI *= params.reciprocalBase;
			next = RWMCSampleCascade(coord, i + 1u, reciprocalBaseI);
		}


		float reliability = 1.f;
		// sample counting-based reliability estimation
		if (params.reciprocalKappa <= 1.f)
		{
			float localReliability = curr.normalizedCenterLuma;
			// reliability in 3x3 pixel block (see robustness)
			float globalReliability = curr.normalizedNeighbourhoodAverageLuma;
			if (notFirstCascade)
			{
				localReliability += prevNormalizedCenterLuma;
				globalReliability += prevNormalizedNeighbourhoodAverageLuma;
			}
			if (notLastCascade)
			{
				localReliability += next.normalizedCenterLuma;
				globalReliability += next.normalizedNeighbourhoodAverageLuma;
			}
			// check if above minimum sampling threshold (avg 9 sample occurences in 3x3 neighbourhood), then use per-pixel reliability (NOTE: tertiary op is in reverse)
			reliability = globalReliability < params.reciprocalN ? globalReliability : localReliability;
			{
				const float accumLuma = calcLuma(accumulation);
				if (accumLuma > Emin)
					Emin = accumLuma;

				const float colorReliability = Emin * reciprocalBaseI * params.colorReliabilityFactor;

				reliability += colorReliability;
				reliability *= params.NOverKappa;
				reliability -= params.reciprocalKappa;
				reliability = clamp(reliability * 0.5f, 0.f, 1.f);
			}
		}
		accumulation += curr.centerValue * reliability;

		prevNormalizedCenterLuma = curr.normalizedCenterLuma;
		prevNormalizedNeighbourhoodAverageLuma = curr.normalizedNeighbourhoodAverageLuma;
		curr = next;
	}

	return accumulation;
}

int32_t2 getCoordinates()
{
    uint32_t width, height;
    outImage.GetDimensions(width, height);
    return int32_t2(glsl::gl_GlobalInvocationID().x % width, glsl::gl_GlobalInvocationID().x / width);
}

// this function is for testing purpose
// simply adds every cascade buffer, output shoud be nearly the same as output of default accumulator (RWMC off)
float32_t3 sumCascade(in const int32_t2 coords)
{
    float32_t3 accumulation = float32_t3(0.0f, 0.0f, 0.0f);

    for (int i = 0; i < 6; ++i)
    {
        float32_t4 cascadeLevel = cascade.Load(uint3(coords, i));
        accumulation += float32_t3(cascadeLevel.r, cascadeLevel.g, cascadeLevel.b);
    }

    accumulation /= 32.0f;

	return accumulation;
}

[numthreads(WorkgroupSize, 1, 1)]
void main(uint32_t3 threadID : SV_DispatchThreadID)
{
    // TODO: remove, ideally shader should not be called at all when we don't use RWMC
    bool useRWMC = true;
    if (!useRWMC)
        return;

    const int32_t2 coords = getCoordinates();
    //float32_t3 color = sumCascade(coords);

    RWMCReweightingParameters reweightingParameters = computeReweightingParameters(pc.cascadeCount, pc.base, pc.sampleCount, pc.minReliableLuma, pc.kappa);
	float32_t3 color = RWMCReweight(reweightingParameters, coords);
	color /= pc.sampleCount;

	outImage[coords] = float32_t4(color, 1.0f);
}
