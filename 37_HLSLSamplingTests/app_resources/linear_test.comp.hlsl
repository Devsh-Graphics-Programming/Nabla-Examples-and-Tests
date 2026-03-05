#pragma shader_stage(compute)

#include "common/linear.hlsl"
#include <nbl/builtin/hlsl/glsl_compat/core.hlsl>

[[vk::binding(0, 0)]] RWStructuredBuffer<LinearInputValues> inputTestValues;
[[vk::binding(1, 0)]] RWStructuredBuffer<LinearTestResults> outputTestValues;

[numthreads(64, 1, 1)]
[shader("compute")]
void main()
{
	const uint32_t invID = nbl::hlsl::glsl::gl_GlobalInvocationID().x;
#ifdef BENCH_ITERS
	// Use invID as base so u is a runtime value — prevents loop DCE after unrolling.
	// Coefficients are hardcoded to a valid non-degenerate distribution.
	const float32_t2 coeffs = float32_t2(0.2f, 0.8f);
	sampling::Linear<float32_t> sampler = sampling::Linear<float32_t>::create(coeffs);
	const float32_t baseU = frac(float32_t(invID) * 0.6180339887f);
	uint32_t acc = 0u;
	[loop]
	for (uint32_t i = 0u; i < uint32_t(BENCH_ITERS); i++)
	{
		float32_t u = frac(baseU + float32_t(i) * 0.6180339887f);
		acc ^= asuint(sampler.generate(u));
	}
	LinearTestResults result = (LinearTestResults)0;
	result.generated = asfloat(acc);
	outputTestValues[invID] = result;
#else
	LinearTestExecutor executor;
	executor(inputTestValues[invID], outputTestValues[invID]);
#endif
}
