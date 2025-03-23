#include "nbl/builtin/hlsl/math/morton.hlsl"

[numthreads(512, 1, 1)]
void main(uint32_t3 ID : SV_DispatchThreadID)
{
	printf("%d %d", nbl::hlsl::morton::impl::decode_masks_array<uint32_t, 2>::Masks[0], nbl::hlsl::morton::impl::decode_masks_array<uint32_t, 2>::Masks[1]);
}