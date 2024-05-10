#pragma wave shader_stage(compute)
#include "common.hlsl"
#include "../emulated_float64.hlsl"

[numthreads(256,1,1)]
void main() 
{
    const emulated::float64_t a = _static_cast<emulated::float64_t>(6.9f);
    const emulated::float64_t b = _static_cast<emulated::float64_t>(4.5f);

    vk::RawBufferStore<emulated::float64_t::storage_t>(0,(a*b).data);

    const float asdf = 1.0f;
    const float asdff = 1.0f;
}