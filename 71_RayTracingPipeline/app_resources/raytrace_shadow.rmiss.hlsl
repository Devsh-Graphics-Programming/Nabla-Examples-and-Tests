#include "common.hlsl"

[shader("miss")]
void main(inout OcclusionPayload payload)
{
    // make positive
    payload.attenuation = -payload.attenuation;
}
