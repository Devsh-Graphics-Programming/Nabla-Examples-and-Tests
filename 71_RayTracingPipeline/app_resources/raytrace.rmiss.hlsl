#include "common.hlsl"

[shader("miss")]
void main(inout HitPayload payload)
{
    payload.rayDistance = -1;
}
