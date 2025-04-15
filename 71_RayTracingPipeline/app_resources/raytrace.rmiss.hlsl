#include "common.hlsl"

[shader("miss")]
void main(inout PrimaryPayload payload)
{
    payload.rayDistance = -1;
}
