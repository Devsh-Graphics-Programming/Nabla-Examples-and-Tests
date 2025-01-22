#include "common.hlsl"

[shader("miss")]
void main(inout ColorPayload p)
{
    p.hitValue = float32_t3(0.3, 0.3, 0.6);

}
