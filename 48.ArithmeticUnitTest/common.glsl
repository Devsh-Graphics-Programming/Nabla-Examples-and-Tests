
#include "irr/builtin/glsl/subgroup/arithmetic_portability.glsl"

layout(set = 0, binding = 0, std430) readonly buffer inputBuff
    {
        uint inputValue[];
    };
layout(set = 0, binding = 1, std430) writeonly buffer outand
    {
        uint andOutput[];
    };
layout(set = 0, binding = 2, std430) writeonly buffer outxor
    {
        uint xorOutput[];
    };
layout(set = 0, binding = 3, std430) writeonly buffer outor
    {
        uint orOutput[];
    };
layout(set = 0, binding = 4, std430) writeonly buffer outadd
    {
        uint addOutput[];
    };
layout(set = 0, binding = 5, std430) writeonly buffer outmul
    {
        uint multOutput[];
    };
layout(set = 0, binding = 6, std430) writeonly buffer outmin
    {
        uint minOutput[];
    };
layout(set = 0, binding = 7, std430) writeonly buffer outmax
{
    uint maxOutput[];
};