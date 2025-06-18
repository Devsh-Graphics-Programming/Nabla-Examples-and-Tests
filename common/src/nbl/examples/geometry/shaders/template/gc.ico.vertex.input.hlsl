#ifndef _NBL_EXAMPLES_GEOMETRY_ICO_VERTEX_INPUT_HLSL_
#define _NBL_EXAMPLES_GEOMETRY_ICO_VERTEX_INPUT_HLSL_

[[vk::binding(0)]] Buffer<float32_t3> position;
[[vk::binding(1)]] Buffer<float32_t3> normal;
[[vk::binding(2)]] Buffer<float32_t2> uv;

#endif // _NBL_EXAMPLES_GEOMETRY_ICO_VERTEX_INPUT_HLSL_
/*
    do not remove this text, WAVE is so bad that you can get errors if no proper ending xD
*/
