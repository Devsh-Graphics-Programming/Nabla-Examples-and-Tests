// Copyright (C) 2018-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "app_resources/common.hlsl"

using namespace nbl::hlsl;

[[vk::push_constant]] nbl::hlsl::this_example::SPushConstants pc;

struct PSInput
{
    float32_t4 position : SV_Position;   
    float32_t4 color    : COLOR0;        
};

struct VSInput
{
    [[vk::location(0)]] float32_t2 position : POSITION;
    [[vk::location(1)]] float32_t3 color    : COLOR0;
    uint instanceID : SV_InstanceID;
};
    
[shader("vertex")]
PSInput VSMain(VSInput input)
{
    static const float32_t4 InstanceColors[3] = { 
           	{1.f, 0.f, 0.f,0.5f} , 
           	{0.f, 1.f, 0.f, 0.5f} , 
           	{0.f, 0.f, 1.f, 0.5f}  
    };
    PSInput output;
    output.position = mul(pc.viewProj, float32_t4(input.position, input.instanceID * 5.f, 1.f));
    output.color = InstanceColors[input.instanceID];
    return output;
}
