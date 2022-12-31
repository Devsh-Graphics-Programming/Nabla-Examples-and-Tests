// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#version 430 core

layout (location = 0) in vec3 Pos;
layout (location = 3) in vec3 Normal;
layout (location = 4) in vec3 GNormal;

layout (location = 0) out vec3 outNormal;
layout (location = 1) out vec3 outPos;
layout (location = 2) flat out vec3 outLightPos;
layout (location = 3) flat out vec3 outCamPos;
layout (location = 4) flat out vec3 outGNormal;
layout (location = 5) flat out float outAlpha;
layout (location = 6) flat out float outAlphaY;

layout (push_constant, row_major) uniform PC {
    mat4 VP;
} pc;

vec3 to_right_hand(in vec3 v)
{
    return v*vec3(-1.0,1.0,1.0);
}

void main()
{
    uint gridX = gl_InstanceIndex / 5;  
    uint gridY = gl_InstanceIndex % 5;
    
    vec2 gridPos = vec2(float(gridX)*0.4-0.8, float(gridY)*0.4-0.8);
    vec3 pos = to_right_hand(Pos);
    outPos = pos;
    vec4 projPos = pc.VP*vec4(pos, 1.0);
    gl_Position = pc.VP*vec4(pos, 1.0) + vec4(gridPos*projPos.w,0.0,0.0);
    outNormal = to_right_hand(normalize(Normal));
    outLightPos = vec3(0.75,0.75,1.75);
    outCamPos = vec3(0.0,0.0,6.5);
    outGNormal = to_right_hand(normalize(Normal));
    outAlpha = gridX * 0.2 + 0.1;
    outAlphaY = gridY * 0.2;
}