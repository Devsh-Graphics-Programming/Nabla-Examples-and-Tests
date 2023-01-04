// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#version 430 core

layout (location = 0) in vec3 Pos;
layout (location = 1) in vec3 Normal;

layout (location = 0) out vec3 outNormal;
layout (location = 1) out vec3 outPos;
layout (location = 2) flat out vec3 outLightPos;
layout (location = 3) flat out vec3 outCamPos;
layout (location = 4) flat out float outAlpha;

layout (push_constant, row_major) uniform PC {
    mat4 VP;
} pc;

void main()
{
    uint gridX = gl_InstanceIndex / 5;  
    uint gridY = gl_InstanceIndex % 5;
    
    vec2 gridPos = vec2(float(gridX)*0.4-0.8, float(gridY)*0.4-0.8);
    vec4 projPos = pc.VP*vec4(Pos, 1.0);
    gl_Position = pc.VP*vec4(Pos, 1.0) + vec4(gridPos*projPos.w,0.0,0.0);

    outPos = Pos;
    outNormal = normalize(Normal);
    outLightPos = vec3(0.75,0.75,1.75);
    outCamPos = vec3(0.0,0.0,6.0);
    outAlpha = gridX * 0.2 + 0.01;
}