#version 430 core
// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

layout (location = 0) in vec3 Pos;

layout (location = 0) out vec4 outColor;

layout(set = 3, binding = 0) uniform sampler2D tex0;

#define M_PI 3.1415926536

float plot(float cand, float pct, float bold){
  return smoothstep( pct-0.005*bold, pct, cand) -
          smoothstep( pct, pct+0.005*bold, cand);
}

float f(vec2 uv) {
    float angle = (atan(-uv.y/abs(uv.x)) + M_PI/2.0)/(M_PI);
    return texture(tex0,vec2(angle,0.5)).x;
}

void main()
{
   vec2 uv = Pos.xy;
    float dist = sqrt(uv.x*uv.x+uv.y*uv.y);
    vec3 col = plot(dist,0.46,0.75)*vec3(1.0);
    float data = 0.45*f(uv);
    col += plot(dist,data,1.25)*vec3(1.0,0.0,0.0);
    outColor = vec4(col,1.0);
}