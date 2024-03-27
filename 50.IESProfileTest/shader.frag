#version 430 core
// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include <nbl/builtin/glsl/ies/functions.glsl>

layout (location = 0) in vec3 Pos;

layout (location = 0) out vec4 outColor;

layout(set = 3, binding = 0) uniform sampler2D inIESCandelaImage;
layout(set = 3, binding = 1) uniform sampler2D inSphericalCoordinatesImage;
layout(set = 3, binding = 2) uniform sampler2D inOUVProjectionDirectionImage;
layout(set = 3, binding = 3) uniform sampler2D inPassTMask;

layout(push_constant) uniform PushConstants
{
    float maxIValue;
	float zAngleDegreeRotation;
	uint mode;
	uint dummy;
} pc;

#define M_PI 3.1415926536

float plot(float cand, float pct, float bold){
  return smoothstep( pct-0.005*bold, pct, cand) -
          smoothstep( pct, pct+0.005*bold, cand);
}

// vertical cut of IES (i.e. cut by plane x = 0)
float f(vec2 uv) {
    return texture(inIESCandelaImage,nbl_glsl_IES_convert_dir_to_uv(normalize(vec3(uv.x, 0.001, uv.y)))).x;
    // float vangle = (abs(atan(uv.x,uv.y)))/(M_PI);
    // float hangle = uv.x <= 0.0 ? 0.0 : 1.0;
    // return texture(inIESCandelaImage,vec2(hangle,vangle)).x;
}

void main()
{
    vec2 ndc = Pos.xy;
	vec2 uv = (ndc + 1) / 2;
	
	if(pc.mode == 0)
	{
		float dist = length(ndc)*1.015625;
		vec3 col = vec3(plot(dist,1.0,0.75));

		float normalizedStrength = f(ndc);
		if (dist<normalizedStrength)
			col += vec3(1.0,0.0,0.0);
		outColor = vec4(col,1.0);
	}
	else if(pc.mode == 1)
	{
		outColor = texture(inIESCandelaImage, uv);		
	}
	else if(pc.mode == 2)
	{
		outColor = texture(inSphericalCoordinatesImage, uv);
	}
	else if(pc.mode == 3)
	{
		outColor = texture(inOUVProjectionDirectionImage, uv);
	}
	else
	{
		outColor = texture(inPassTMask, uv);
	}	
}