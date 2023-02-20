#version 430 core
// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

layout(location = 0) in vec2 TexCoord;
layout(location = 0) out vec4 pixelColor;

#define MARCHING_STEP 64.
#define SDFScale 2.
#define SDFIterationAmount 10

float sceneSDF(vec3 samplePoint)
{
    vec3 a1 = vec3(1,1,1);
    vec3 a2 = vec3(-1,-1,1);
    vec3 a3 = vec3(1,-1,-1);
    vec3 a4 = vec3(-1,1,-1);
    vec3 c;
    int n = 0;
    float dist, d;
	
    while (n < SDFIterationAmount) 
	{
		c = a1; 
		dist = length(samplePoint - a1);
		d = length(samplePoint - a2); 

		if (d < dist) 
		{
			c = a2; 
			dist = d; 
		}
		
		d = length(samplePoint - a3); 
		if (d < dist) 
		{ 
			c = a3;
			dist = d; 
		}
		
		d = length(samplePoint - a4); 
		if (d < dist) 
		{ 
			c = a4; 
			dist = d; 
		}
		samplePoint = SDFScale * samplePoint - c * (SDFScale);
		n++;
    }

    return length(samplePoint) * pow(SDFScale, float(-n));
}

vec3 getCameraviewMarchRayDir(vec2 uv, vec3 cameraPosition, vec3 cameraTarget)
{
    // Calculate camera's "orthonormal basis", i.e. its transform matrix components
    vec3 camForward = normalize(cameraTarget - cameraPosition);
    vec3 camRight = normalize(cross(vec3(0.0, 1.0, 0.0), camForward));
    vec3 camUp = normalize(cross(camForward, camRight));
     
    float fPersp = 2.0;
    vec3 vDir = normalize(uv.x * camRight + uv.y * camUp + camForward * fPersp);
 
    return vDir;
}

float getFinalRayMarchDistance(vec3 position, vec3 direction, float start, float end, inout float i)
{
    float depth = start;
    for(i = 0.; i < MARCHING_STEP; i++)
    {
        float currentSDF =  sceneSDF(position + direction * depth);
        if(currentSDF < 0.005f)
        {
            return depth;
        }
        depth += currentSDF;
        if(depth >= end)
            return end;
    }
}

void main()
{
	// TODO! get from Push Contants
    vec3 cameraTarget = vec3(0, 0, 0);
    vec2 uv = TexCoord;
	
    // TODO! get from Push Contants
	vec3 cameraPosition;

    vec3 viewMarchRay = getCameraviewMarchRayDir(uv, cameraPosition, cameraTarget);
    
    float i;
    float finalMarchDistance = getFinalRayMarchDistance(cameraPosition, viewMarchRay, 0.f,200.f,i);
    vec3 color = vec3(finalMarchDistance);
    
    if((finalMarchDistance - 200.f) > 0.001f)
    {
        color = vec3(0.0529, 0.0808, 0.1922);
    }
    else
    {
        color = vec3(finalMarchDistance*0.1); 
    }
    
    pixelColor = vec4(color,1.0);
}