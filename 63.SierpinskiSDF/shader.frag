#version 430 core
// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

layout (set = 1, binding = 0, row_major, std140) uniform UBO 
{
    vec4 position;
    vec4 target;
	vec4 screenResolution; //! only x & y components
	vec4 time; //! only x component
} cameraVectors;

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

vec2 normalizeScreenCoords(vec2 screenCoord)
{
    vec2 result = 2.0 * (screenCoord/cameraVectors.screenResolution.xy - 0.5);
    result.x *= cameraVectors.screenResolution.x/cameraVectors.screenResolution.y;
    return result;
}

void main()
{
	//! use it if you want camera movement by yourself
	// vec3 cameraPosition = cameraVectors.position.xyz;
	//! vec3 cameraTarget = cameraVectors.target.xyz;
	
	float time = cameraVectors.time.x / 10000000.f;
	
	vec3 cameraPosition = vec3(cos(time), 0.f, sin(time));
    vec3 cameraTarget = vec3(0.f, 0.f, 0.f);
    vec2 uv = normalizeScreenCoords(gl_FragCoord.xy);
	
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
		color = vec3(0.75 + sin(time), 0.515, 0.053 + cos(time)) * float(i)/float(MARCHING_STEP);
    }
    
    pixelColor = vec4(color,1.0);
}