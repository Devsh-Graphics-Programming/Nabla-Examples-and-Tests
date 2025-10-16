// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "common.hlsl"

[[vk::binding(0, 0)]] RWTexture2D<float32_t> outIESCandelaImage;
[[vk::binding(1, 0)]] RWTexture2D<float32_t2> outSphericalCoordinatesImage;
[[vk::binding(2, 0)]] RWTexture2D<float32_t3> outOUVProjectionDirectionImage;
[[vk::binding(3, 0)]] RWTexture2D<float32_t2> outPassTMask;

[[vk::push_constant]] struct PushConstants pc;

float32_t3 octahedronUVToDir(float32_t2 uv)
{
    float32_t3 position = float32_t3((uv * 2.0 - 1.0).xy, 0.0);
	float32_t2 absP = float32_t2(abs(position.x), abs(position.y));
	
	position.z = 1.0 - absP.x - absP.y; 
	
	if (position.z < 0.0) 
	{
		position.x = sign(position.x) * (1.0 - absP.y);
		position.y = sign(position.y) * (1.0 - absP.x);
	}

	// rotate position vector around Z-axis with "pc.zAngleDegreeRotation"
	if(pc.zAngleDegreeRotation != 0.0)
	{
		float32_t rDegree = pc.zAngleDegreeRotation;
		
		const float32_t zAngleRadians = float32_t(rDegree * M_PI / 180.0);
		const float32_t cosineV = cos(zAngleRadians);
		const float32_t sineV = sin(zAngleRadians);

		position = float32_t3(cosineV * position.x - cosineV * position.y, sineV * position.x + sineV * position.y, position.z);
	}
	
	return normalize(position);
}

//! Returns spherical coordinates with physics convention in radians
/*
	https://en.wikipedia.org/wiki/Spherical_coordinate_system#/media/File:3D_Spherical.svg
	Retval.x is "theta" polar angle in range [0, PI] & Retval.y "phi" is azimuthal angle
	in [-PI, PI] range
*/

float32_t2 sphericalDirToRadians(float32_t3 direction)
{
	float32_t theta = acos(clamp(direction.z/length(direction), -1.0, 1.0));
	float32_t phi = atan2(direction.y, direction.x); // TODO: check it
	
	return float32_t2(theta, phi);
}

uint32_t implGetVUB(const float32_t angle)
{	
    for(uint32_t i = 0; i < pc.vAnglesCount; ++i)
        if(pc.getVerticalAngle(i) > angle)
            return i;

    return pc.vAnglesCount;
}

uint32_t implGetHUB(const float32_t angle)
{	
	for (uint32_t i = 0; i < pc.hAnglesCount; ++i)
		if (pc.getHorizontalAngle(i) > angle)
			return i;

    return pc.hAnglesCount;
}

uint32_t getVLB(const float32_t angle)
{
	return uint32_t(max(int(implGetVUB(angle)) - 1, 0));
}

uint32_t getHLB(const float32_t angle)
{
	return uint32_t(max(int(implGetHUB(angle)) - 1, 0));
}

uint32_t getVUB(const float32_t angle)
{
	return uint32_t(min(int(implGetVUB(angle)), int(pc.vAnglesCount) - 1));
}

uint32_t getHUB(const float32_t angle)
{
	return uint32_t(min(int(implGetHUB(angle)), int(pc.hAnglesCount) - 1));
}

float32_t getValue(uint32_t i, uint32_t j)
{
	return pc.getData(pc.vAnglesCount * i + j);
}

// symmetry
#define ISOTROPIC 0u
#define QUAD_SYMETRIC 1u
#define HALF_SYMETRIC 2u
#define NO_LATERAL_SYMMET 3u

uint32_t getSymmetry() // TODO: to reduce check time we could pass it with PCs
{
	if(pc.hAnglesCount < 2) // careful here, somebody can break it by feeding us with too much data by mistake
		return ISOTROPIC;
	
	const float32_t hABack = pc.getHorizontalAngle(pc.hAnglesCount - 1);
	
	if(hABack == 90)
		return QUAD_SYMETRIC;
	else if(hABack == 180) // note that OTHER_HALF_SYMMETRIC = HALF_SYMETRIC here
		return HALF_SYMETRIC;
	else
		return NO_LATERAL_SYMMET;
}

float32_t wrapPhi(const float32_t phi, const uint32_t symmetry) //! wrap phi spherical coordinate compoment to range defined by symmetry
{
	switch (symmetry)
	{
		case ISOTROPIC:
			return 0.0;
		case QUAD_SYMETRIC: //! phi MIRROR_REPEAT wrap onto [0, 90] degrees range
		{
			float32_t wrapPhi = abs(phi); //! first MIRROR
			
			if(wrapPhi > M_HALF_PI) //! then REPEAT
				wrapPhi = clamp(M_HALF_PI - (wrapPhi - M_HALF_PI), 0, M_HALF_PI);
			
			return wrapPhi; //! eg. maps (in degrees) 91,269,271 -> 89 and 179,181,359 -> 1
		}
		case HALF_SYMETRIC: //! phi MIRROR wrap onto [0, 180] degrees range
			return abs(phi); //! eg. maps (in degress) 181 -> 179 or 359 -> 1
		case NO_LATERAL_SYMMET:
		{
			if(phi < 0)
				return phi + 2.0 * M_PI;
			else
				return phi;
		}
	}
	
	return 69;
}

float32_t sampleI(const float32_t2 sphericalCoordinates, const uint32_t symmetry)
{
	const float32_t vAngle = degrees(sphericalCoordinates.x), hAngle = degrees(wrapPhi(sphericalCoordinates.y, symmetry));
	
	float32_t vABack = pc.getVerticalAngle(pc.vAnglesCount - 1);
	float32_t hABack = pc.getHorizontalAngle(pc.hAnglesCount - 1);

	if (vAngle > vABack)
		return 0.0;
	
	// bilinear interpolation
	uint32_t j0 = getVLB(vAngle);
	uint32_t j1 = getVUB(vAngle);
	uint32_t i0 = symmetry == ISOTROPIC ? 0 : getHLB(hAngle); 
	uint32_t i1 = symmetry == ISOTROPIC ? 0 : getHUB(hAngle);
	
	float32_t uReciprocal = i1 == i0 ? 1.0 : 1.0 / (pc.getHorizontalAngle(i1) - pc.getHorizontalAngle(i0));
	float32_t vReciprocal = j1 == j0 ? 1.0 : 1.0 / (pc.getVerticalAngle(j1) - pc.getVerticalAngle(j0));
	
	float32_t u = (hAngle - pc.getHorizontalAngle(i0)) * uReciprocal;
	float32_t v = (vAngle - pc.getVerticalAngle(j0)) * vReciprocal;
	
	float32_t s0 = getValue(i0, j0) * (1.0 - v) + getValue(i0, j1) * (v);
	float32_t s1 = getValue(i1, j0) * (1.0 - v) + getValue(i1, j1) * (v);
	
	return s0 * (1.0 - u) + s1 * u;
}

//! Checks if (x,y) /in [0,PI] x [-PI,PI] product
/*
	IES vertical range is [0, 180] degrees
	and horizontal range is [0, 360] degrees 
	but for easier computations (MIRROR & MIRROW_REPEAT operations) 
	we represent horizontal range as [-180, 180] given spherical coordinates
*/

bool isWithinSCDomain(const float32_t2 p)
{
    const float32_t2 lb = float32_t2(0, -M_PI);
    const float32_t2 ub = float32_t2(M_PI, M_PI);

    return all(lb <= p) && all(p <= ub);
}

[numthreads(WORKGROUP_DIMENSION, WORKGROUP_DIMENSION, 1)]
[shader("compute")]
void main(uint32_t3 ID : SV_DispatchThreadID)
{
	uint32_t2 destinationSize;
	outIESCandelaImage.GetDimensions(destinationSize.x, destinationSize.y);
	const uint32_t2 pixelCoordinates = uint32_t2(glsl::gl_GlobalInvocationID().x, glsl::gl_GlobalInvocationID().y);
	
	const float32_t VERTICAL_INVERSE = 1.0f / float32_t(destinationSize.x);
	const float32_t HORIZONTAL_INVERSE = 1.0f / float32_t(destinationSize.y);
	
	if (all(pixelCoordinates < destinationSize))
	{
		const float32_t2 uv = float32_t2((float32_t(pixelCoordinates.x) + 0.5) * VERTICAL_INVERSE, (float32_t(pixelCoordinates.y) + 0.5) * HORIZONTAL_INVERSE);
		const float32_t3 direction = octahedronUVToDir(uv);
		const float32_t2 sphericalCoordinates = sphericalDirToRadians(direction); // third radius spherical compoment is normalized and skipped
		
		const float32_t intensity = sampleI(sphericalCoordinates, getSymmetry());
		
		const float32_t normD = length(direction);
		float32_t2 mask;
		
		if(1.0 - QUANT_ERROR_ADMISSIBLE <= normD && normD <= 1.0 + QUANT_ERROR_ADMISSIBLE)
			mask.x = 1.0; // pass
		else
			mask.x = 0;
			
		if(isWithinSCDomain(sphericalCoordinates))
			mask.y = 1.0; // pass
		else
			mask.y = 0;

		outIESCandelaImage[pixelCoordinates] = uint32_t(intensity / pc.maxIValue);
		outSphericalCoordinatesImage[pixelCoordinates] = sphericalCoordinates;
		outOUVProjectionDirectionImage[pixelCoordinates] = direction;
		outPassTMask[pixelCoordinates] = mask;
	}
}