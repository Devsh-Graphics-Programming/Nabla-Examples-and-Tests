#ifndef RQG_COMMON_HLSL
#define RQG_COMMON_HLSL

#include "nbl/builtin/hlsl/cpp_compat.hlsl"

NBL_CONSTEXPR uint32_t WorkgroupSize = 16;

struct Material
{
	float32_t3 ambient;
    float32_t3 diffuse;
    float32_t3 specular;
    float32_t shininess;
    float32_t dissolve; // 1 == opaque; 0 == fully transparent
    uint32_t illum; // illumination model (see http://www.fileformat.info/format/material/)
};

struct SProceduralGeomInfo
{
    float32_t3 center;
    float32_t radius;
    Material material;
};

struct Aabb
{
    float32_t3 minimum;
    float32_t3 maximum;
};

struct STriangleGeomInfo
{
    uint64_t vertexBufferAddress;
    uint64_t indexBufferAddress;

    uint32_t vertexStride : 29;
    uint32_t indexType : 2; // 16 bit, 32 bit or none
    uint32_t smoothNormals : 1;	// flat for cube, rectangle, disk

    uint32_t objType;

    Material material;
};

enum E_GEOM_TYPE : int32_t
{
    EGT_TRIANGLES,
    EGT_PROCEDURAL,
    EGT_COUNT
};

enum E_LIGHT_TYPE : int32_t
{
    ELT_DIRECTIONAL,
    ELT_POINT,
    ELT_SPOT,
    ELT_COUNT
};

enum E_RAY_TYPE : int32_t
{
    ERT_PRIMARY, // Ray shoot from camera
    ERT_OCCLUSION,
    ERT_COUNT
};

enum E_MISS_TYPE : int32_t
{
    EMT_PRIMARY,
    EMT_OCCLUSION,
    EMT_COUNT
};

struct Light
{
    float32_t3 direction;
    float32_t3 position;
    float32_t intensity;
    float32_t innerCutoff;
    float32_t outerCutoff;
    int32_t type;

#ifndef __HLSL_VERSION
    bool operator==(const Light&) const = default;
#endif

};

struct SPushConstants
{
    Light light;

    float32_t3 camPos;
    float32_t4x4 invMVP;

    uint64_t proceduralGeomInfoBuffer;
    uint64_t triangleGeomInfoBuffer;
    uint32_t frameCounter;
};


struct RayLight
{
    float32_t3 inHitPosition;
    float32_t outLightDistance;
    float32_t3 outLightDir;
    float32_t outIntensity;
};

#ifdef __HLSL_VERSION

struct [raypayload] ColorPayload
{
	float32_t3 hitValue : read(caller) : write(closesthit,miss);
    uint32_t seed : read(closesthit,anyhit) : write(caller);
};

struct [raypayload] ShadowPayload
{
	bool isShadowed : read(caller) : write(caller,miss);
    uint32_t seed : read(anyhit) : write(caller);
};

enum ObjectType : uint32_t  // matches c++
{
    OT_CUBE = 0,
    OT_SPHERE,
    OT_CYLINDER,
    OT_RECTANGLE,
    OT_DISK,
    OT_ARROW,
    OT_CONE,
    OT_ICOSPHERE,

    OT_COUNT
};

static uint32_t s_offsetsToNormalBytes[OT_COUNT] = { 18, 24, 24, 20, 20, 24, 16, 12 };	// based on normals data position
float32_t3 computeDiffuse(Material mat, float32_t3 light_dir, float32_t3 normal)
{
	// Lambertian
	float32_t dotNL = max(dot(normal, light_dir), 0.0);
	float32_t3 c = mat.diffuse * dotNL;
	if (mat.illum >= 1)
		c += mat.ambient;
	return c;
}

float32_t3 computeSpecular(Material mat, float32_t3 view_dir, 
	float32_t3 light_dir, float32_t3 normal)
{
	if (mat.illum < 2)
		return float32_t3(0, 0, 0);

	// Compute specular only if not in shadow
	const float32_t kPi = 3.14159265;
	const float32_t kShininess = max(mat.shininess, 4.0);

	// Specular
	const float32_t kEnergyConservation = (2.0 + kShininess) / (2.0 * kPi);
	float32_t3 V = normalize(-view_dir);
	float32_t3 R = reflect(-light_dir, normal);
	float32_t specular = kEnergyConservation * pow(max(dot(V, R), 0.0), kShininess);

	return float32_t3(mat.specular * specular);
}
#endif

#endif  // RQG_COMMON_HLSL
