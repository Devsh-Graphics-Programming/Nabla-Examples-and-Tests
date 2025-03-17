#ifndef RQG_COMMON_HLSL
#define RQG_COMMON_HLSL

#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/cpp_compat/basic.h"

NBL_CONSTEXPR uint32_t WorkgroupSize = 16;
NBL_CONSTEXPR uint32_t MAX_UNORM_10 = 1023;
NBL_CONSTEXPR uint32_t MAX_UNORM_22 = 4194303;

inline uint32_t packUnorm10(float32_t v)
{
    return trunc(v * float32_t(MAX_UNORM_10) + 0.5f);
}

inline float32_t unpackUnorm10(uint32_t packed)
{
    return float32_t(packed & 0x3ff) * (1.0f / float32_t(MAX_UNORM_10));
}

inline uint32_t packUnorm22(float32_t v)
{
    const float maxValue = float32_t(MAX_UNORM_22);
    return trunc(v * maxValue + 0.5f);
}

inline float32_t unpackUnorm22(uint32_t packed)
{
    const float maxValue = float32_t(MAX_UNORM_22);
    return float32_t(packed & 0x3fffff) * (1.0f / maxValue);
}

inline uint32_t packUnorm3x10(float32_t3 v)
{
    return (packUnorm10(v.z) << 20 | (packUnorm10(v.y) << 10 | packUnorm10(v.x)));
}

inline float32_t3 unpackUnorm3x10(uint32_t packed)
{
    return float32_t3(unpackUnorm10(packed), unpackUnorm10(packed >> 10), unpackUnorm10(packed >> 20));
}

struct Material
{
	float32_t3 ambient;
    float32_t3 diffuse;
    float32_t3 specular;
    float32_t shininess;
    float32_t alpha;

    bool isTransparent() NBL_CONST_MEMBER_FUNC
    {
        return alpha < 1.0;
    }
};

struct MaterialPacked
{
	uint32_t ambient;
    uint32_t diffuse;
    uint32_t specular;
    uint32_t shininess: 22;
    uint32_t alpha : 10;

    bool isTransparent() NBL_CONST_MEMBER_FUNC
    {
        return alpha != MAX_UNORM_10;
    }
};

struct SProceduralGeomInfo
{
    MaterialPacked material;
    float32_t3 center;
    float32_t radius;
};

struct STriangleGeomInfo
{
    MaterialPacked material;
    uint64_t vertexBufferAddress;
    uint64_t indexBufferAddress;

    uint32_t vertexStride : 26;
    uint32_t objType: 3;
    uint32_t indexType : 2; // 16 bit, 32 bit or none
    uint32_t smoothNormals : 1;	// flat for cube, rectangle, disk

};

enum E_GEOM_TYPE : uint16_t
{
    EGT_TRIANGLES,
    EGT_PROCEDURAL,
    EGT_COUNT
};

enum E_RAY_TYPE : uint16_t
{
    ERT_PRIMARY, // Ray shoot from camera
    ERT_OCCLUSION,
    ERT_COUNT
};

enum E_MISS_TYPE : uint16_t
{
    EMT_PRIMARY,
    EMT_OCCLUSION,
    EMT_COUNT
};

enum E_LIGHT_TYPE : uint16_t
{
    ELT_DIRECTIONAL,
    ELT_POINT,
    ELT_SPOT,
    ELT_COUNT
};

struct Light
{
    float32_t3 direction;
    float32_t3 position;
    float32_t outerCutoff;
    uint16_t type;


#ifndef __HLSL_VERSION
    bool operator==(const Light&) const = default;
#endif

};

static const float LightIntensity = 100.0f;

struct SPushConstants
{
    uint64_t proceduralGeomInfoBuffer;
    uint64_t triangleGeomInfoBuffer;

    float32_t3 camPos;
    uint32_t frameCounter;
    float32_t4x4 invMVP;

    Light light;
};


struct RayLight
{
    float32_t3 inHitPosition;
    float32_t outLightDistance;
    float32_t3 outLightDir;
    float32_t outIntensity;
};

struct ProceduralHitAttribute
{
    MaterialPacked material;
    float32_t3 center;
};


#ifdef __HLSL_VERSION

struct [raypayload] OcclusionPayload
{
    float32_t attenuation : read(caller) : write(caller, anyhit);
};

struct [raypayload] PrimaryPayload
{
    MaterialPacked material : read(caller) : write(closesthit);
    float32_t3 worldNormal : read(caller) : write(closesthit);
    float32_t rayDistance : read(caller) : write(closesthit, miss);
    float32_t alphaThreshold : read(closesthit, anyhit) : write(caller);
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
	float32_t dotNL = max(dot(normal, light_dir), 0.0);
	float32_t3 c = mat.diffuse * dotNL;
	return c;
}

float32_t3 computeSpecular(Material mat, float32_t3 view_dir, 
	float32_t3 light_dir, float32_t3 normal)
{
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

namespace nbl
{
namespace hlsl
{
namespace impl
{

template<>
struct static_cast_helper<Material, MaterialPacked>
{
    static inline Material cast(MaterialPacked packed)
    {
        Material material;
        material.ambient = unpackUnorm3x10(packed.ambient);
        material.diffuse = unpackUnorm3x10(packed.diffuse);
        material.specular = unpackUnorm3x10(packed.specular);
        material.shininess = unpackUnorm22(packed.shininess);
        material.alpha = unpackUnorm10(packed.alpha);
        return material;
    }
};

template<>
struct static_cast_helper<MaterialPacked, Material>
{
    static inline MaterialPacked cast(Material material)
    {
        MaterialPacked packed;
        packed.ambient = packUnorm3x10(material.ambient);
        packed.diffuse = packUnorm3x10(material.diffuse);
        packed.specular = packUnorm3x10(material.specular);
        packed.shininess = packUnorm22(material.shininess);
        packed.alpha = packUnorm10(material.alpha);
        return packed;
    }
};

}
}
}

#endif  // RQG_COMMON_HLSL
