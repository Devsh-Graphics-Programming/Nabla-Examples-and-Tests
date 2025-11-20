#ifndef _NBL_HLSL_EXT_PATHTRACING_SCENE_INCLUDED_
#define _NBL_HLSL_EXT_PATHTRACING_SCENE_INCLUDED_

#include "common.hlsl"

namespace nbl
{
namespace hlsl
{
namespace ext
{

template<typename T>
struct Scene
{
    using scalar_type = T;
    using vector3_type = vector<T, 3>;
    using this_t = Scene<T>;

    // NBL_CONSTEXPR_STATIC_INLINE uint32_t maxSphereCount = 25;
    // NBL_CONSTEXPR_STATIC_INLINE uint32_t maxTriangleCount = 12;
    // NBL_CONSTEXPR_STATIC_INLINE uint32_t maxRectangleCount = 12;

#if SPHERE_COUNT < 1
#define SCENE_SPHERE_COUNT 1
#else
#define SCENE_SPHERE_COUNT SPHERE_COUNT
#endif

#if TRIANGLE_COUNT < 1
#define SCENE_TRIANGLE_COUNT 1
#else
#define SCENE_TRIANGLE_COUNT TRIANGLE_COUNT
#endif

#if RECTANGLE_COUNT < 1
#define SCENE_RECTANGLE_COUNT 1
#else
#define SCENE_RECTANGLE_COUNT RECTANGLE_COUNT
#endif

    Shape<scalar_type, PST_SPHERE> spheres[SCENE_SPHERE_COUNT];
    Shape<scalar_type, PST_TRIANGLE> triangles[SCENE_TRIANGLE_COUNT];
    Shape<scalar_type, PST_RECTANGLE> rectangles[SCENE_RECTANGLE_COUNT];

    uint32_t sphereCount;
    uint32_t triangleCount;
    uint32_t rectangleCount;

    // NBL_CONSTEXPR_STATIC_INLINE uint32_t maxLightCount = 4;
    // NBL_CONSTEXPR_STATIC_INLINE uint32_t maxBxdfCount = 16;

    static this_t create(
        NBL_CONST_REF_ARG(Shape<scalar_type, PST_SPHERE>) spheres[SCENE_SPHERE_COUNT],
        NBL_CONST_REF_ARG(Shape<scalar_type, PST_TRIANGLE>) triangles[SCENE_TRIANGLE_COUNT],
        NBL_CONST_REF_ARG(Shape<scalar_type, PST_RECTANGLE>) rectangles[SCENE_RECTANGLE_COUNT],
        uint32_t sphereCount, uint32_t triangleCount, uint32_t rectangleCount)
    {
        this_t retval;
        retval.spheres = spheres;
        retval.triangles = triangles;
        retval.rectangles = rectangles;
        retval.sphereCount = sphereCount;
        retval.triangleCount = triangleCount;
        retval.rectangleCount = rectangleCount;
        return retval;
    }

#undef SCENE_SPHERE_COUNT
#undef SCENE_TRIANGLE_COUNT
#undef SCENE_RECTANGLE_COUNT

    // TODO: get these to work with AS types as well
    uint32_t getBsdfLightIDs(NBL_CONST_REF_ARG(ObjectID) objectID)
    {
        return (objectID.shapeType == PST_SPHERE) ? spheres[objectID.id].bsdfLightIDs :
                (objectID.shapeType == PST_TRIANGLE) ? triangles[objectID.id].bsdfLightIDs :
                (objectID.shapeType == PST_RECTANGLE) ? rectangles[objectID.id].bsdfLightIDs : -1;
    }

    vector3_type getNormal(NBL_CONST_REF_ARG(ObjectID) objectID, NBL_CONST_REF_ARG(vector3_type) intersection)
    {
        return (objectID.shapeType == PST_SPHERE) ? spheres[objectID.id].getNormal(intersection) :
                (objectID.shapeType == PST_TRIANGLE) ? triangles[objectID.id].getNormalTimesArea() :
                (objectID.shapeType == PST_RECTANGLE) ? rectangles[objectID.id].getNormalTimesArea() :
                hlsl::promote<vector3_type>(0.0);
    }
};

}
}
}

#endif
