#ifndef _NBL_HLSL_EXT_PATHTRACING_SCENE_INCLUDED_
#define _NBL_HLSL_EXT_PATHTRACING_SCENE_INCLUDED_

#include "common.hlsl"
#include "material_system.hlsl"
#include "next_event_estimator.hlsl"
#include "intersector.hlsl"

namespace nbl
{
namespace hlsl
{
namespace ext
{

template<typename Light, typename BxdfNode>
struct Scene
{
    using light_type = Light;
    using bxdfnode_type = BxdfNode;

    NBL_CONSTEXPR_STATIC_INLINE uint32_t maxSphereCount = 25;
    NBL_CONSTEXPR_STATIC_INLINE uint32_t maxTriangleCount = 12;
    NBL_CONSTEXPR_STATIC_INLINE uint32_t maxRectangleCount = 12;

    Shape<PST_SPHERE> spheres[maxSphereCount];
    Shape<PST_TRIANGLE> triangles[maxTriangleCount];
    Shape<PST_RECTANGLE> rectangles[maxRectangleCount];

    uint32_t sphereCount;
    uint32_t triangleCount;
    uint32_t rectangleCount;

    NBL_CONSTEXPR_STATIC_INLINE uint32_t maxLightCount = 4;

    light_type lights[maxLightCount];
    uint32_t lightCount;
    
    NBL_CONSTEXPR_STATIC_INLINE uint32_t maxBxdfCount = 16; // TODO: limit change?

    bxdfnode_type bxdfs[maxBxdfCount];
    uint32_t bxdfCount;

    // AS ases;

    Intersector::IntersectData toIntersectData(uint32_t mode, ProceduralShapeType type)
    {
        Intersector::IntersectData retval;
        retval.mode = mode;

        uint32_t objCount = (type == PST_SPHERE) ? sphereCount :
                            (type == PST_TRIANGLE) ? triangleCount :
                            (type == PST_RECTANGLE) ? rectangleCount :
                            -1;
        retval.data[0] = objCount;
        retval.data[1] = type;
        
        switch (type)
        {
            case PST_SPHERE:
            {
                for (int i = 0; i < objCount; i++)
                {
                    Shape<PST_SPHERE> sphere = spheres[i];
                    retval.data[2 + i * Shape<PST_SPHERE>::ObjSize] = asuint(sphere.position.x);
                    retval.data[2 + i * Shape<PST_SPHERE>::ObjSize + 1] = asuint(sphere.position.y);
                    retval.data[2 + i * Shape<PST_SPHERE>::ObjSize + 2] = asuint(sphere.position.z);
                    retval.data[2 + i * Shape<PST_SPHERE>::ObjSize + 3] = asuint(sphere.radius);
                    retval.data[2 + i * Shape<PST_SPHERE>::ObjSize + 4] = sphere.bsdfLightIDs;
                }
            }
            break;
            case PST_TRIANGLE:
            {
                for (int i = 0; i < objCount; i++)
                {
                    Shape<PST_TRIANGLE> tri = triangles[i];
                    retval.data[2 + i * Shape<PST_TRIANGLE>::ObjSize] = asuint(tri.vertex0.x);
                    retval.data[2 + i * Shape<PST_TRIANGLE>::ObjSize + 1] = asuint(tri.vertex0.y);
                    retval.data[2 + i * Shape<PST_TRIANGLE>::ObjSize + 2] = asuint(tri.vertex0.z);
                    retval.data[2 + i * Shape<PST_TRIANGLE>::ObjSize + 3] = asuint(tri.vertex1.x);
                    retval.data[2 + i * Shape<PST_TRIANGLE>::ObjSize + 4] = asuint(tri.vertex1.y);
                    retval.data[2 + i * Shape<PST_TRIANGLE>::ObjSize + 5] = asuint(tri.vertex1.z);
                    retval.data[2 + i * Shape<PST_TRIANGLE>::ObjSize + 6] = asuint(tri.vertex2.x);
                    retval.data[2 + i * Shape<PST_TRIANGLE>::ObjSize + 7] = asuint(tri.vertex2.y);
                    retval.data[2 + i * Shape<PST_TRIANGLE>::ObjSize + 8] = asuint(tri.vertex2.z);
                    retval.data[2 + i * Shape<PST_TRIANGLE>::ObjSize + 9] = tri.bsdfLightIDs;
                }
            }
            break;
            case PST_RECTANGLE:
            {
                for (int i = 0; i < objCount; i++)
                {
                    Shape<PST_RECTANGLE> rect = rectangles[i];
                    retval.data[2 + i * Shape<PST_RECTANGLE>::ObjSize] = asuint(rect.offset.x);
                    retval.data[2 + i * Shape<PST_RECTANGLE>::ObjSize + 1] = asuint(rect.offset.y);
                    retval.data[2 + i * Shape<PST_RECTANGLE>::ObjSize + 2] = asuint(rect.offset.z);
                    retval.data[2 + i * Shape<PST_RECTANGLE>::ObjSize + 3] = asuint(rect.edge0.x);
                    retval.data[2 + i * Shape<PST_RECTANGLE>::ObjSize + 4] = asuint(rect.edge0.y);
                    retval.data[2 + i * Shape<PST_RECTANGLE>::ObjSize + 5] = asuint(rect.edge0.z);
                    retval.data[2 + i * Shape<PST_RECTANGLE>::ObjSize + 6] = asuint(rect.edge1.x);
                    retval.data[2 + i * Shape<PST_RECTANGLE>::ObjSize + 7] = asuint(rect.edge1.y);
                    retval.data[2 + i * Shape<PST_RECTANGLE>::ObjSize + 8] = asuint(rect.edge1.z);
                    retval.data[2 + i * Shape<PST_RECTANGLE>::ObjSize + 9] = rect.bsdfLightIDs;
                }
            }
            break;
            default:
                // for ASes
                break;
        }
        return retval;
    }

    NextEventEstimator::Event toNextEvent(uint32_t lightID)
    {
        NextEventEstimator::Event retval;

        ObjectID objectID = lights[lightID].objectID;
        retval.mode = objectID.mode;

        retval.data[0] = lightCount;
        retval.data[1] = objectID.type;

        uint32_t id = objectID.id;
        switch (type)
        {
            case PST_SPHERE:
            {
                Shape<PST_SPHERE> sphere = spheres[id];
                retval.data[2] = asuint(sphere.position.x);
                retval.data[3] = asuint(sphere.position.y);
                retval.data[4] = asuint(sphere.position.z);
                retval.data[5] = asuint(sphere.radius);
                retval.data[6] = sphere.bsdfLightIDs;
            }
            break;
            case PST_TRIANGLE:
            {
                Shape<PST_TRIANGLE> tri = triangles[id];
                retval.data[2] = asuint(tri.vertex0.x);
                retval.data[3] = asuint(tri.vertex0.y);
                retval.data[4] = asuint(tri.vertex0.z);
                retval.data[5] = asuint(tri.vertex1.x);
                retval.data[6] = asuint(tri.vertex1.y);
                retval.data[7] = asuint(tri.vertex1.z);
                retval.data[8] = asuint(tri.vertex2.x);
                retval.data[9] = asuint(tri.vertex2.y);
                retval.data[10] = asuint(tri.vertex2.z);
                retval.data[11] = tri.bsdfLightIDs;
            }
            break;
            case PST_RECTANGLE:
            {
                Shape<PST_RECTANGLE> rect = rectangles[id];
                retval.data[2] = asuint(rect.offset.x);
                retval.data[3] = asuint(rect.offset.y);
                retval.data[4] = asuint(rect.offset.z);
                retval.data[5] = asuint(rect.edge0.x);
                retval.data[6] = asuint(rect.edge0.y);
                retval.data[7] = asuint(rect.edge0.z);
                retval.data[8] = asuint(rect.edge1.x);
                retval.data[9] = asuint(rect.edge1.y);
                retval.data[10] = asuint(rect.edge1.z);
                retval.data[11] = rect.bsdfLightIDs;
            }
            break;
            default:
                // for ASes
                break;
        }
        return retval;
    }

    // TODO: get these to work with AS types as well
    uint32_t getBsdfLightIDs(uint32_t id)
    {
        return (objectID.type == PST_SPHERE) ? spheres[id].bsdfLightIDs :
                (objectID.type == PST_TRIANGLE) ? triangles[id].bsdfLightIDs :
                (objectID.type == PST_RECTANGLE) ? rectangles[id].bsdfLightIDs : -1;
    }

    float32_t3 getNormal(uint32_t id, NBL_CONST_REF_ARG(float32_t3) intersection)
    {
        return (objectID.type == PST_SPHERE) ? scene.spheres[id].getNormal(intersection) :
                (objectID.type == PST_TRIANGLE) ? scene.triangles[id].getNormalTimesArea() :
                (objectID.type == PST_RECTANGLE) ? scene.rectangles[id].getNormalTimesArea() :
                (float32_t3)0.0;
    }
};

}
}
}

#endif
