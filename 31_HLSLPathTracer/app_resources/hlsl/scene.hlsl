#ifndef _NBL_HLSL_EXT_PATHTRACING_SCENE_INCLUDED_
#define _NBL_HLSL_EXT_PATHTRACING_SCENE_INCLUDED_

#include "common.hlsl"

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

    // obsolete?
    // Intersector::IntersectData toIntersectData(uint32_t mode, ProceduralShapeType type)
    // {
    //     Intersector::IntersectData retval;
    //     retval.mode = mode;

    //     uint32_t objCount = (type == PST_SPHERE) ? sphereCount :
    //                         (type == PST_TRIANGLE) ? triangleCount :
    //                         (type == PST_RECTANGLE) ? rectangleCount :
    //                         -1;
    //     retval.data[0] = objCount;
    //     retval.data[1] = type;

    //     switch (type)
    //     {
    //         case PST_SPHERE:
    //         {
    //             for (int i = 0; i < objCount; i++)
    //             {
    //                 Shape<PST_SPHERE> sphere = spheres[i];
    //                 uint32_t3 uintPos = bit_cast<uint32_t3, float32_t3>(sphere.position);
    //                 retval.data[2 + i * Shape<PST_SPHERE>::ObjSize] = uintPos.x;
    //                 retval.data[2 + i * Shape<PST_SPHERE>::ObjSize + 1] = uintPos.y;
    //                 retval.data[2 + i * Shape<PST_SPHERE>::ObjSize + 2] = uintPos.z;
    //                 retval.data[2 + i * Shape<PST_SPHERE>::ObjSize + 3] = bit_cast<uint32_t, float32_t>(sphere.radius2);
    //                 retval.data[2 + i * Shape<PST_SPHERE>::ObjSize + 4] = sphere.bsdfLightIDs;
    //             }
    //         }
    //         break;
    //         case PST_TRIANGLE:
    //         {
    //             for (int i = 0; i < objCount; i++)
    //             {
    //                 Shape<PST_TRIANGLE> tri = triangles[i];
    //                 retval.data[2 + i * Shape<PST_TRIANGLE>::ObjSize] = asuint(tri.vertex0.x);
    //                 retval.data[2 + i * Shape<PST_TRIANGLE>::ObjSize + 1] = asuint(tri.vertex0.y);
    //                 retval.data[2 + i * Shape<PST_TRIANGLE>::ObjSize + 2] = asuint(tri.vertex0.z);
    //                 retval.data[2 + i * Shape<PST_TRIANGLE>::ObjSize + 3] = asuint(tri.vertex1.x);
    //                 retval.data[2 + i * Shape<PST_TRIANGLE>::ObjSize + 4] = asuint(tri.vertex1.y);
    //                 retval.data[2 + i * Shape<PST_TRIANGLE>::ObjSize + 5] = asuint(tri.vertex1.z);
    //                 retval.data[2 + i * Shape<PST_TRIANGLE>::ObjSize + 6] = asuint(tri.vertex2.x);
    //                 retval.data[2 + i * Shape<PST_TRIANGLE>::ObjSize + 7] = asuint(tri.vertex2.y);
    //                 retval.data[2 + i * Shape<PST_TRIANGLE>::ObjSize + 8] = asuint(tri.vertex2.z);
    //                 retval.data[2 + i * Shape<PST_TRIANGLE>::ObjSize + 9] = tri.bsdfLightIDs;
    //             }
    //         }
    //         break;
    //         case PST_RECTANGLE:
    //         {
    //             for (int i = 0; i < objCount; i++)
    //             {
    //                 Shape<PST_RECTANGLE> rect = rectangles[i];
    //                 retval.data[2 + i * Shape<PST_RECTANGLE>::ObjSize] = asuint(rect.offset.x);
    //                 retval.data[2 + i * Shape<PST_RECTANGLE>::ObjSize + 1] = asuint(rect.offset.y);
    //                 retval.data[2 + i * Shape<PST_RECTANGLE>::ObjSize + 2] = asuint(rect.offset.z);
    //                 retval.data[2 + i * Shape<PST_RECTANGLE>::ObjSize + 3] = asuint(rect.edge0.x);
    //                 retval.data[2 + i * Shape<PST_RECTANGLE>::ObjSize + 4] = asuint(rect.edge0.y);
    //                 retval.data[2 + i * Shape<PST_RECTANGLE>::ObjSize + 5] = asuint(rect.edge0.z);
    //                 retval.data[2 + i * Shape<PST_RECTANGLE>::ObjSize + 6] = asuint(rect.edge1.x);
    //                 retval.data[2 + i * Shape<PST_RECTANGLE>::ObjSize + 7] = asuint(rect.edge1.y);
    //                 retval.data[2 + i * Shape<PST_RECTANGLE>::ObjSize + 8] = asuint(rect.edge1.z);
    //                 retval.data[2 + i * Shape<PST_RECTANGLE>::ObjSize + 9] = rect.bsdfLightIDs;
    //             }
    //         }
    //         break;
    //         default:
    //             // for ASes
    //             break;
    //     }
    //     return retval;
    // }

    NextEventEstimator::Event toNextEvent(uint32_t lightID)
    {
        NextEventEstimator::Event retval;

        ObjectID objectID = lights[lightID].objectID;
        retval.mode = objectID.mode;

        retval.data[0] = lightCount;
        retval.data[1] = objectID.shapeType;

        uint32_t id = objectID.id;
        switch (objectID.shapeType)
        {
            case PST_SPHERE:
            {
                Shape<PST_SPHERE> sphere = spheres[id];
                uint32_t3 position = bit_cast<uint32_t3>(sphere.position);
                retval.data[2] = position.x;
                retval.data[3] = position.y;
                retval.data[4] = position.z;
                retval.data[5] = bit_cast<uint32_t>(sphere.radius2);
                retval.data[6] = sphere.bsdfLightIDs;
            }
            break;
            case PST_TRIANGLE:
            {
                Shape<PST_TRIANGLE> tri = triangles[id];
                uint32_t3 vertex = bit_cast<uint32_t3>(tri.vertex0);
                retval.data[2] = vertex.x;
                retval.data[3] = vertex.y;
                retval.data[4] = vertex.z;
                vertex = bit_cast<uint32_t3>(tri.vertex1);
                retval.data[5] = vertex.x;
                retval.data[6] = vertex.y;
                retval.data[7] = vertex.z;
                vertex = bit_cast<uint32_t3>(tri.vertex2);
                retval.data[8] = vertex.x;
                retval.data[9] = vertex.y;
                retval.data[10] = vertex.z;
                retval.data[11] = tri.bsdfLightIDs;
            }
            break;
            case PST_RECTANGLE:
            {
                Shape<PST_RECTANGLE> rect = rectangles[id];
                uint32_t3 tmp = bit_cast<uint32_t3>(rect.offset);
                retval.data[2] = tmp.x;
                retval.data[3] = tmp.y;
                retval.data[4] = tmp.z;
                tmp = bit_cast<uint32_t3>(rect.edge0);
                retval.data[5] = tmp.x;
                retval.data[6] = tmp.y;
                retval.data[7] = tmp.z;
                tmp = bit_cast<uint32_t3>(rect.edge1);
                retval.data[8] = tmp.x;
                retval.data[9] = tmp.y;
                retval.data[10] = tmp.z;
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
    uint32_t getBsdfLightIDs(NBL_CONST_REF_ARG(ObjectID) objectID)
    {
        return (objectID.shapeType == PST_SPHERE) ? spheres[objectID.id].bsdfLightIDs :
                (objectID.shapeType == PST_TRIANGLE) ? triangles[objectID.id].bsdfLightIDs :
                (objectID.shapeType == PST_RECTANGLE) ? rectangles[objectID.id].bsdfLightIDs : -1;
    }

    float32_t3 getNormal(NBL_CONST_REF_ARG(ObjectID) objectID, NBL_CONST_REF_ARG(float32_t3) intersection)
    {
        return (objectID.shapeType == PST_SPHERE) ? spheres[objectID.id].getNormal(intersection) :
                (objectID.shapeType == PST_TRIANGLE) ? triangles[objectID.id].getNormalTimesArea() :
                (objectID.shapeType == PST_RECTANGLE) ? rectangles[objectID.id].getNormalTimesArea() :
                (float32_t3)0.0;
    }
};

}
}
}

#endif
