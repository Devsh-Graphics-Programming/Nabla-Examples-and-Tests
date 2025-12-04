#ifndef _NBL_HLSL_EXT_PATHTRACING_SCENE_INCLUDED_
#define _NBL_HLSL_EXT_PATHTRACING_SCENE_INCLUDED_

#include "common.hlsl"
#include "example_common.hlsl"

using namespace nbl;
using namespace hlsl;

template<typename T>
struct SceneBase
{
    using scalar_type = T;
    using vector3_type = vector<T, 3>;
    using this_t = SceneBase<T>;

    NBL_CONSTEXPR_STATIC_INLINE uint32_t SCENE_SPHERE_COUNT = 9u;
    NBL_CONSTEXPR_STATIC_INLINE uint32_t SCENE_LIGHT_COUNT = 1u;
    NBL_CONSTEXPR_STATIC_INLINE uint32_t SCENE_BXDF_COUNT = 8u;



    // TODO: can static const array of structs and init?
    // static const Shape<scalar_type, PST_SPHERE> scene_spheres[SCENE_SPHERE_COUNT] = {
    //     Shape<scalar_type, PST_SPHERE>::create(vector3_type(0.0, -100.5, -1.0), 100.0, 0u, Light<vector3_type>::INVALID_ID),
    //     Shape<scalar_type, PST_SPHERE>::create(vector3_type(2.0, 0.0, -1.0), 0.5, 1u, Light<vector3_type>::INVALID_ID),
    //     Shape<scalar_type, PST_SPHERE>::create(vector3_type(0.0, 0.0, -1.0), 0.5, 2u, Light<vector3_type>::INVALID_ID),
    //     Shape<scalar_type, PST_SPHERE>::create(vector3_type(-2.0, 0.0, -1.0), 0.5, 3u, Light<vector3_type>::INVALID_ID),
    //     Shape<scalar_type, PST_SPHERE>::create(vector3_type(2.0, 0.0, 1.0), 0.5, 4u, Light<vector3_type>::INVALID_ID),
    //     Shape<scalar_type, PST_SPHERE>::create(vector3_type(0.0, 0.0, 1.0), 0.5, 4u, Light<vector3_type>::INVALID_ID),
    //     Shape<scalar_type, PST_SPHERE>::create(vector3_type(-2.0, 0.0, 1.0), 0.5, 5u, Light<vector3_type>::INVALID_ID),
    //     Shape<scalar_type, PST_SPHERE>::create(vector3_type(0.5, 1.0, 0.5), 0.5, 6u, Light<vector3_type>::INVALID_ID)
    // };
    Shape<scalar_type, PST_SPHERE> scene_spheres[SCENE_SPHERE_COUNT];
};


template<typename T, ProceduralShapeType LightShape>
struct Scene;

template<typename T>
struct Scene<T, PST_SPHERE> : SceneBase<T>
{
    using scalar_type = T;
    using vector3_type = vector<T, 3>;
    using this_t = Scene<T, PST_SPHERE>;
    using base_t = SceneBase<T>;
    using id_type = ObjectID;

    NBL_CONSTEXPR_STATIC_INLINE uint32_t SphereCount = base_t::SCENE_SPHERE_COUNT + base_t::SCENE_LIGHT_COUNT;
    NBL_CONSTEXPR_STATIC_INLINE uint32_t TriangleCount = 0u;
    NBL_CONSTEXPR_STATIC_INLINE uint32_t RectangleCount = 0u;

    Shape<scalar_type, PST_SPHERE> light_spheres[1];
    Shape<scalar_type, PST_TRIANGLE> light_triangles[1];
    Shape<scalar_type, PST_RECTANGLE> light_rectangles[1];

    Shape<scalar_type, PST_SPHERE> getSphere(uint32_t idx)
    {
        assert(idx < SphereCount);
        if (idx < base_t::SCENE_SPHERE_COUNT)
            return base_t::scene_spheres[idx];
        else
            return light_spheres[idx-base_t::SCENE_SPHERE_COUNT];
    }

    Shape<scalar_type, PST_TRIANGLE> getTriangle(uint32_t idx)
    {
        assert(false);
        return light_triangles[0];
    }

    Shape<scalar_type, PST_RECTANGLE> getRectangle(uint32_t idx)
    {
        assert(false);
        return light_rectangles[0];
    }

     void updateLight(NBL_CONST_REF_ARG(float32_t3x4) generalPurposeLightMatrix)
    {
        light_spheres[0].updateTransform(generalPurposeLightMatrix);
    }
    
    uint32_t getBsdfLightIDs(NBL_CONST_REF_ARG(id_type) objectID)
    {
        assert(objectID.shapeType == PST_SPHERE);
        return getSphere(objectID.id).bsdfLightIDs;
    }

    vector3_type getNormal(NBL_CONST_REF_ARG(id_type) objectID, NBL_CONST_REF_ARG(vector3_type) intersection)
    {
        assert(objectID.shapeType == PST_SPHERE);
        return getSphere(objectID.id).getNormal(intersection);
    }
};

template<typename T>
struct Scene<T, PST_TRIANGLE> : SceneBase<T>
{
    using scalar_type = T;
    using vector3_type = vector<T, 3>;
    using this_t = Scene<T, PST_TRIANGLE>;
    using base_t = SceneBase<T>;
    using id_type = ObjectID;

    NBL_CONSTEXPR_STATIC_INLINE uint32_t SphereCount = base_t::SCENE_SPHERE_COUNT;
    NBL_CONSTEXPR_STATIC_INLINE uint32_t TriangleCount = base_t::SCENE_LIGHT_COUNT;
    NBL_CONSTEXPR_STATIC_INLINE uint32_t RectangleCount = 0u;

    Shape<scalar_type, PST_SPHERE> light_spheres[1];
    Shape<scalar_type, PST_TRIANGLE> light_triangles[1];
    Shape<scalar_type, PST_RECTANGLE> light_rectangles[1];

    Shape<scalar_type, PST_SPHERE> getSphere(uint32_t idx)
    {
        assert(idx < SphereCount);
        return base_t::scene_spheres[idx];
    }
    Shape<scalar_type, PST_TRIANGLE> getTriangle(uint32_t idx)
    {
        assert(idx < TriangleCount);
        return light_triangles[idx];
    }
    Shape<scalar_type, PST_RECTANGLE> getRectangle(uint32_t idx)
    {
        assert(false);
        return light_rectangles[0];
    }
    
    void updateLight(NBL_CONST_REF_ARG(float32_t3x4) generalPurposeLightMatrix)
    {
        light_triangles[0].updateTransform(generalPurposeLightMatrix);
    }

    uint32_t getBsdfLightIDs(NBL_CONST_REF_ARG(id_type) objectID)
    {
        assert(objectID.shapeType == PST_SPHERE || objectID.shapeType == PST_TRIANGLE);
        return objectID.shapeType == PST_SPHERE ? getSphere(objectID.id).bsdfLightIDs : getTriangle(objectID.id).bsdfLightIDs;
    }

    vector3_type getNormal(NBL_CONST_REF_ARG(id_type) objectID, NBL_CONST_REF_ARG(vector3_type) intersection)
    {
        assert(objectID.shapeType == PST_SPHERE || objectID.shapeType == PST_TRIANGLE);
        return objectID.shapeType == PST_SPHERE ? getSphere(objectID.id).getNormal(intersection) : getTriangle(objectID.id).getNormalTimesArea();
    }
};

template<typename T>
struct Scene<T, PST_RECTANGLE> : SceneBase<T>
{
    using scalar_type = T;
    using vector3_type = vector<T, 3>;
    using this_t = Scene<T, PST_RECTANGLE>;
    using base_t = SceneBase<T>;
    using id_type = ObjectID;

    NBL_CONSTEXPR_STATIC_INLINE uint32_t SphereCount = base_t::SCENE_SPHERE_COUNT;
    NBL_CONSTEXPR_STATIC_INLINE uint32_t TriangleCount = 0u;
    NBL_CONSTEXPR_STATIC_INLINE uint32_t RectangleCount = base_t::SCENE_LIGHT_COUNT;

    Shape<scalar_type, PST_SPHERE> light_spheres[1];
    Shape<scalar_type, PST_TRIANGLE> light_triangles[1];
    Shape<scalar_type, PST_RECTANGLE> light_rectangles[1];

    Shape<scalar_type, PST_SPHERE> getSphere(uint32_t idx)
    {
        assert(idx < SphereCount);
        return base_t::scene_spheres[idx];
    }
    Shape<scalar_type, PST_TRIANGLE> getTriangle(uint32_t idx)
    {
        assert(false);
        return light_triangles[0];
    }
    Shape<scalar_type, PST_RECTANGLE> getRectangle(uint32_t idx)
    {
        assert(idx < RectangleCount);
        return light_rectangles[idx];
    }

    void updateLight(NBL_CONST_REF_ARG(float32_t3x4) generalPurposeLightMatrix)
    {
        light_rectangles[0].updateTransform(generalPurposeLightMatrix);
    }

    uint32_t getBsdfLightIDs(NBL_CONST_REF_ARG(id_type) objectID)
    {
        assert(objectID.shapeType == PST_SPHERE || objectID.shapeType == PST_RECTANGLE);
        return objectID.shapeType == PST_SPHERE ? getSphere(objectID.id).bsdfLightIDs : getRectangle(objectID.id).bsdfLightIDs;
    }

    vector3_type getNormal(NBL_CONST_REF_ARG(id_type) objectID, NBL_CONST_REF_ARG(vector3_type) intersection)
    {
        assert(objectID.shapeType == PST_SPHERE || objectID.shapeType == PST_RECTANGLE);
        return objectID.shapeType == PST_SPHERE ? getSphere(objectID.id).getNormal(intersection) : getRectangle(objectID.id).getNormalTimesArea();
    }
};

#endif
