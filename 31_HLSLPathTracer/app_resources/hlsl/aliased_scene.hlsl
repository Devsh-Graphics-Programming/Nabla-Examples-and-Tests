// static const uint32_t aliased_scene[] = {
//     // spheres: float32_t3 position - float32_t radius2 - uint32_t bsdfLightIDs
//     // (8) or (9) if sphere light
//     bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(-100.5), bit_cast<uint32_t>(-1.0), bit_cast<uint32_t>(10000.0), glsl::bitfieldInsert<uint32_t>(0u, light_type::INVALID_ID, 16, 16),
//     bit_cast<uint32_t>(2.0), bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(-1.0), bit_cast<uint32_t>(0.25), glsl::bitfieldInsert<uint32_t>(1u, light_type::INVALID_ID, 16, 16),
//     bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(-1.0), bit_cast<uint32_t>(0.25), glsl::bitfieldInsert<uint32_t>(2u, light_type::INVALID_ID, 16, 16),
//     bit_cast<uint32_t>(-2.0), bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(-1.0), bit_cast<uint32_t>(0.25), glsl::bitfieldInsert<uint32_t>(3u, light_type::INVALID_ID, 16, 16),
//     bit_cast<uint32_t>(2.0), bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(1.0), bit_cast<uint32_t>(0.25), glsl::bitfieldInsert<uint32_t>(4u, light_type::INVALID_ID, 16, 16),
//     bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(1.0), bit_cast<uint32_t>(0.25), glsl::bitfieldInsert<uint32_t>(4u, light_type::INVALID_ID, 16, 16),
//     bit_cast<uint32_t>(-2.0), bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(1.0), bit_cast<uint32_t>(0.25), glsl::bitfieldInsert<uint32_t>(5u, light_type::INVALID_ID, 16, 16),
//     bit_cast<uint32_t>(0.5), bit_cast<uint32_t>(1.0), bit_cast<uint32_t>(0.5), bit_cast<uint32_t>(0.25), glsl::bitfieldInsert<uint32_t>(6u, light_type::INVALID_ID, 16, 16),
// #ifdef SPHERE_LIGHT
//     bit_cast<uint32_t>(-1.5), bit_cast<uint32_t>(1.5), bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.09), glsl::bitfieldInsert<uint32_t>(bxdfnode_type::INVALID_ID, 0u, 16, 16),
// #endif

//     // triangles: float32_t3 vertex0 - float32_t3 vertex1 - float32_t3 vertex2 - uint32_t bsdfLightIDs
//     // (1) always fill anyways because scene needs to have 1 triangle in array
//     bit_cast<uint32_t>(-18.0), bit_cast<uint32_t>(3.5), bit_cast<uint32_t>(3.0),
//     bit_cast<uint32_t>(-12.0), bit_cast<uint32_t>(3.5), bit_cast<uint32_t>(0.0),
//     bit_cast<uint32_t>(-15.0), bit_cast<uint32_t>(8.0), bit_cast<uint32_t>(-3.0),
//     glsl::bitfieldInsert<uint32_t>(bxdfnode_type::INVALID_ID, 0u, 16, 16),

//     // rectangles: float32_t3 offset - float32_t3 edge0 - float32_t3 edge1 - uint32_t bsdfLightIDs
//     // (1) same as triangle
//     bit_cast<uint32_t>(-3.8), bit_cast<uint32_t>(0.35), bit_cast<uint32_t>(1.3),
//     bit_cast<uint32_t>(6.261), bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(-3.1305),
//     bit_cast<uint32_t>(0.0298), bit_cast<uint32_t>(-0.745), bit_cast<uint32_t>(0.0596),
//     glsl::bitfieldInsert<uint32_t>(bxdfnode_type::INVALID_ID, 0u, 16, 16),

//     // sphereCount, triangleCount, rectangleCount
//     SPHERE_COUNT, TRIANGLE_COUNT, RECTANGLE_COUNT,

//     // lights: float32_t3 radiance - uint32_t id - uint32_t mode - uint32_t shapeType
//     bit_cast<uint32_t>(30.0), bit_cast<uint32_t>(25.0), bit_cast<uint32_t>(15.0),
// #ifdef SPHERE_LIGHT
//     8u,
// #else
//     0u,
// #endif
//     2u, LIGHT_TYPE,

//     // lightCount
//     LIGHT_COUNT,

//     // 7 bxdfs: float32_t3 albedo - uint32_t materialType - params_type params
//     // params_type: bool is_aniso - float32_t2 A - float32_t3 ior0 - float32_t3 ior1 - float eta - float32_t3 eta2 - float32_t3 luminosityContributionHint
//     // bxdf 0
//     bit_cast<uint32_t>(0.8), bit_cast<uint32_t>(0.8), bit_cast<uint32_t>(0.8),
//     0u,
//     0u, bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0),
//     bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0),
//     bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0),
//     bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0),
//     bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0),

//     // bxdf 1
//     bit_cast<uint32_t>(0.8), bit_cast<uint32_t>(0.4), bit_cast<uint32_t>(0.4),
//     0u,
//     0u, bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0),
//     bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0),
//     bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0),
//     bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0),
//     bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0),

//     // bxdf 2
//     bit_cast<uint32_t>(0.4), bit_cast<uint32_t>(0.8), bit_cast<uint32_t>(0.4),
//     0u,
//     0u, bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0),
//     bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0),
//     bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0),
//     bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0),
//     bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0),

//     // bxdf 3
//     bit_cast<uint32_t>(1.0), bit_cast<uint32_t>(1.0), bit_cast<uint32_t>(1.0),
//     1u,
//     0u, bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0),
//     bit_cast<uint32_t>(1.02), bit_cast<uint32_t>(1.02), bit_cast<uint32_t>(1.3),
//     bit_cast<uint32_t>(1.0), bit_cast<uint32_t>(1.0), bit_cast<uint32_t>(2.0),
//     bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0),
//     bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0),

//     // bxdf 4
//     bit_cast<uint32_t>(1.0), bit_cast<uint32_t>(1.0), bit_cast<uint32_t>(1.0),
//     1u,
//     0u, bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0),
//     bit_cast<uint32_t>(1.02), bit_cast<uint32_t>(1.3), bit_cast<uint32_t>(1.02),
//     bit_cast<uint32_t>(1.0), bit_cast<uint32_t>(2.0), bit_cast<uint32_t>(1.0),
//     bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0),
//     bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0),

//     // bxdf 5
//     bit_cast<uint32_t>(1.0), bit_cast<uint32_t>(1.0), bit_cast<uint32_t>(1.0),
//     1u,
//     0u, bit_cast<uint32_t>(0.15), bit_cast<uint32_t>(0.15),
//     bit_cast<uint32_t>(1.02), bit_cast<uint32_t>(1.3), bit_cast<uint32_t>(1.02),
//     bit_cast<uint32_t>(1.0), bit_cast<uint32_t>(2.0), bit_cast<uint32_t>(1.0),
//     bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0),
//     bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0),

//     // bxdf 6
//     bit_cast<uint32_t>(1.0), bit_cast<uint32_t>(1.0), bit_cast<uint32_t>(1.0),
//     2u,
//     0u, bit_cast<uint32_t>(0.0625), bit_cast<uint32_t>(0.0625),
//     bit_cast<uint32_t>(1.0), bit_cast<uint32_t>(1.0), bit_cast<uint32_t>(1.0),
//     bit_cast<uint32_t>(0.71), bit_cast<uint32_t>(0.69), bit_cast<uint32_t>(0.67),
//     bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0),
//     bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0),

//     // bxdfCount
//     BXDF_COUNT
// };

static const uint32_t aliased_spheres[] = {
    bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(-100.5), bit_cast<uint32_t>(-1.0), bit_cast<uint32_t>(10000.0), glsl::bitfieldInsert<uint32_t>(0u, light_type::INVALID_ID, 16, 16),
    bit_cast<uint32_t>(2.0), bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(-1.0), bit_cast<uint32_t>(0.25), glsl::bitfieldInsert<uint32_t>(1u, light_type::INVALID_ID, 16, 16),
    bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(-1.0), bit_cast<uint32_t>(0.25), glsl::bitfieldInsert<uint32_t>(2u, light_type::INVALID_ID, 16, 16),
    bit_cast<uint32_t>(-2.0), bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(-1.0), bit_cast<uint32_t>(0.25), glsl::bitfieldInsert<uint32_t>(3u, light_type::INVALID_ID, 16, 16),
    bit_cast<uint32_t>(2.0), bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(1.0), bit_cast<uint32_t>(0.25), glsl::bitfieldInsert<uint32_t>(4u, light_type::INVALID_ID, 16, 16),
    bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(1.0), bit_cast<uint32_t>(0.25), glsl::bitfieldInsert<uint32_t>(4u, light_type::INVALID_ID, 16, 16),
    bit_cast<uint32_t>(-2.0), bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(1.0), bit_cast<uint32_t>(0.25), glsl::bitfieldInsert<uint32_t>(5u, light_type::INVALID_ID, 16, 16),
    bit_cast<uint32_t>(0.5), bit_cast<uint32_t>(1.0), bit_cast<uint32_t>(0.5), bit_cast<uint32_t>(0.25), glsl::bitfieldInsert<uint32_t>(6u, light_type::INVALID_ID, 16, 16)
#ifdef SPHERE_LIGHT
    ,bit_cast<uint32_t>(-1.5), bit_cast<uint32_t>(1.5), bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.09), glsl::bitfieldInsert<uint32_t>(bxdfnode_type::INVALID_ID, 0u, 16, 16)
#endif
};

static const uint32_t aliased_triangles[] = {
    bit_cast<uint32_t>(-18.0), bit_cast<uint32_t>(3.5), bit_cast<uint32_t>(3.0),
    bit_cast<uint32_t>(-12.0), bit_cast<uint32_t>(3.5), bit_cast<uint32_t>(0.0),
    bit_cast<uint32_t>(-15.0), bit_cast<uint32_t>(8.0), bit_cast<uint32_t>(-3.0),
    glsl::bitfieldInsert<uint32_t>(bxdfnode_type::INVALID_ID, 0u, 16, 16)
};

static const uint32_t aliased_rectangles[] = {
    bit_cast<uint32_t>(-3.8), bit_cast<uint32_t>(0.35), bit_cast<uint32_t>(1.3),
    bit_cast<uint32_t>(6.261), bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(-3.1305),
    bit_cast<uint32_t>(0.0298), bit_cast<uint32_t>(-0.745), bit_cast<uint32_t>(0.0596),
    glsl::bitfieldInsert<uint32_t>(bxdfnode_type::INVALID_ID, 0u, 16, 16)
};

static const uint32_t aliased_lights[] = {
    bit_cast<uint32_t>(30.0), bit_cast<uint32_t>(25.0), bit_cast<uint32_t>(15.0),
#ifdef SPHERE_LIGHT
    8u,
#else
    0u,
#endif
    2u, LIGHT_TYPE
};

static const uint32_t aliased_bxdfs[] = {
    // bxdf 0
    bit_cast<uint32_t>(0.8), bit_cast<uint32_t>(0.8), bit_cast<uint32_t>(0.8),
    0u,
    0u, bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0),
    bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0),
    bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0),
    bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0),
    bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0),

    // bxdf 1
    bit_cast<uint32_t>(0.8), bit_cast<uint32_t>(0.4), bit_cast<uint32_t>(0.4),
    0u,
    0u, bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0),
    bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0),
    bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0),
    bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0),
    bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0),

    // bxdf 2
    bit_cast<uint32_t>(0.4), bit_cast<uint32_t>(0.8), bit_cast<uint32_t>(0.4),
    0u,
    0u, bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0),
    bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0),
    bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0),
    bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0),
    bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0),

    // bxdf 3
    bit_cast<uint32_t>(1.0), bit_cast<uint32_t>(1.0), bit_cast<uint32_t>(1.0),
    1u,
    0u, bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0),
    bit_cast<uint32_t>(1.02), bit_cast<uint32_t>(1.02), bit_cast<uint32_t>(1.3),
    bit_cast<uint32_t>(1.0), bit_cast<uint32_t>(1.0), bit_cast<uint32_t>(2.0),
    bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0),
    bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0),

    // bxdf 4
    bit_cast<uint32_t>(1.0), bit_cast<uint32_t>(1.0), bit_cast<uint32_t>(1.0),
    1u,
    0u, bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0),
    bit_cast<uint32_t>(1.02), bit_cast<uint32_t>(1.3), bit_cast<uint32_t>(1.02),
    bit_cast<uint32_t>(1.0), bit_cast<uint32_t>(2.0), bit_cast<uint32_t>(1.0),
    bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0),
    bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0),

    // bxdf 5
    bit_cast<uint32_t>(1.0), bit_cast<uint32_t>(1.0), bit_cast<uint32_t>(1.0),
    1u,
    0u, bit_cast<uint32_t>(0.15), bit_cast<uint32_t>(0.15),
    bit_cast<uint32_t>(1.02), bit_cast<uint32_t>(1.3), bit_cast<uint32_t>(1.02),
    bit_cast<uint32_t>(1.0), bit_cast<uint32_t>(2.0), bit_cast<uint32_t>(1.0),
    bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0),
    bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0),

    // bxdf 6
    bit_cast<uint32_t>(1.0), bit_cast<uint32_t>(1.0), bit_cast<uint32_t>(1.0),
    2u,
    0u, bit_cast<uint32_t>(0.0625), bit_cast<uint32_t>(0.0625),
    bit_cast<uint32_t>(1.0), bit_cast<uint32_t>(1.0), bit_cast<uint32_t>(1.0),
    bit_cast<uint32_t>(0.71), bit_cast<uint32_t>(0.69), bit_cast<uint32_t>(0.67),
    bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0),
    bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0), bit_cast<uint32_t>(0.0)
};
