#ifndef _NBL_THIS_EXAMPLE_COMMON_H_INCLUDED_
#define _NBL_THIS_EXAMPLE_COMMON_H_INCLUDED_

#include "nbl/examples/examples.hpp"

using namespace nbl;
using namespace nbl::core;
using namespace nbl::hlsl;
using namespace nbl::system;
using namespace nbl::asset;
using namespace nbl::ui;
using namespace nbl::video;
using namespace nbl::application_templates;
using namespace nbl::examples;

#include "app_resources/common.hlsl"

namespace nbl::scene
{

using PolygonGeometryData = core::smart_refctd_ptr<ICPUPolygonGeometry>;
using GeometryCollectionData = core::smart_refctd_ptr<ICPUGeometryCollection>;
using GeometryData = std::variant<PolygonGeometryData, GeometryCollectionData>;
struct ReferenceObjectCpu
{
  core::matrix3x4SIMD transform;
	GeometryData data;
  uint32_t instanceID;
};

}


#endif // _NBL_THIS_EXAMPLE_COMMON_H_INCLUDED_