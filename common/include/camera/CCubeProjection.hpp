#ifndef _NBL_CCUBE_PROJECTION_HPP_
#define _NBL_CCUBE_PROJECTION_HPP_

#include "ILinearProjection.hpp"

namespace nbl::hlsl
{

template<ProjectionMatrix T>
struct CCubeProjectionConstraints
{
    using matrix_t = typename T;
    using projection_t = typename IProjection<typename matrix_t>;
    using projection_range_t = std::array<typename core::smart_refctd_ptr<projection_t>, 6u>;
    using base_t = ILinearProjection<typename projection_range_t>;
};

//! Class providing linear cube projections with projection matrix per face of a cube, each projection matrix represents a single view-port
template<ProjectionMatrix T = float64_t4x4>
class CCubeProjection : public CCubeProjectionConstraints<T>::base_t
{
public:
    using constraints_t = CCubeProjectionConstraints<T>;
    using base_t = typename constraints_t::base_t;

    CCubeProjection(constraints_t::projection_range_t&& projections = {}) : base_t(std::move(projections)) {}

    constraints_t::projection_range_t& getCubeFaceProjections()
    {
        return base_t::getViewportProjections();
    }
};

} // nbl::hlsl namespace

#endif // _NBL_CCUBE_PROJECTION_HPP_