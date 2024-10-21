#ifndef _NBL_ILINEAR_PROJECTION_HPP_
#define _NBL_ILINEAR_PROJECTION_HPP_

#include "IProjection.hpp"

namespace nbl::hlsl
{

//! Interface class for linear projections range storage - a projection matrix represents single view-port
template<typename T>
class ILinearProjection : protected IProjectionRange<T>
{
public:
    using base_t = typename IProjectionRange<T>;
    using range_t = typename base_t::range_t;
    using projection_t = typename base_t::projection_t;

    ILinearProjection(range_t&& projections) : base_t(projections) {}

protected:
    inline range_t& getViewportProjections()
    {
        return base_t::m_projectionRange;
    }
};

} // nbl::hlsl namespace

#endif // _NBL_ILINEAR_PROJECTION_HPP_