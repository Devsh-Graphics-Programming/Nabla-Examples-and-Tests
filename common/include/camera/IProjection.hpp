#ifndef _NBL_I_PROJECTION_HPP_
#define _NBL_I_PROJECTION_HPP_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>

namespace nbl::core
{

/// @brief Interface class for any type of projection
class IProjection
{
public:
    /// @brief underlying type for all vectors we project or un-project (inverse), projections *may* transform vectors in less dimensions
    using projection_vector_t = hlsl::float64_t4;

    enum class ProjectionType
    {
        /// @brief Any raw linear transformation, for example it may represent Perspective, Orthographic, Oblique, Axonometric, Shear projections
        Linear,

        /// @brief Specialized linear projection for planar projections with parameters
        Planar,

        /// @brief Extension of planar projection represented by pre-transform & planar transform combined projecting onto R3 cave quad
        CaveQuad,

        /// @brief Specialized CaveQuad projection, represents planar projections onto cube with 6 quad cube faces
        Cube,

        Spherical,
        ThinLens,
        
        Count
    };
    
    IProjection() = default;
    virtual ~IProjection() = default;

    /// @brief Transform a vector from its input space into projection space.
    ///
    /// @param vecToProjectionSpace Vector to transform into projection space.
    /// @param output Result vector in projection space.
    virtual void project(const projection_vector_t& vecToProjectionSpace, projection_vector_t& output) const = 0;

    /// @brief Transform a vector from projection space back to the original space.
    ///
    /// The inverse transform may fail because the original projection may be singular.
    ///
    /// @param vecFromProjectionSpace Vector in projection space.
    /// @param output Result vector in the original space.
    /// @return `true` when the inverse transform succeeded, otherwise `false`.
    virtual bool unproject(const projection_vector_t& vecFromProjectionSpace, projection_vector_t& output) const = 0;

    /// @brief Return the specific type of the projection.
    ///
    /// Examples include linear, spherical, and thin-lens projections as defined
    /// by `ProjectionType`.
    ///
    /// @return The type of this projection.
    virtual ProjectionType getProjectionType() const = 0;
};

} // namespace nbl::core

#endif // _NBL_IPROJECTION_HPP_
