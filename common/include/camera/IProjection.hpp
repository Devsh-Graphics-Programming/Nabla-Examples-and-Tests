#ifndef _NBL_I_PROJECTION_HPP_
#define _NBL_I_PROJECTION_HPP_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>

namespace nbl::hlsl
{

//! Interface class for any type of projection
class IProjection
{
public:
    //! underlying type for all vectors we project or un-project (inverse), projections *may* transform vectors in less dimensions
    using projection_vector_t = float64_t4;

    enum class ProjectionType
    {
        //! Any raw linear transformation, for example it may represent Perspective, Orthographic, Oblique, Axonometric, Shear projections
        Linear,

        //! Specialized linear projection for planar projections with parameters
        Planar,

        //! Extension of planar projection represented by pre-transform & planar transform combined projecting onto R3 cave quad
        CaveQuad,

        //! Specialized CaveQuad projection, represents planar projections onto cube with 6 quad cube faces
        Cube,

        Spherical,
        ThinLens,
        
        Count
    };
    
    IProjection() = default;
    virtual ~IProjection() = default;

    /**
    * @brief Transforms a vector from its input space into the projection space.
    *
    * @param "vecToProjectionSpace" is a vector to transform from its space into projection space.
    * @param "output" is a vector which is "vecToProjectionSpace" transformed into projection space.
    * @return void. "output" is the vector in projection space.
    */
    virtual void project(const projection_vector_t& vecToProjectionSpace, projection_vector_t& output) const = 0;

    /**
    * @brief Transforms a vector from the projection space back to the original space.
    * Note the inverse transform may fail because original projection may be singular.
    *
    * @param "vecFromProjectionSpace" is a vector in the projection space to transform back to original space.
    * @param "output" is a vector which is "vecFromProjectionSpace" transformed back to its original space.
    * @return true if inverse succeeded and then "output" is the vector in the original space. False otherwise.
    */
    virtual bool unproject(const projection_vector_t& vecFromProjectionSpace, projection_vector_t& output) const = 0;

    /**
    * @brief Returns the specific type of the projection
    * (e.g., linear, spherical, thin-lens) as defined by the
    * ProjectionType enumeration.
    *
    * @return The type of this projection.
    */
    virtual ProjectionType getProjectionType() const = 0;
};

} // namespace nbl::hlsl

#endif // _NBL_IPROJECTION_HPP_