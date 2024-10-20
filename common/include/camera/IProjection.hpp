#ifndef _NBL_IPROJECTION_HPP_
#define _NBL_IPROJECTION_HPP_

#include <nbl/builtin/hlsl/cpp_compat/matrix.hlsl>

namespace nbl::hlsl
{
//! Interface class for projection matrices range storage
template<std::ranges::range Range = std::array<float64_t4x4, 1u>>
requires nbl::is_any_of_v<std::ranges::range_value_t<Range>, float64_t4x4, float32_t4x4>
class IProjection
{
public:
    using value_t = std::ranges::iterator_t<Range>;
    using range_t = Range;

protected:
    IProjection(const range_t& matrices) : m_projMatrices(matrices) {}
    range_t m_projMatrices;
};

} // nbl::hlsl namespace

#endif // _NBL_IPROJECTION_HPP_