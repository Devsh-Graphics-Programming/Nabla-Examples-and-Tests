#ifndef _NBL_CGENERAL_PURPOSE_GIMBAL_HPP_
#define _NBL_CGENERAL_PURPOSE_GIMBAL_HPP_

#include "IGimbal.hpp"

// TODO: DIFFERENT NAMESPACE
namespace nbl::hlsl
{
    template<typename T = float64_t>
    class CGeneralPurposeGimbal : public IGimbal<T>
    {
    public:
        using base_t = IGimbal<T>;

        CGeneralPurposeGimbal(typename base_t::SCreationParameters&& parameters) : base_t(std::move(parameters)) {}
        ~CGeneralPurposeGimbal() = default;
    };
}

#endif // _NBL_IGIMBAL_HPP_