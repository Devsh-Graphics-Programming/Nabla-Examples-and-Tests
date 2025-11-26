// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "IES.hpp"

const asset::CIESProfile* IES::getProfile() const
{
    auto* meta = bundle.getMetadata();
    if (meta)
        return &meta->selfCast<const asset::CIESProfileMetadata>()->profile;

    return nullptr;
}

video::IGPUImage* IES::getActiveImage() const
{
    switch (mode)
    {
    case EM_IES_C:
        return views.candela->getCreationParameters().image.get();
    case EM_SPERICAL_C:
        return views.spherical->getCreationParameters().image.get();
    case EM_DIRECTION:
        return views.direction->getCreationParameters().image.get();
    case EM_PASS_T_MASK:
        return views.mask->getCreationParameters().image.get();

    case EM_CDC:
    default:
        return nullptr;
    }
}

const char* IES::modeToRS(E_MODE mode)
{
    switch (mode)
    {
    case IES::EM_CDC:
        return "Candlepower Distribution Curve";
    case IES::EM_IES_C:
        return "Sample IES Candela";
    case IES::EM_SPERICAL_C:
        return "Sample Spherical Coordinates";
    case IES::EM_DIRECTION:
        return "Sample Direction";
    case IES::EM_PASS_T_MASK:
        return "Sample Pass Mask";
    default:
        return "ERROR (mode)";
    }
}

const char* IES::symmetryToRS(CIESProfile::properties_t::LuminairePlanesSymmetry symmetry)
{
    switch (symmetry)
    {
    case asset::CIESProfile::properties_t::ISOTROPIC:
        return "ISOTROPIC";
    case asset::CIESProfile::properties_t::QUAD_SYMETRIC:
        return "QUAD_SYMETRIC";
    case asset::CIESProfile::properties_t::HALF_SYMETRIC:
        return "HALF_SYMETRIC";
    case asset::CIESProfile::properties_t::OTHER_HALF_SYMMETRIC:
        return "OTHER_HALF_SYMMETRIC";
    case asset::CIESProfile::properties_t::NO_LATERAL_SYMMET:
        return "NO_LATERAL_SYMMET";
    default:
        return "ERROR (symmetry)";
    }
}