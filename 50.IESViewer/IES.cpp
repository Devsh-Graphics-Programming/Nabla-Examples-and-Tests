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

video::IGPUImage* IES::getActiveImage(E_MODE mode) const
{
    switch (mode)
    {
    case EM_OCTAHEDRAL_MAP:
        return views.candelaOctahedralMap->getCreationParameters().image.get();

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
    case IES::EM_OCTAHEDRAL_MAP:
        return "Candela Octahedral Map";
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