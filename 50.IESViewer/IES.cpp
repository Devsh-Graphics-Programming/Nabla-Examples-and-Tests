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

const char* IES::typeToRS(CIESProfile::properties_t::PhotometricType type)
{
    switch (type)
    {
    case asset::CIESProfile::properties_t::TYPE_C:
        return "TYPE_C";
    case asset::CIESProfile::properties_t::TYPE_B:
        return "TYPE_B";
    case asset::CIESProfile::properties_t::TYPE_A:
        return "TYPE_A";
    case asset::CIESProfile::properties_t::TYPE_NONE:
    default:
        return "TYPE_NONE";
    }
}

const char* IES::versionToRS(CIESProfile::properties_t::Version version)
{
    switch (version)
    {
    case asset::CIESProfile::properties_t::V_1995:
        return "V_1995";
    case asset::CIESProfile::properties_t::V_2002:
        return "V_2002";
    default:
        return "V_UNKNOWN";
    }
}
