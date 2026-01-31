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
