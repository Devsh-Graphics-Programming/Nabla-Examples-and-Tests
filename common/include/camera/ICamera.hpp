// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _I_CAMERA_HPP_
#define _I_CAMERA_HPP_

#include "camera/IGimbalController.hpp"

namespace nbl::hlsl // TODO: DIFFERENT NAMESPACE
{

class ICamera : public IGimbalController, virtual public core::IReferenceCounted
{ 
public:
    using IGimbalController::IGimbalController;

    //! Manipulation mode for virtual events
    //! TODO: this should belong to IObjectTransform or something
    enum ManipulationMode
    {
        // Interpret virtual events as accumulated impulse representing relative manipulation with respect to view gimbal base 
        Local,

        // Interpret virtual events as accumulated impulse representing relative manipulation with respect to world base
        World
    };

    // Gimbal with view parameters representing a camera in world space
    class CGimbal : public IGimbal<float64_t>
    {
    public:
        using base_t = IGimbal<float64_t>;

        CGimbal(typename base_t::SCreationParameters&& parameters) : base_t(std::move(parameters)) { updateView(); }
        ~CGimbal() = default;

        inline void updateView()
        {            
            const auto& gRight = base_t::getXAxis(), gUp = base_t::getYAxis(), gForward = base_t::getZAxis();

            auto isNormalized = [](const auto& v, precision_t epsilon) -> bool
            {
                return glm::epsilonEqual(glm::length(v), 1.0, epsilon);
            };

            auto isOrthogonal = [](const auto& a, const auto& b, precision_t epsilon) -> bool
            {
                return glm::epsilonEqual(glm::dot(a, b), 0.0, epsilon);
            };

            auto isOrthoBase = [&](const auto& x, const auto& y, const auto& z, precision_t epsilon = 1e-6) -> bool
            {
                return isNormalized(x, epsilon) && isNormalized(y, epsilon) && isNormalized(z, epsilon) &&
                    isOrthogonal(x, y, epsilon) && isOrthogonal(x, z, epsilon) && isOrthogonal(y, z, epsilon);
            };

            assert(isOrthoBase(gRight, gUp, gForward));

            const auto& position = base_t::getPosition();

            m_viewMatrix[0u] = float64_t4(gRight, -glm::dot(gRight, position));
            m_viewMatrix[1u] = float64_t4(gUp, -glm::dot(gUp, position));
            m_viewMatrix[2u] = float64_t4(gForward, -glm::dot(gForward, position));
        }

        inline const float64_t3x4& getViewMatrix() const { return m_viewMatrix; }

    private:
        float64_t3x4 m_viewMatrix;
    };

    ICamera() {}
	~ICamera() = default;

	// Returns a gimbal which *models the camera view*
	virtual const CGimbal& getGimbal() = 0u;

    // Manipulates camera with virtual events, returns true if *any* manipulation happens, it may fail partially or fully because each camera type has certain constraints which determine how it actually works
    // TODO: this really needs to be moved to more abstract interface, eg. IObjectTransform or something and ICamera should inherit it (its also an object!)
    virtual bool manipulate(std::span<const CVirtualGimbalEvent> virtualEvents, ManipulationMode mode) = 0; 

	// VirtualEventType bitmask for a camera view gimbal manipulation requests filtering
	virtual const uint32_t getAllowedVirtualEvents(ManipulationMode mode) = 0u;

    // Identifier of a camera type
    virtual const std::string_view getIdentifier() = 0u;
};

}

#endif // _I_CAMERA_HPP_