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
	virtual ~ICamera() = default;

	// Returns a gimbal which *models the camera view*
	virtual const CGimbal& getGimbal() = 0u;

    // Manipulates camera with virtual events, returns true if *any* manipulation happens, it may fail partially or fully because each camera type has certain constraints which determine how it actually works
    // TODO: this really needs to be moved to more abstract interface, eg. IObjectTransform or something and ICamera should inherit it (its also an object!)
    virtual bool manipulate(std::span<const CVirtualGimbalEvent> virtualEvents, const float64_t4x4 const* referenceFrame = nullptr) = 0;

	// VirtualEventType bitmask for a camera view gimbal manipulation requests filtering
	virtual const uint32_t getAllowedVirtualEvents() = 0u;

    // Identifier of a camera type
    virtual const std::string_view getIdentifier() = 0u;

    // (***)
    inline void setMoveSpeedScale(double scalar)
    {
        m_moveSpeedScale = scalar;
    }

    // (***)
    inline void setRotationSpeedScale(double scalar)
    {
        m_rotationSpeedScale = scalar;
    }

    inline double getMoveSpeedScale() const { return m_moveSpeedScale; }
    inline double getRotationSpeedScale() const { return m_rotationSpeedScale; }

protected:
    
    // (***) TODO: I need to think whether a camera should own this or controllers should be able 
    // to set sensitivity to scale magnitudes of generated events we put into manipulate method
    double m_moveSpeedScale = 0.01, m_rotationSpeedScale = 0.003;
};

}

#endif // _I_CAMERA_HPP_