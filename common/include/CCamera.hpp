// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _CAMERA_IMPL_
#define _CAMERA_IMPL_

#include <nabla.h>
#include <iostream>
#include <cstdio>
#include <fstream>
#include <chrono>

#include "camera/ICameraControl.hpp"

// FPS Camera, we will have more types soon

template<ProjectionMatrix T = float64_t4x4>
class Camera : public ICameraController<typename T>
{ 
public:
	using matrix_t = T;

	Camera() = default;
	~Camera() = default;

	/*
		TODO: controller + gimbal to do all of this -> override virtual manipulate method
	*/

public:

	void mouseProcess(const nbl::ui::IMouseEventChannel::range_t& events)
	{
		for (auto eventIt=events.begin(); eventIt!=events.end(); eventIt++)
		{
			auto ev = *eventIt;

			if(ev.type == nbl::ui::SMouseEvent::EET_CLICK && ev.clickEvent.mouseButton == nbl::ui::EMB_LEFT_BUTTON)
				if(ev.clickEvent.action == nbl::ui::SMouseEvent::SClickEvent::EA_PRESSED) 
					mouseDown = true;
				else if (ev.clickEvent.action == nbl::ui::SMouseEvent::SClickEvent::EA_RELEASED)
					mouseDown = false;

			if(ev.type == nbl::ui::SMouseEvent::EET_MOVEMENT && mouseDown) 
			{
				nbl::core::vectorSIMDf pos = getPosition();
				nbl::core::vectorSIMDf localTarget = getTarget() - pos;

				// Get Relative Rotation for localTarget in Radians
				float relativeRotationX, relativeRotationY;
				relativeRotationY = atan2(localTarget.X, localTarget.Z);
				const double z1 = nbl::core::sqrt(localTarget.X*localTarget.X + localTarget.Z*localTarget.Z);
				relativeRotationX = atan2(z1, localTarget.Y) - nbl::core::PI<float>()/2;
				
				constexpr float RotateSpeedScale = 0.003f; 
				relativeRotationX -= ev.movementEvent.relativeMovementY * rotateSpeed * RotateSpeedScale * -1.0f;
				float tmpYRot = ev.movementEvent.relativeMovementX * rotateSpeed * RotateSpeedScale * -1.0f;

				if (leftHanded)
					relativeRotationY -= tmpYRot;
				else
					relativeRotationY += tmpYRot;

				const double MaxVerticalAngle = nbl::core::radians<float>(88.0f);

				if (relativeRotationX > MaxVerticalAngle*2 && relativeRotationX < 2 * nbl::core::PI<float>()-MaxVerticalAngle)
					relativeRotationX = 2 * nbl::core::PI<float>()-MaxVerticalAngle;
				else
					if (relativeRotationX > MaxVerticalAngle && relativeRotationX < 2 * nbl::core::PI<float>()-MaxVerticalAngle)
						relativeRotationX = MaxVerticalAngle;

				localTarget.set(0,0, nbl::core::max(1.f, nbl::core::length(pos)[0]), 1.f);

				nbl::core::matrix3x4SIMD mat;
				mat.setRotation(nbl::core::quaternion(relativeRotationX, relativeRotationY, 0));
				mat.transformVect(localTarget);
				
				setTarget(localTarget + pos);
			}
		}
	}

	void keyboardProcess(const nbl::ui::IKeyboardEventChannel::range_t& events)
	{
		for(uint32_t k = 0; k < E_CAMERA_MOVE_KEYS::ECMK_COUNT; ++k)
			perActionDt[k] = 0.0;

		/*
		* If a Key was already being held down from previous frames
		* Compute with this assumption that the key will be held down for this whole frame as well,
		* And If an UP event was sent It will get subtracted it from this value. (Currently Disabled Because we Need better Oracle)
		*/

		for(uint32_t k = 0; k < E_CAMERA_MOVE_KEYS::ECMK_COUNT; ++k) 
			if(keysDown[k]) 
			{
				auto timeDiff = std::chrono::duration_cast<std::chrono::milliseconds>(nextPresentationTimeStamp - lastVirtualUpTimeStamp).count();
				assert(timeDiff >= 0);
				perActionDt[k] += timeDiff;
			}

		for (auto eventIt=events.begin(); eventIt!=events.end(); eventIt++)
		{
			const auto ev = *eventIt;
			
			// accumulate the periods for which a key was down
			const auto timeDiff = std::chrono::duration_cast<std::chrono::milliseconds>(nextPresentationTimeStamp - ev.timeStamp).count();
			assert(timeDiff >= 0);

			// handle camera movement
			for (const auto logicalKey : { ECMK_MOVE_FORWARD, ECMK_MOVE_BACKWARD, ECMK_MOVE_LEFT, ECMK_MOVE_RIGHT })
			{
				const auto code = keysMap[logicalKey];

				if (ev.keyCode == code)
				{
					if (ev.action == nbl::ui::SKeyboardEvent::ECA_PRESSED && !keysDown[logicalKey]) 
					{
						perActionDt[logicalKey] += timeDiff;
						keysDown[logicalKey] = true;
					}
					else if (ev.action == nbl::ui::SKeyboardEvent::ECA_RELEASED) 
					{
						// perActionDt[logicalKey] -= timeDiff; 
						keysDown[logicalKey] = false;
					}
				}
			}

			// handle reset to default state
			if (ev.keyCode == nbl::ui::EKC_HOME)
				if (ev.action == nbl::ui::SKeyboardEvent::ECA_RELEASED)
				{
					position = initialPosition;
					target = initialTarget;
					recomputeViewMatrix();
				}
		}
	}

	void beginInputProcessing(std::chrono::microseconds _nextPresentationTimeStamp)
	{
		nextPresentationTimeStamp = _nextPresentationTimeStamp;
		return;
	}
	
	void endInputProcessing(std::chrono::microseconds _nextPresentationTimeStamp)
	{
		nbl::core::vectorSIMDf pos = getPosition();
		nbl::core::vectorSIMDf localTarget = getTarget() - pos;

		if (!firstUpdate)
		{
			nbl::core::vectorSIMDf movedir = localTarget;
			movedir.makeSafe3D();
			movedir = nbl::core::normalize(movedir);

			constexpr float MoveSpeedScale = 0.02f; 

			pos += movedir * perActionDt[E_CAMERA_MOVE_KEYS::ECMK_MOVE_FORWARD] * moveSpeed * MoveSpeedScale;
			pos -= movedir * perActionDt[E_CAMERA_MOVE_KEYS::ECMK_MOVE_BACKWARD] * moveSpeed * MoveSpeedScale;

			// strafing
		
			// if upvector and vector to the target are the same, we have a
			// problem. so solve this problem:
			nbl::core::vectorSIMDf up = nbl::core::normalize(upVector);
			nbl::core::vectorSIMDf cross = nbl::core::cross(localTarget, up);
			bool upVectorNeedsChange = nbl::core::lengthsquared(cross)[0] == 0;
			if (upVectorNeedsChange)
			{
				up = nbl::core::normalize(backupUpVector);
			}

			nbl::core::vectorSIMDf strafevect = localTarget;
			if (leftHanded)
				strafevect = nbl::core::cross(strafevect, up);
			else
				strafevect = nbl::core::cross(up, strafevect);

			strafevect = nbl::core::normalize(strafevect);

			pos += strafevect * perActionDt[E_CAMERA_MOVE_KEYS::ECMK_MOVE_LEFT] * moveSpeed * MoveSpeedScale;
			pos -= strafevect * perActionDt[E_CAMERA_MOVE_KEYS::ECMK_MOVE_RIGHT] * moveSpeed * MoveSpeedScale;
		}
		else
			firstUpdate = false;

		setPosition(pos);
		setTarget(localTarget+pos);

		lastVirtualUpTimeStamp = nextPresentationTimeStamp;
	}

private:

	inline void initDefaultKeysMap() { mapKeysToWASD(); }
	
	inline void allKeysUp() 
	{
		for (uint32_t i=0; i< E_CAMERA_MOVE_KEYS::ECMK_COUNT; ++i)
			keysDown[i] = false;

		mouseDown = false;
	}
};

#endif // _CAMERA_IMPL_