// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#version 430 core

layout (location = 0) in vec3 Pos;
layout (location = 3) in vec3 Normal;

layout (location = 0) out vec3 outPos;

void main()
{
    outPos = Pos;
    gl_Position = vec4(Pos, 1.0);
}