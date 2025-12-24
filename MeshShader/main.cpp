// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "common.hpp"
#include "SampleApp.h"

/* 
Renders scene texture to an offscreen framebuffer whose color attachment is then sampled into a imgui window.

Written with Nabla's UI extension and got integrated with ImGuizmo to handle scene's object translations.
*/
int main(int argc, char** argv) {
	//expanded macro for easier IDE peeking
	return MeshSampleApp::main<MeshSampleApp>(argc, argv);
}