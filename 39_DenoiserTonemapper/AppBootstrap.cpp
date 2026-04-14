// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "AppBootstrap.hpp"

#include <iostream>

using namespace nbl;

nbl::core::vector<std::string> getInputArguments(int argc, char* argv[])
{
	core::vector<std::string> arguments;
	arguments.reserve(argc > 0 ? argc : 1);
	arguments.emplace_back(argv[0]);

	if (argc>1)
	{
		std::cout << "Guess input from Commandline arguments" << std::endl;
		for (auto i = 1; i < argc; ++i)
			arguments.emplace_back(argv[i]);
	}
	else
	{
		std::cout << "No arguments provided, running demo mode from ../exampleInputArguments.txt" << std::endl;
		arguments.emplace_back("-batch");
		arguments.emplace_back("../exampleInputArguments.txt");
	}

	return arguments;
}
