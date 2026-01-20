#ifndef _THIS_EXAMPLE_APP_INPUT_PARSER_HPP_
#define _THIS_EXAMPLE_APP_INPUT_PARSER_HPP_

// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/examples/examples.hpp"

struct AppInputParser
{
public:
    struct Output 
    {
        std::vector<std::string> inputList;
        bool withGUI;
        bool writeAssets;
    };

    AppInputParser(nbl::system::logger_opt_ptr _logger = nullptr) : logger(_logger) {}
    bool parse(Output& out, const std::string jFilePath, const std::string cwd = ".");

private:
    nbl::system::logger_opt_ptr logger;
};

#endif // _THIS_EXAMPLE_APP_INPUT_PARSER_HPP_