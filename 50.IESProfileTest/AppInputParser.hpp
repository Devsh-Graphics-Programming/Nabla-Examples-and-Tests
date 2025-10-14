#ifndef _THIS_EXAMPLE_APP_INPUT_PARSER_HPP_
#define _THIS_EXAMPLE_APP_INPUT_PARSER_HPP_

// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/examples/examples.hpp"
#include "nlohmann/json.hpp"

struct AppInputParser
{
public:
    AppInputParser(nbl::system::logger_opt_ptr _logger = nullptr) : logger(_logger) {}
    bool parse(std::vector<std::string>& out, const std::string input, const std::string cwd = ".");

private:
    nbl::system::logger_opt_ptr logger;
};

#endif // _THIS_EXAMPLE_APP_INPUT_PARSER_HPP_