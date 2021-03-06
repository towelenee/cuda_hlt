#pragma once

#include <vector>
#include <iostream>
#include <stdint.h>
#include <stdio.h>

#include "SystemOfUnits.h"

/**
 * Generic StrException launcher
 */
class StrException : public std::exception
{
public:
    std::string s;
    StrException(std::string ss) : s(ss) {}
    ~StrException() throw () {} // Updated
    const char* what() const throw() { return s.c_str(); }
};
