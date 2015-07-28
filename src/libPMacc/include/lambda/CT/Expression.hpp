/**
 * Copyright 2013 Heiko Burau, Rene Widera
 *
 * This file is part of libPMacc.
 *
 * libPMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * libPMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with libPMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "../Expression.hpp"

namespace PMacc
{
namespace lambda
{
namespace CT
{

template<typename Expr, int _terminalTypeIdx>
struct Expression;

template<typename Child0, int _terminalTypeIdx>
struct Expression<lambda::Expression<exprTypes::terminal, mpl::vector<Child0> >, _terminalTypeIdx>
{
    static const int nextTerminalTypeIdx = _terminalTypeIdx + 1;
};

template<int I, int _terminalTypeIdx>
struct Expression<lambda::Expression<exprTypes::terminal, mpl::vector<placeholder<I> > >, _terminalTypeIdx>
{
    static const int nextTerminalTypeIdx = _terminalTypeIdx;
};

template<int I, int _terminalTypeIdx>
struct Expression<lambda::Expression<exprTypes::terminal, mpl::vector<mpl::int_<I> > >, _terminalTypeIdx>
{
    static const int nextTerminalTypeIdx = _terminalTypeIdx;
};

template<typename _Child0, typename _Child1, int _terminalTypeIdx>
struct Expression<lambda::Expression<exprTypes::assign, mpl::vector<_Child0, _Child1> >, _terminalTypeIdx>
{
    typedef CT::Expression<_Child0, _terminalTypeIdx> Child0;
    typedef CT::Expression<_Child1, Child0::nextTerminalTypeIdx> Child1;
    static const int nextTerminalTypeIdx = Child1::nextTerminalTypeIdx;
};

template<typename _Child0, typename _Child1, int _terminalTypeIdx>
struct Expression<lambda::Expression<exprTypes::plus, mpl::vector<_Child0, _Child1> >, _terminalTypeIdx>
{
    typedef CT::Expression<_Child0, _terminalTypeIdx> Child0;
    typedef CT::Expression<_Child1, Child0::nextTerminalTypeIdx> Child1;
    static const int nextTerminalTypeIdx = Child1::nextTerminalTypeIdx;
};

template<typename _Child0, typename _Child1, int _terminalTypeIdx>
struct Expression<lambda::Expression<exprTypes::minus, mpl::vector<_Child0, _Child1> >, _terminalTypeIdx>
{
    typedef CT::Expression<_Child0, _terminalTypeIdx> Child0;
    typedef CT::Expression<_Child1, Child0::nextTerminalTypeIdx> Child1;
    static const int nextTerminalTypeIdx = Child1::nextTerminalTypeIdx;
};

template<typename _Child0, typename _Child1, int _terminalTypeIdx>
struct Expression<lambda::Expression<exprTypes::multiply, mpl::vector<_Child0, _Child1> >, _terminalTypeIdx>
{
    typedef CT::Expression<_Child0, _terminalTypeIdx> Child0;
    typedef CT::Expression<_Child1, Child0::nextTerminalTypeIdx> Child1;
    static const int nextTerminalTypeIdx = Child1::nextTerminalTypeIdx;
};

template<typename _Child0, typename _Child1, int _terminalTypeIdx>
struct Expression<lambda::Expression<exprTypes::divide, mpl::vector<_Child0, _Child1> >, _terminalTypeIdx>
{
    typedef CT::Expression<_Child0, _terminalTypeIdx> Child0;
    typedef CT::Expression<_Child1, Child0::nextTerminalTypeIdx> Child1;
    static const int nextTerminalTypeIdx = Child1::nextTerminalTypeIdx;
};

template<typename _Child0, typename _Child1, int _terminalTypeIdx>
struct Expression<lambda::Expression<exprTypes::comma, mpl::vector<_Child0, _Child1> >, _terminalTypeIdx>
{
    typedef CT::Expression<_Child0, _terminalTypeIdx> Child0;
    typedef CT::Expression<_Child1, Child0::nextTerminalTypeIdx> Child1;
    static const int nextTerminalTypeIdx = Child1::nextTerminalTypeIdx;
};

template<typename _Child0, typename _Child1, int _terminalTypeIdx>
struct Expression<lambda::Expression<exprTypes::call, mpl::vector<_Child0, _Child1> >, _terminalTypeIdx>
{
    typedef CT::Expression<_Child0, _terminalTypeIdx> Child0;
    typedef CT::Expression<_Child1, Child0::nextTerminalTypeIdx> Child1;
    static const int nextTerminalTypeIdx = Child1::nextTerminalTypeIdx;
};

template<typename _Child0, typename _Child1, typename _Child2, int _terminalTypeIdx>
struct Expression<lambda::Expression<exprTypes::call, mpl::vector<_Child0, _Child1, _Child2> >, _terminalTypeIdx>
{
    typedef CT::Expression<_Child0, _terminalTypeIdx> Child0;
    typedef CT::Expression<_Child1, Child0::nextTerminalTypeIdx> Child1;
    typedef CT::Expression<_Child2, Child1::nextTerminalTypeIdx> Child2;
    static const int nextTerminalTypeIdx = Child2::nextTerminalTypeIdx;
};

template<typename _Child0, typename _Child1, typename _Child2, typename _Child3, int _terminalTypeIdx>
struct Expression<lambda::Expression<exprTypes::call, mpl::vector<_Child0, _Child1, _Child2, _Child3> >, _terminalTypeIdx>
{
    typedef CT::Expression<_Child0, _terminalTypeIdx> Child0;
    typedef CT::Expression<_Child1, Child0::nextTerminalTypeIdx> Child1;
    typedef CT::Expression<_Child2, Child1::nextTerminalTypeIdx> Child2;
    typedef CT::Expression<_Child3, Child2::nextTerminalTypeIdx> Child3;
    static const int nextTerminalTypeIdx = Child3::nextTerminalTypeIdx;
};

template<typename _Child0, typename _Child1, typename _Child2, typename _Child3, typename _Child4, int _terminalTypeIdx>
struct Expression<lambda::Expression<exprTypes::call, mpl::vector<_Child0, _Child1, _Child2, _Child3, _Child4> >, _terminalTypeIdx>
{
    typedef CT::Expression<_Child0, _terminalTypeIdx> Child0;
    typedef CT::Expression<_Child1, Child0::nextTerminalTypeIdx> Child1;
    typedef CT::Expression<_Child2, Child1::nextTerminalTypeIdx> Child2;
    typedef CT::Expression<_Child3, Child2::nextTerminalTypeIdx> Child3;
    typedef CT::Expression<_Child4, Child3::nextTerminalTypeIdx> Child4;
    static const int nextTerminalTypeIdx = Child4::nextTerminalTypeIdx;
};

template<typename _Child0, typename _Child1, typename _Child2, typename _Child3, typename _Child4, typename _Child5, int _terminalTypeIdx>
struct Expression<lambda::Expression<exprTypes::call, mpl::vector<_Child0, _Child1, _Child2, _Child3, _Child4, _Child5> >, _terminalTypeIdx>
{
    typedef CT::Expression<_Child0, _terminalTypeIdx> Child0;
    typedef CT::Expression<_Child1, Child0::nextTerminalTypeIdx> Child1;
    typedef CT::Expression<_Child2, Child1::nextTerminalTypeIdx> Child2;
    typedef CT::Expression<_Child3, Child2::nextTerminalTypeIdx> Child3;
    typedef CT::Expression<_Child4, Child3::nextTerminalTypeIdx> Child4;
    typedef CT::Expression<_Child5, Child4::nextTerminalTypeIdx> Child5;
    static const int nextTerminalTypeIdx = Child5::nextTerminalTypeIdx;
};

template<typename _Child0, typename _Child1, int _terminalTypeIdx>
struct Expression<lambda::Expression<exprTypes::subscript, mpl::vector<_Child0, _Child1> >, _terminalTypeIdx>
{
    typedef CT::Expression<_Child0, _terminalTypeIdx> Child0;
    typedef CT::Expression<_Child1, Child0::nextTerminalTypeIdx> Child1;
    static const int nextTerminalTypeIdx = Child1::nextTerminalTypeIdx;
};

} // CT
} // lambda
} // PMacc
