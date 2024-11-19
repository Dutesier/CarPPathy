/******************************************************************************
 * Project:  CarPPathy
 * Brief:    Following Andrej Karpathy courses on Neural Networks.
 *
 * This software is provided "as is," without warranty of any kind, express
 * or implied, including but not limited to the warranties of merchantability,
 * fitness for a particular purpose, and noninfringement. In no event shall
 * the authors or copyright holders be liable for any claim, damages, or
 * other liability, whether in an action of contract, tort, or otherwise,
 * arising from, out of, or in connection with the software or the use or
 * other dealings in the software.
 *
 * Author:   Dutesier
 *
 ******************************************************************************/

#include "Value.hpp"
#include <cmath>
#include <cstddef>
#include <exception>
#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace cppty
{

Value::Value(double data, const std::string& label)
{
    m_value = std::make_shared<ValueImpl>(data, label);
}

Value::Value(
    double data,
    const std::vector<std::shared_ptr<ValueImpl>>& previous,
    OperationType type,
    const std::string& label)
{
    m_value = std::make_shared<ValueImpl>(data, previous, type, label);
}

Value::Value(const ValueImpl& val)
{
    m_value = std::make_shared<ValueImpl>(val);
}

Value Value::operator+(const Value& other) const
{
    return Value(ValueImpl{ m_value->getData() + other.m_value->getData(),
                            { m_value, other.m_value },
                            OperationType::Addition,
                            m_value->getLabel() + "+" + other.m_value->getLabel() });
}

Value Value::operator-(const Value& other) const
{
    auto negatedOther = other * (-1);
    return *this + negatedOther;
}

Value Value::operator*(const Value& other) const
{
    return Value(ValueImpl{ m_value->getData() * other.m_value->getData(),
                            { m_value, other.m_value },
                            OperationType::Multiplication,
                            m_value->getLabel() + "*" + other.m_value->getLabel() });
}

Value Value::operator/(const Value& other) const
{
    auto otherPowMinusOne = other.pow(-1);
    return *this * otherPowMinusOne;
}

Value Value::pow(double i) const
{
    std::ostringstream stream;
    stream << i;
    const auto& n = m_value->getData();
    auto t = std::pow(n, i);
    auto ret = ValueImpl(t, { m_value }, Pow, m_value->getLabel() + ".pow(" + stream.str() + ")");
    ret.storePower(i);
    return ret;
}

Value Value::tanh()
{

    const auto& n = m_value->getData();
    auto t = (std::exp(2 * n) - 1) / (std::exp(2 * n) + 1);
    auto ret = ValueImpl(t, { m_value }, Tanh, m_value->getLabel() + ".tanh()");
    return ret;
}

Value Value::exp()
{
    const auto& n = m_value->getData();
    auto t = std::exp(n);
    auto ret = ValueImpl(t, { m_value }, Exp, m_value->getLabel() + ".exp()");
    return ret;
}

void Value::setLabel(const std::string& newLabel)
{
    m_value->setLabel(newLabel);
}

void Value::setGrad(double grad)
{
    m_value->setGrad(grad);
}

void Value::backpropagate()
{
    if (!m_value)
    {
        return; // Should throw or something here
    }
    m_value->setGrad(1); // Edge case
    auto topo = buildTopographic(m_value);
    for (auto i = topo.rbegin(); i != topo.rend(); i++)
    {
        if (!*i)
        {
            return;
        }
        (*i)->backpropagate();
    }
}

void Value::drawDotFile(const std::string& filename)
{
    exportDot(m_value, filename);
}

double Value::data()
{
    if (!m_value)
    {
        std::cout << "No underlying value" << std::endl;
        throw std::exception{};
    }
    return m_value->getData();
}

double Value::grad()
{
    if (!m_value)
    {
        std::cout << "No underlying value" << std::endl;
        throw std::exception{};
    }
    return m_value->getGrad();
}

Value operator+(double lhs, const Value& rhs)
{
    std::ostringstream stream;
    stream << lhs;
    return Value(lhs, stream.str()) + rhs;
}
Value operator+(const Value& lhs, double rhs)
{
    std::ostringstream stream;
    stream << rhs;
    return lhs + Value(rhs, stream.str());
}

Value operator*(double lhs, const Value& rhs)
{
    std::ostringstream stream;
    stream << lhs;
    return Value(lhs, stream.str()) * rhs;
}
Value operator*(const Value& lhs, double rhs)
{
    std::ostringstream stream;
    stream << rhs;
    return lhs * Value(rhs, stream.str());
}

Value operator/(double lhs, const Value& rhs)
{
    std::ostringstream stream;
    stream << lhs;
    return Value(lhs, stream.str()) / rhs;
}
Value operator/(const Value& lhs, double rhs)
{
    std::ostringstream stream;
    stream << rhs;
    return lhs / Value(rhs, stream.str());
}

Value operator-(double lhs, const Value& rhs)
{
    std::ostringstream stream;
    stream << lhs;
    return Value(lhs, stream.str()) - rhs;
}
Value operator-(const Value& lhs, double rhs)
{
    std::ostringstream stream;
    stream << rhs;
    return lhs - Value(rhs, stream.str());
}

} // namespace cppty