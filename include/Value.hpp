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

#pragma once

#include "ValueImpl.hpp"

#include <cmath>
#include <complex>
#include <memory>
#include <vector>

namespace cppty
{

class Value
{
public:
    Value(double data, const std::string& label);
    Value(
        double data,
        const std::vector<std::shared_ptr<ValueImpl>>& previous,
        OperationType type,
        const std::string& label);
    Value(const ValueImpl& val);

    Value operator+(const Value& other) const;
    Value operator-(const Value& other) const;
    Value operator*(const Value& other) const;
    Value operator/(const Value& other) const;

    Value pow(double i) const;
    Value tanh();
    Value exp();

    void setLabel(const std::string& newLabel);
    void setGrad(double grad);
    void backpropagate();

    void drawDotFile(const std::string& filename);

    double data();
    double grad();

private:
    std::shared_ptr<ValueImpl> m_value;
};

Value operator+(double lhs, const Value& rhs);
Value operator+(const Value& lhs, double rhs);

Value operator*(double lhs, const Value& rhs);
Value operator*(const Value& lhs, double rhs);

Value operator/(double lhs, const Value& rhs);
Value operator/(const Value& lhs, double rhs);

Value operator-(double lhs, const Value& rhs);
Value operator-(const Value& lhs, double rhs);

} // namespace cppty
