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

#include <functional>
#include <iostream>
#include <memory>
#include <vector>

namespace cppty
{

class Value
{

public:
    Value(double data, const std::string& label = "unlabelled");
    Value operator+(const Value& other);
    Value operator*(const Value& other);

    const std::string& getLabel() const { return label; }
    Value& getLeft() { return previous.at(0); }
    Value& getRight() { return previous.at(1); }
    void setLabel(const std::string& newLabel) { label = newLabel; };
    double getData() const { return data; }
    double getGrad() const { return grad; }
    std::string op() const;
    void setGrad(double newGrad) { grad = newGrad; };

    void backpropagate();

    Value tanh();

    size_t prevSize() const { return previous.size(); }

private:
    double data;
    std::string label;
    double grad = 0.0;

    std::function<void(void)> backward;

    std::vector<Value> previous;
    // std::unique_ptr<Value> left;
    // std::unique_ptr<Value> right;

    enum OperationType
    {
        None,
        Addition,
        Multiplication,
        Tanh
    };
    OperationType type = OperationType::None;

    void tanhBackward();
    void additionBackward();
    void multiplicationBackward();
};

void exportDot(Value& root, const std::string& filename, const std::string& rankdir = "LR");

std::vector<Value*> buildTopographic(Value& root);
} // namespace cppty