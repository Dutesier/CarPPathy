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
#include <memory>
#include <optional>
#include <vector>

namespace cppty
{

enum OperationType
{
    None,
    Addition,
    Multiplication,
    Tanh,
    Exp,
    Pow
};

class ValueImpl
{

public:
    ValueImpl(double data, const std::string& label = "unlabelled");
    ValueImpl(
        double data,
        const std::vector<std::shared_ptr<ValueImpl>>& previous,
        OperationType type,
        const std::string& label = "unlabelled");

    const std::string& getLabel() const { return label; }
    std::shared_ptr<ValueImpl> getLeft() { return previous.at(0); }
    std::shared_ptr<ValueImpl> getRight() { return previous.at(1); }
    void setLabel(const std::string& newLabel) { label = newLabel; };
    double getData() const { return data; }
    double getGrad() const { return grad; }
    std::string op() const;
    void setGrad(double newGrad) { grad = newGrad; };

    void backpropagate();

    size_t prevSize() const { return previous.size(); }
    void storePower(double i);

private:
    double data;
    std::string label;
    double grad = 0.0;

    std::function<void(void)> backward;

    std::vector<std::shared_ptr<ValueImpl>> previous;

    OperationType type = OperationType::None;
    std::optional<double> m_builtFromPower;

    void tanhBackward();
    void additionBackward();
    void multiplicationBackward();
};

void exportDot(std::shared_ptr<ValueImpl> root, const std::string& filename, const std::string& rankdir = "LR");

std::vector<std::shared_ptr<ValueImpl>> buildTopographic(std::shared_ptr<ValueImpl> root);
} // namespace cppty