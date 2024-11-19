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

#include "MLP.hpp"
#include "Neuron.hpp"
#include "Value.hpp"

#include <vector>

namespace cppty
{

MLP::MLP(unsigned int numberOfInputs, const std::vector<unsigned int>& numberOfOutputs)
{
    std::vector<std::tuple<unsigned int, unsigned int>> result;

    unsigned int previous = numberOfInputs; // Start with `nin` as the "previous" element

    for (const auto& current : numberOfOutputs)
    {
        layers.emplace_back(previous, current);
        previous = current; // Update `previous` to the current value
    }
}

std::vector<Value> MLP::operator()(const std::vector<Value>& x)
{
    auto copyedX = x;
    for (auto layer : layers)
    {
        copyedX = layer(copyedX);
    }
    return copyedX;
}

} // namespace cppty