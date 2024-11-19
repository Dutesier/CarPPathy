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

#include "Layer.hpp"
#include "Neuron.hpp"
#include "Value.hpp"

#include <vector>

namespace cppty
{

Layer::Layer(unsigned int numberOfInputs, unsigned int numberOfOutputs)
{
    for (unsigned int i = 0; i < numberOfOutputs; ++i)
    {
        neurons.emplace_back(numberOfInputs);
    }
}

std::vector<Value> Layer::operator()(const std::vector<Value>& x)
{
    std::vector<Value> outputs;
    for (auto neuron : neurons)
    {
        outputs.emplace_back(neuron(x));
    }
    return outputs;
}

} // namespace cppty