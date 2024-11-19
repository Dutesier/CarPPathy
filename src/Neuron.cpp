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

#include "Neuron.hpp"
#include "Value.hpp"

#include <numeric>
#include <random>
#include <vector>

namespace
{

double generateRandomDouble(double lower, double upper)
{
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_real_distribution<double> distribution(lower, upper);

    return distribution(generator);
}
} // namespace

namespace cppty
{

Neuron::Neuron(unsigned int numberOfInputs)
{
    for (unsigned int i = 0; i < numberOfInputs; ++i)
    {
        std::string label = "w";
        label.push_back('0' + i);
        weights.emplace_back(generateRandomDouble(-1, 1), label);
    }
    bias = Value{ generateRandomDouble(-1, 1), "bias" };
}

Value Neuron::operator()(const std::vector<Value>& x)
{
    if (weights.size() != x.size())
    {
        throw std::invalid_argument("Vectors must be of the same size.");
    }

    Value result{ bias };
    for (std::size_t i = 0; i < weights.size(); ++i)
    {
        result = result + (weights[i] * x[i]);
    }
    return (result).tanh();
}

} // namespace cppty