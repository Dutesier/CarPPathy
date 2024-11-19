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

#include "Layer.hpp"
#include <vector>

namespace cppty
{

class MLP
{
public:
    MLP(unsigned int numberOfInputs, const std::vector<unsigned int>& numberOfOutputs);

    std::vector<Value> operator()(const std::vector<Value>& x);

private:
    std::vector<Layer> layers;
};

} // namespace cppty