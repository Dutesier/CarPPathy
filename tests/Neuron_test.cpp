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

#include <cmath>

#include <gtest/gtest.h>
#include <vector>

using namespace cppty;

class NeuronTest : public testing::Test
{
};

TEST_F(NeuronTest, BasicNeuron)
{
    std::vector<Value> x = { { 2.0 }, { 3.0 } };
    auto n = Neuron(2);

    n(x).drawDotFile("Neuron.dot");

    ASSERT_TRUE(false);
}
