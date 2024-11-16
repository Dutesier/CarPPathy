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

#include <gtest/gtest.h>
#include <memory>

using namespace cppty;

class ValueTest : public testing::Test
{
};

TEST_F(ValueTest, CarPPAthyCompiles)
{
    Value a(2, "a");
    Value b(-3, "b");
    Value c(10, "c");
    auto e = a * b;
    e.setLabel("e");
    auto d = e + c;
    d.setLabel("d");
    auto f = Value(-2, "f");
    auto L = d * f;
    L.setLabel("L");

    exportDot(L, "visualize_graph.dot");

    ASSERT_TRUE(false);
}

TEST_F(ValueTest, Neuron)
{
    // inputs
    Value x1(2, "x1");
    Value x2(0, "x2");
    // Weights
    Value w1(-3, "w1");
    Value w2(1, "w2");
    // a bias
    Value b(6.8813735870195432, "b");

    auto x1w1 = x1 * w1;
    auto x2w2 = x2 * w2;
    auto x1w1x2w2 = x1w1 + x2w2;

    auto n = x1w1x2w2 + b;
    n.setLabel("n");

    auto o = n.tanh();
    o.setLabel("o");

    o.setGrad(1);

    auto topo = buildTopographic(o);
    for (auto i = topo.rbegin(); i != topo.rend(); i++)
    {
        if (!*i)
        {
            return;
        }
        (*i)->backpropagate();
    }

    exportDot(o, "visualize_graph.dot");

    ASSERT_TRUE(false);
}