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

#include <cmath>

#include <cstddef>
#include <gtest/gtest.h>
#include <vector>

using namespace cppty;

class MLPTest : public testing::Test
{
};

TEST_F(MLPTest, BasicMultiLayerPerceptron)
{
    std::vector<Value> x = { { 2.0 }, { 3.0 }, { -1.0 } };
    auto mlp = MLP(3, { 4, 4, 1 });

    auto o = mlp(x);
    o.at(0).drawDotFile("MLP.dot");

    ASSERT_TRUE(false);
}

TEST_F(MLPTest, SmallDataset)
{
    std::vector<std::vector<Value>> xs = { { { 2.0 }, { 3.0 }, { -1.0 } },
                                           { { 3.0 }, { -1.0 }, { 0.5 } },
                                           { { 0.5 }, { 1.0 }, { 1.0 } },
                                           { { 1.0 }, { 1.0 }, { -1.0 } } };
    std::vector<Value> ys = { { 1.0 }, { -1.0 }, { -1.0 }, { 1.0 } };
    auto mlp = MLP(3, { 4, 4, 1 });

    std::vector<Value> predictions;
    for (auto i : xs)
    {
        predictions.emplace_back(mlp(i).at(0));
    }

    for (auto p : predictions)
    {
        std::cout << p.data() << std::endl;
    }

    std::vector<Value> losses;
    for (size_t i = 0; i < predictions.size(); ++i)
    {
        losses.emplace_back((predictions[i] - ys[i]).pow(2));
    }
    Value loss{ 0 };
    for (auto l : losses)
    {
        loss = loss + l;
        std::cout << l.data() << std::endl;
    }

    std::cout << "LOSS " << loss.data() << std::endl;
    loss.backpropagate();

    loss.drawDotFile("loss.dot");

    ASSERT_TRUE(false);
}
