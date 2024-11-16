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
#include <cstddef>
#include <fstream>
#include <iostream>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace cppty
{

Value::Value(double data, const std::string& label)
    : data(data)
    , label(label)
{
    backward = [] {};
}

Value Value::operator+(const Value& other)
{
    Value val = data + other.data;
    val.label = label + "+" + other.label;
    val.type = OperationType::Addition;
    val.previous.emplace_back(std::move(*this));
    val.previous.emplace_back(other);

    return val;
}

Value Value::operator*(const Value& other)
{
    Value val = data * other.data;
    val.label = label + "*" + other.label;
    val.type = OperationType::Multiplication;
    val.previous.emplace_back(std::move(*this));
    val.previous.emplace_back(other);

    return val;
}

void Value::backpropagate()
{
    double t = 0;
    switch (type)
    {
    case Addition:
        previous.at(0).grad = 1 * grad;
        previous.at(1).grad = 1 * grad;
        return;
    case Multiplication:
        previous.at(0).grad = previous.at(1).data * grad;
        previous.at(1).grad = previous.at(0).data * grad;
        return;
    case Tanh:
        t = (std::exp(2 * previous.at(0).data) - 1) / (std::exp(2 * previous.at(0).data) + 1);
        previous.at(0).grad = (1 - std::pow(t, 2)) * grad;
        return;
    case None:
    default:
        return;
    }
}

std::string Value::op() const
{
    switch (type)
    {
    case Addition:
        return "+";
    case Multiplication:
        return "*";
    case Tanh:
        return "tanh";
    case None:
    default:
        return "";
    }
}

Value Value::tanh()
{
    const auto& n = data;
    auto t = (std::exp(2 * n) - 1) / (std::exp(2 * n) + 1);
    auto ret = Value(t, label + ".tanh()");
    ret.type = Tanh;
    ret.previous.emplace_back(std::move(*this));
    ret.backward = [&ret, t] { ret.previous.at(0).grad = (1 - std::pow(t, 2)) * ret.grad; };
    return ret;
}

// Helper function to trace the computational graph
void traceGraph(Value& root, std::set<Value*>& nodes, std::set<std::pair<Value*, Value*>>& edges)
{
    if (nodes.find(&root) != nodes.end())
        return;

    nodes.insert(&root);
    if (root.prevSize())
    {
        edges.emplace(&root.getLeft(), &root);
        traceGraph(root.getLeft(), nodes, edges);
    }

    if (root.prevSize() > 1)
    {
        edges.emplace(&root.getRight(), &root);
        traceGraph(root.getRight(), nodes, edges);
    }
}

// Function to export the graph as a DOT file
void exportDot(Value& root, const std::string& filename, const std::string& rankdir)
{
    std::set<Value*> nodes;
    std::set<std::pair<Value*, Value*>> edges;

    traceGraph(root, nodes, edges);

    std::ofstream dotFile(filename);

    if (!dotFile.is_open())
    {
        std::cerr << "Error: Could not open file for writing: " << filename << std::endl;
        return;
    }

    dotFile << "digraph G {" << std::endl;
    dotFile << "rankdir=" << rankdir << ";" << std::endl;

    std::unordered_map<const Value*, std::string> nodeIds;

    // Generate nodes
    for (const auto& node : nodes)
    {
        const std::string& nodeId = node->getLabel();
        nodeIds[node] = nodeId;

        dotFile << "\"" << nodeId << "\" [label=\"{ " << nodeId << "| data " << node->getData() << " | grad "
                << node->getGrad() << " }\", shape=record];" << std::endl;

        if (!node->op().empty())
        {
            std::string opNodeId = nodeId + "_op";
            dotFile << "\"" << opNodeId << "\" [label=\"" << node->op() << "\"];" << std::endl;
            dotFile << "\"" << opNodeId << "\" -> \"" << nodeId << "\";" << std::endl;
        }
    }

    // Generate edges
    for (const auto& edge : edges)
    {
        auto n1 = edge.first;
        auto n2 = edge.second;

        std::string opNodeId = nodeIds[n2] + "_op";
        if (!nodeIds[n1].empty())
            dotFile << "\"" << nodeIds[n1] << "\" -> \"" << opNodeId << "\";" << std::endl;
    }

    dotFile << "}" << std::endl;

    dotFile.close();
    std::cout << "Graph exported to " << filename << std::endl;
}

std::vector<Value*> buildTopographic(Value& root)
{
    std::vector<Value*> topo;
    std::set<const Value*> visited;
    std::function<void(Value & root)> build;
    build = [&topo, &visited, &build](Value& root)
    {
        if (!visited.contains(&root))
        {
            visited.emplace(&root);
            if (root.prevSize())
            {
                build(root.getLeft());
            }
            if (root.prevSize() > 1)
            {
                build(root.getRight());
            }
            topo.emplace_back(&root);
        }
    };
    build(root);
    return topo;
}

} // namespace cppty