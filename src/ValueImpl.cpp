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

#include "ValueImpl.hpp"

#include <cmath>
#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace cppty
{

ValueImpl::ValueImpl(double data, const std::string& label)
    : data(data)
    , label(label)
{
    backward = [] {};
}

ValueImpl::ValueImpl(
    double data,
    const std::vector<std::shared_ptr<ValueImpl>>& previous,
    OperationType type,
    const std::string& label)
    : data(data)
    , label(label)
    , previous(previous)
    , type(type)
{
}

void ValueImpl::backpropagate()
{
    double t = 0;
    switch (type)
    {
    case Addition:
        previous.at(0)->grad += 1 * grad;
        previous.at(1)->grad += 1 * grad;
        return;
    case Multiplication:
        previous.at(0)->grad += previous.at(1)->data * grad;
        previous.at(1)->grad += previous.at(0)->data * grad;
        return;
    case Tanh:
        t = (std::exp(2 * previous.at(0)->data) - 1) / (std::exp(2 * previous.at(0)->data) + 1);
        previous.at(0)->grad += (1 - std::pow(t, 2)) * grad;
        return;
    case Exp:
        previous.at(0)->grad += data * grad;
        return;
    case Pow:
        if (!m_builtFromPower)
        {
            std::cout << "Value not built from power: " << label << std::endl;
            throw std::exception{};
        }
        previous.at(0)->grad += *m_builtFromPower * std::pow(previous.at(0)->data, *m_builtFromPower - 1) * grad;
        return;
    case None:
    default:
        return;
    }
}

std::string ValueImpl::op() const
{
    switch (type)
    {
    case Addition:
        return "+";
    case Multiplication:
        return "*";
    case Tanh:
        return "tanh";
    case Exp:
        return "e(x)";
    case Pow:
        return "x^power";
    case None:
    default:
        return "";
    }
}

void ValueImpl::storePower(double i)
{
    m_builtFromPower = i;
}

// Helper function to trace the computational graph
void traceGraph(
    std::shared_ptr<ValueImpl> root,
    std::set<std::shared_ptr<ValueImpl>>& nodes,
    std::set<std::pair<std::shared_ptr<ValueImpl>, std::shared_ptr<ValueImpl>>>& edges)
{
    if (nodes.find(root) != nodes.end())
        return;

    nodes.insert(root);
    if (root->prevSize())
    {
        edges.emplace(root->getLeft(), root);
        traceGraph(root->getLeft(), nodes, edges);
    }

    if (root->prevSize() > 1)
    {
        edges.emplace(root->getRight(), root);
        traceGraph(root->getRight(), nodes, edges);
    }
}

// Function to export the graph as a DOT file
void exportDot(std::shared_ptr<ValueImpl> root, const std::string& filename, const std::string& rankdir)
{
    std::set<std::shared_ptr<ValueImpl>> nodes;
    std::set<std::pair<std::shared_ptr<ValueImpl>, std::shared_ptr<ValueImpl>>> edges;

    traceGraph(root, nodes, edges);

    std::ofstream dotFile(filename);

    if (!dotFile.is_open())
    {
        std::cerr << "Error: Could not open file for writing: " << filename << std::endl;
        return;
    }

    dotFile << "digraph G {" << std::endl;
    dotFile << "rankdir=" << rankdir << ";" << std::endl;

    std::unordered_map<std::shared_ptr<ValueImpl>, std::string> nodeIds;

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

std::vector<std::shared_ptr<ValueImpl>> buildTopographic(std::shared_ptr<ValueImpl> root)
{
    std::vector<std::shared_ptr<ValueImpl>> topo;
    std::set<std::shared_ptr<ValueImpl>> visited;
    std::function<void(std::shared_ptr<ValueImpl> root)> build;
    build = [&topo, &visited, &build](std::shared_ptr<ValueImpl> root)
    {
        if (!visited.contains(root))
        {
            visited.emplace(root);
            if (root->prevSize())
            {
                build(root->getLeft());
            }
            if (root->prevSize() > 1)
            {
                build(root->getRight());
            }
            topo.emplace_back(root);
        }
    };
    build(root);
    return topo;
}

} // namespace cppty