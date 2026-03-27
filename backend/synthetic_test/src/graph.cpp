/**
 * @file graph.cpp
 * @brief Реализация Graph Storage
 */

#include "graph.h"
#include <algorithm>
#include <sstream>

namespace graphrag {

GraphStorage::GraphStorage(const std::string& name)
    : BaseComponent(name) {
    // GraphStorage CALLS BaseComponent конструктор автоматически
    parser_ = ParserFactory::createParser("cpp");
}

bool GraphStorage::initialize() {
    Logger::info("Initializing GraphStorage: " + getName());
    
    if (parser_) {
        parser_->initialize();
    }
    
    setStatus(ComponentStatus::RUNNING);
    return true;
}

bool GraphStorage::shutdown() {
    Logger::info("Shutting down GraphStorage: " + getName());
    
    if (parser_) {
        parser_->shutdown();
    }
    
    setStatus(ComponentStatus::STOPPED);
    return true;
}

std::string GraphStorage::getVersion() const {
    return "1.0.0";
}

void GraphStorage::process() {
    Logger::debug("GraphStorage processing...");
}

void GraphStorage::addNode(const GraphNode& node) {
    // Проверка на дубликат
    auto it = std::find_if(nodes_.begin(), nodes_.end(),
        [&node](const GraphNode& n) { return n.node_id == node.node_id; });
    
    if (it == nodes_.end()) {
        nodes_.push_back(node);
        Logger::info("Added node: " + node.name);
    }
}

void GraphStorage::addEdge(const GraphEdge& edge) {
    // Валидация: оба узла должны существовать
    bool source_exists = false;
    bool target_exists = false;
    
    for (const auto& node : nodes_) {
        if (node.node_id == edge.source_id) source_exists = true;
        if (node.node_id == edge.target_id) target_exists = true;
    }
    
    if (source_exists && target_exists) {
        edges_.push_back(edge);
        Logger::info("Added edge: " + edge.source_id + " -> " + edge.target_id);
    }
}

std::vector<GraphNode> GraphStorage::getNeighbors(const std::string& node_id, int depth) {
    std::vector<GraphNode> result;
    
    // Прямые соседи (depth=1)
    for (const auto& edge : edges_) {
        if (edge.source_id == node_id) {
            auto node = getNode(edge.target_id);
            if (!node.node_id.empty()) {
                result.push_back(node);
            }
        }
    }
    
    // Рекурсивно для большей глубины
    if (depth > 1) {
        std::vector<GraphNode> deeper;
        for (const auto& neighbor : result) {
            auto sub = getNeighbors(neighbor.node_id, depth - 1);
            deeper.insert(deeper.end(), sub.begin(), sub.end());
        }
        result.insert(result.end(), deeper.begin(), deeper.end());
    }
    
    return result;
}

std::vector<GraphNode> GraphStorage::search(const std::string& query) {
    std::vector<GraphNode> result;
    std::string lower_query = query;
    std::transform(lower_query.begin(), lower_query.end(), lower_query.begin(), ::tolower);
    
    for (const auto& node : nodes_) {
        std::string lower_name = node.name;
        std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(), ::tolower);
        
        if (lower_name.find(lower_query) != std::string::npos) {
            result.push_back(node);
        }
    }
    
    return result;
}

void GraphStorage::bulkAddNodes(const std::vector<GraphNode>& nodes) {
    for (const auto& node : nodes) {
        addNode(node);
    }
    Logger::info("Bulk added " + std::to_string(nodes.size()) + " nodes");
}

void GraphStorage::bulkAddEdges(const std::vector<GraphEdge>& edges) {
    for (const auto& edge : edges) {
        addEdge(edge);
    }
    Logger::info("Bulk added " + std::to_string(edges.size()) + " edges");
}

std::vector<GraphNode> GraphStorage::getAllNodes() const {
    return nodes_;
}

std::vector<GraphEdge> GraphStorage::getAllEdges() const {
    return edges_;
}

GraphNode GraphStorage::getNode(const std::string& node_id) const {
    GraphNode empty;
    for (const auto& node : nodes_) {
        if (node.node_id == node_id) {
            return node;
        }
    }
    return empty;
}

std::string GraphStorage::generateNodeId(const std::string& type, const std::string& name) {
    std::ostringstream oss;
    oss << type << ":" << name;
    return oss.str();
}

void GraphStorage::buildEdgesFromAST(const ParseResult& parse_result) {
    // Извлекаем вызовы функций из AST и создаем ребра CALLS
    for (const auto& node : parse_result.nodes) {
        for (const auto& call : node.calls) {
            GraphEdge edge;
            edge.source_id = generateNodeId(node.type, node.name);
            edge.target_id = generateNodeId("function", call);
            edge.type = EdgeType::CALLS;
            addEdge(edge);
        }
    }
    
    // Извлекаем импорты и создаем ребра INCLUDES
    for (const auto& import : parse_result.imports) {
        GraphEdge edge;
        edge.source_id = parse_result.file_path;
        edge.target_id = import;
        edge.type = EdgeType::INCLUDES;
        addEdge(edge);
    }
}

}  // namespace graphrag
