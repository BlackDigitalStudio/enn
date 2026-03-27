/**
 * @file parser.cpp
 * @brief Реализация AST Parser
 */

#include "parser.h"
#include <fstream>
#include <sstream>

namespace graphrag {

ASTParser::ASTParser(const std::string& name) 
    : BaseComponent(name), max_depth_(10) {
}

bool ASTParser::initialize() {
    Logger::info("Initializing ASTParser: " + getName());
    setStatus(ComponentStatus::RUNNING);
    return true;
}

bool ASTParser::shutdown() {
    Logger::info("Shutting down ASTParser: " + getName());
    setStatus(ComponentStatus::STOPPED);
    return true;
}

std::string ASTParser::getVersion() const {
    return "1.0.0";
}

void ASTParser::process() {
    Logger::debug("ASTParser processing...");
}

ParseResult ASTParser::parseFile(const std::string& file_path) {
    Logger::debug("Parsing file: " + file_path);
    
    ParseResult result;
    result.file_path = file_path;
    result.success = false;
    
    std::ifstream file(file_path);
    if (!file.is_open()) {
        result.error_message = "Cannot open file: " + file_path;
        return result;
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string code = buffer.str();
    file.close();
    
    return parseCode(code, "cpp");
}

ParseResult ASTParser::parseCode(const std::string& code, const std::string& language) {
    ParseResult result;
    result.success = false;
    
    if (!validateSyntax(code)) {
        result.error_message = "Invalid syntax";
        return result;
    }
    
    ASTNode root = buildAST(code);
    result.nodes.push_back(root);
    result.success = true;
    
    Logger::debug("Parsed " + std::to_string(result.nodes.size()) + " nodes");
    return result;
}

bool ASTParser::validateSyntax(const std::string& code) {
    // Заглушка - в реальной реализации использует tree-sitter
    return !code.empty();
}

ASTNode ASTParser::buildAST(const std::string& code) {
    ASTNode root;
    root.type = "translation_unit";
    root.name = "root";
    root.line_start = 1;
    root.line_end = 1;
    
    // Заглушка - в реальной реализации использует tree-sitter
    return root;
}

std::vector<ASTNode> ASTParser::extractFunctions(const ASTNode& root) {
    std::vector<ASTNode> functions;
    // Рекурсивный обход AST
    for (const auto& child : root.children) {
        if (child.type == "function_definition") {
            functions.push_back(child);
        }
    }
    return functions;
}

std::vector<ASTNode> ASTParser::extractClasses(const ASTNode& root) {
    std::vector<ASTNode> classes;
    for (const auto& child : root.children) {
        if (child.type == "class_specifier") {
            classes.push_back(child);
        }
    }
    return classes;
}

std::unique_ptr<ASTParser> ParserFactory::createParser(const std::string& language) {
    if (language == "cpp" || language == "c" || language == "c++") {
        return std::make_unique<ASTParser>("C++Parser");
    }
    return nullptr;
}

}  // namespace graphrag
