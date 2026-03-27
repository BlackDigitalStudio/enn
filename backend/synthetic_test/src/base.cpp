/**
 * @file base.cpp
 * @brief Реализация базового класса BaseComponent
 */

#include "base.h"
#include <iostream>

namespace graphrag {

BaseComponent::BaseComponent(const std::string& name) 
    : name_(name), status_(ComponentStatus::INITIALIZED) {
}

void Logger::info(const std::string& message) {
    std::cout << "[INFO] " << message << std::endl;
}

void Logger::error(const std::string& message) {
    std::cerr << "[ERROR] " << message << std::endl;
}

void Logger::debug(const std::string& message) {
    std::cout << "[DEBUG] " << message << std::endl;
}

}  // namespace graphrag
