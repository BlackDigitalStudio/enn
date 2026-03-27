/**
 * @file base.h
 * @brief Базовый класс для всех компонентов системы
 * 
 * Архитектура: Базовый абстрактный класс с виртуальными методами.
 * Используется как основа для наследования другими компонентами.
 */

#pragma once

#include <string>
#include <vector>
#include <memory>

namespace graphrag {

/**
 * @brief Статус компонента в системе
 */
enum class ComponentStatus {
    INITIALIZED,
    RUNNING,
    STOPPED,
    ERROR
};

/**
 * @brief Базовый абстрактный класс компонента
 * 
 * Все компоненты системы наследуются от этого класса.
 * Обеспечивает единый интерфейс управления жизненным циклом.
 */
class BaseComponent {
public:
    explicit BaseComponent(const std::string& name);
    virtual ~BaseComponent() = default;
    
    // Виртуальные методы - должны быть переопределены
    virtual bool initialize() = 0;
    virtual bool shutdown() = 0;
    virtual std::string getVersion() const = 0;
    
    // Чисто виртуальный метод - наследники ОБЯЗАНЫ реализовать
    virtual void process() = 0;
    
    // Общие методы для всех компонентов
    std::string getName() const { return name_; }
    ComponentStatus getStatus() const { return status_; }
    void setStatus(ComponentStatus status) { status_ = status; }
    
protected:
    std::string name_;
    ComponentStatus status_;
    std::vector<std::string> dependencies_;
};

/**
 * @brief Вспомогательный класс для логирования
 */
class Logger {
public:
    static void info(const std::string& message);
    static void error(const std::string& message);
    static void debug(const std::string& message);
};

}  // namespace graphrag
