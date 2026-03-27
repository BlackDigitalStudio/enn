#include "physics_engine.h"
#include <cmath>

// --- PhysicsBody ---
PhysicsBody::PhysicsBody(float mass, float x, float y)
    : m_mass(mass), m_posX(x), m_posY(y),
      m_velX(0), m_velY(0), m_forceX(0), m_forceY(0) {}

PhysicsBody::~PhysicsBody() {}

void PhysicsBody::applyForce(float fx, float fy) {
    m_forceX += fx;
    m_forceY += fy;
}

float PhysicsBody::getMass() const {
    return m_mass;
}

// --- RigidBody ---
RigidBody::RigidBody(float mass, float x, float y, float radius)
    : PhysicsBody(mass, x, y), m_radius(radius) {}

void RigidBody::update(float dt) {
    float ax = m_forceX / m_mass;
    float ay = m_forceY / m_mass;
    m_velX += ax * dt;
    m_velY += ay * dt;
    m_posX += m_velX * dt;
    m_posY += m_velY * dt;
    m_forceX = 0;
    m_forceY = 0;
}

bool RigidBody::checkCollision(const RigidBody& other) const {
    float dx = m_posX - other.m_posX;
    float dy = m_posY - other.m_posY;
    float dist = std::sqrt(dx * dx + dy * dy);
    return dist < (m_radius + other.m_radius);
}

float RigidBody::getRadius() const {
    return m_radius;
}

// --- PhysicsWorld ---
PhysicsWorld::PhysicsWorld(float gravity) : m_gravity(gravity) {}

PhysicsWorld::~PhysicsWorld() {
    for (auto* body : m_bodies) {
        delete body;
    }
}

void PhysicsWorld::addBody(RigidBody* body) {
    m_bodies.push_back(body);
}

void PhysicsWorld::step(float dt) {
    for (auto* body : m_bodies) {
        body->applyForce(0, -m_gravity * body->getMass());
        body->update(dt);
    }
    resolveCollisions();
}

int PhysicsWorld::getBodyCount() const {
    return static_cast<int>(m_bodies.size());
}

void PhysicsWorld::resolveCollisions() {
    for (size_t i = 0; i < m_bodies.size(); ++i) {
        for (size_t j = i + 1; j < m_bodies.size(); ++j) {
            if (m_bodies[i]->checkCollision(*m_bodies[j])) {
                // Simple elastic collision placeholder
            }
        }
    }
}
