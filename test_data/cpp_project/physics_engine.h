#pragma once
#include <vector>
#include <string>

// Base class for all physics objects
class PhysicsBody {
public:
    PhysicsBody(float mass, float x, float y);
    virtual ~PhysicsBody();

    virtual void update(float dt) = 0;
    void applyForce(float fx, float fy);
    float getMass() const;

protected:
    float m_mass;
    float m_posX, m_posY;
    float m_velX, m_velY;
    float m_forceX, m_forceY;
};

// Rigid body with collision
class RigidBody : public PhysicsBody {
public:
    RigidBody(float mass, float x, float y, float radius);
    void update(float dt) override;
    bool checkCollision(const RigidBody& other) const;
    float getRadius() const;

private:
    float m_radius;
};

// Manages all physics simulation
class PhysicsWorld {
public:
    PhysicsWorld(float gravity = 9.81f);
    ~PhysicsWorld();

    void addBody(RigidBody* body);
    void step(float dt);
    void resolveCollisions();
    int getBodyCount() const;

private:
    std::vector<RigidBody*> m_bodies;
    float m_gravity;
};
