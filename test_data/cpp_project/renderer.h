#pragma once
#include "physics_engine.h"

class Renderer {
public:
    Renderer(int width, int height);
    ~Renderer();

    void drawBody(const RigidBody& body);
    void drawWorld(const PhysicsWorld& world);
    void clear();
    void present();

private:
    int m_width;
    int m_height;
    bool m_initialized;
};
