#include "physics_engine.h"
#include "renderer.h"

int main() {
    PhysicsWorld world(9.81f);

    RigidBody* ball1 = new RigidBody(1.0f, 0.0f, 10.0f, 0.5f);
    RigidBody* ball2 = new RigidBody(2.0f, 1.0f, 10.0f, 0.7f);

    world.addBody(ball1);
    world.addBody(ball2);

    Renderer renderer(800, 600);

    for (int i = 0; i < 1000; ++i) {
        world.step(0.016f);
        renderer.clear();
        renderer.drawWorld(world);
        renderer.present();
    }

    return 0;
}
