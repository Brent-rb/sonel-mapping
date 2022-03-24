#include "MainWindow.h"
#include <thread>

MainWindow::MainWindow(
		const std::string& title,
		Model* model,
		const Camera& camera,
		const float worldScale,
		SonelMapperConfig config
): GlfCameraWindow(title, camera.from, camera.at, camera.up, worldScale), 
	sonelManager(model, config) {
	
	sonelManager.setCamera(camera);
	sonelManager.calculate();
}

void MainWindow::render() {
    bool shouldDraw = false;
    auto targetMillis = static_cast<unsigned long>(1000.0f / targetFrameRate);

    if (mode == FRAME_BY_FRAME && renderFrame) {
        shouldDraw = true;
        renderFrame = false;
    }
    if (mode == PLAY && timeSinceLastFrame > static_cast<float>(targetMillis)) {
        sonelManager.nextFrame();
        shouldDraw = true;
        timeSinceLastFrame = 0.0f;
    }
    if (cameraFrame.modified) {
        sonelManager.setCamera(
                Camera{
                        cameraFrame.get_from(),
                        cameraFrame.get_at(),
                        cameraFrame.get_up()
                }
        );

        cameraFrame.modified = false;
        shouldDraw = true;
    }

    auto frameStart = std::chrono::high_resolution_clock::now();
    if (shouldDraw) {
        sonelManager.render();
    }
    auto frameEnd = std::chrono::high_resolution_clock::now();
    auto frameTime = frameEnd - frameStart;
    unsigned long frameMillis = std::chrono::duration_cast<std::chrono::milliseconds>(frameTime).count();
    timeSinceLastFrame += frameMillis;

    if (timeSinceLastFrame < targetMillis) {
        if (shouldDraw)
            printf("[MainWindow] Frame took %lu ms, sleeping for %lu ms\n", frameMillis, targetMillis - frameMillis);

        timeSinceLastFrame += 1000 / 60;
        std::this_thread::sleep_for(std::chrono::milliseconds(1000 / 60));
    }
}

void MainWindow::key(int key, int mods) {
    GlfCameraWindow::key(key, mods);

    switch (key) {
        case '[':
            mode = FRAME_BY_FRAME;
            sonelManager.previousFrame();
            renderFrame = true;
            break;
        case ']':
            mode = FRAME_BY_FRAME;
            sonelManager.nextFrame();
            renderFrame = true;
            break;
        case '=':
            if (mode == PLAY) {
                printf("[MainWindow] Switched to frame by frame mode.\n");
                mode = FRAME_BY_FRAME;
            }
            else {
                printf("[MainWindow] Switched to play mode.\n");
                mode = PLAY;
            }
            break;
    }
}

void MainWindow::draw() {
    drawCanvas();
}

void MainWindow::drawCanvas() {
    sonelManager.downloadPixels(pixels.data());

    if (fbTexture == 0)
        glGenTextures(1, &fbTexture);

    glBindTexture(GL_TEXTURE_2D, fbTexture);
    GLenum texFormat = GL_RGBA;
    GLenum texelType = GL_UNSIGNED_BYTE;
    glTexImage2D(GL_TEXTURE_2D, 0, texFormat, fbSize.x, fbSize.y, 0, GL_RGBA,
                 texelType, pixels.data());

    glDisable(GL_LIGHTING);
    glColor3f(1, 1, 1);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, fbTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glDisable(GL_DEPTH_TEST);

    glViewport(0, 0, fbSize.x, fbSize.y);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.f, (float)fbSize.x, 0.f, (float)fbSize.y, -1.f, 1.f);

    glBegin(GL_QUADS);
    {
        glTexCoord2f(0.f, 0.f);
        glVertex3f(0.f, 0.f, 0.f);

        glTexCoord2f(0.f, 1.f);
        glVertex3f(0.f, (float)fbSize.y, 0.f);

        glTexCoord2f(1.f, 1.f);
        glVertex3f((float)fbSize.x, (float)fbSize.y, 0.f);

        glTexCoord2f(1.f, 0.f);
        glVertex3f((float)fbSize.x, 0.f, 0.f);
    }
    glEnd();
}

void MainWindow::resize(const vec2i& newSize) {
	fbSize = newSize;
	sonelManager.resize(newSize);
	pixels.resize((uint64_t)newSize.x * newSize.y);
}