#include "MainWindow.h"

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
	if (cameraFrame.modified) {
		sonelManager.setCamera(
			Camera{
				cameraFrame.get_from(),
				cameraFrame.get_at(),
				cameraFrame.get_up()
			}
		);
		
		cameraFrame.modified = false;
	}
	
	sonelManager.render();
}

void MainWindow::draw() {
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