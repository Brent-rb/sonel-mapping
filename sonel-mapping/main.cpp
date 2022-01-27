// ======================================================================== //
// Copyright 2018-2019 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include "SonelMapper.h"
#include "Model.h"
#include "MainWindow.h"
#include "SoundSource.h"

// our helper library for window handling
#include <GL/gl.h>

struct Scene {
	Model* model;
	Camera camera;
	std::vector<SoundSource> soundSources;
	int frequencies;
	float echogramDuration;
	float soundSpeed;
	float earSize;
};

Scene loadSponza();

extern "C" int main(int argC, char** argV) {
	try {
		Scene scene = loadSponza();

		// something approximating the scale of the world, so the
		// camera knows how much to move for any given user interaction:
		const float worldScale = length(scene.model->bounds.span());

		SonelMapperConfig config = {
			scene.soundSources,
			scene.echogramDuration,
			scene.soundSpeed,
			scene.earSize,
			static_cast<uint32_t>(scene.frequencies)
		};

		MainWindow* window = new MainWindow(
			"Sonel-Mapping",
			scene.model, 
			scene.camera,
			worldScale,
			config
		);

		window->run();

	}
	catch (std::runtime_error& e) {
		std::cout << GDT_TERMINAL_RED << "FATAL ERROR: " << e.what() << GDT_TERMINAL_DEFAULT << std::endl;
		exit(1);
	}

	return 0;
}

Scene loadSponza() {
	Model* model = loadObj("../../models/sponza.obj");

	Camera camera = {
		/*from*/vec3f(0, 150, model->bounds.center().z),
		/* at */vec3f(-900, 150, model->bounds.center().z),
		/* up */vec3f(0.f,1.f,0.f)
	};

	std::vector<SoundSource> soundSources;
	std::vector<SoundFrequency> frequencies1;
	std::vector<SoundFrequency> frequencies2;

	SoundFrequency frequency1(1000, 1000000, 8), frequency2(2000, 1000000, 8), frequency3(4000, 1000000, 8), frequency4(8000, 1000000, 8);
	std::vector<float> decibels1, decibels2, decibels3, decibels4;
	for (int i = 0; i < 10; i++) {
		decibels2.push_back(0.0);
		decibels3.push_back(0.0);
		decibels3.push_back(0.0);
		decibels4.push_back(0.0);
		decibels4.push_back(0.0);
		decibels4.push_back(0.0);
		decibels4.push_back(0.0);
	}

	decibels1.push_back(70.0f);
	decibels2.push_back(70.0f);
	decibels3.push_back(70.0f);
	decibels4.push_back(70.0f);

	frequency1.setDecibels(decibels1);
	frequency2.setDecibels(decibels2);
	frequency3.setDecibels(decibels3);
	frequency4.setDecibels(decibels4);

	frequencies1.push_back(frequency1);
	frequencies1.push_back(frequency3);
	frequencies2.push_back(frequency2);
	frequencies2.push_back(frequency4);

	SoundSource source1;
	source1.direction = normalize(vec3f(-1.0f, 0.0f, 0.0f));
	source1.position = vec3f(-900, 150, model->bounds.center().z);
	source1.setFrequencies(frequencies1);

	SoundSource source2;
	source2.direction = normalize(vec3f(-1.0f, 0.0f, 0.0f));
	source2.position = vec3f(-800, 150, model->bounds.center().z + 100);
	source2.setFrequencies(frequencies2);

	soundSources.push_back(source1);
	soundSources.push_back(source2);

	// some simple, hard-coded light ... obviously, only works for sponza
	const float light_size = 200.f;

	QuadLight light = {
		/* origin */ vec3f(-1000 - light_size,800,-light_size),
		/* edge 1 */ vec3f(2.f * light_size,0,0),
		/* edge 2 */ vec3f(0,0,2.f * light_size),
		/* power */  vec3f(3000000.f)
	};

	return {
		model,
		camera,
		soundSources,
		4,
		6.0f, // Seconds
		343.0f, // m/s
		0.3f // meter
	};
}