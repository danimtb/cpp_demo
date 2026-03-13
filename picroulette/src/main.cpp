#include "Config.hpp"
#include "PicRoulette.h"
#include "utils.hpp"
#include <SDL.h>
#include <SDL_ttf.h>
#include <iostream>

int main(int argc, char *argv[]) {

    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        std::cerr << "SDL_Init Error: " << SDL_GetError() << std::endl;
        return 1;
    }

    if (TTF_Init() < 0) {
        std::cerr << "TTF_Init Error: " << TTF_GetError() << std::endl;
        return 1;
    }

    conan::Config cfg(argc, argv, conan::assetPath("config.json"));
    auto picRoulette = conan::PicRoulette(cfg);
    picRoulette.run();

    SDL_Quit();
    TTF_Quit();
}
