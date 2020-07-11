#!/usr/bin/env bash

set -e
cd "$( dirname "${BASH_SOURCE[0]}" )/.." >/dev/null 2>&1 && pwd

brew install sdl sdl_image sdl_mixer sdl_ttf portmidi
pip install https://github.com/pygame/pygame/archive/master.zip

git --version >/dev/null 2>&1
GIT_IS_AVAILABLE=$?

if not [ $GIT_IS_AVAILABLE -eq 0 ]; then
  brew install git
fi

git clone https://github.com/nmtremblay/lofi-samples.git
