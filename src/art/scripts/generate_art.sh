#!/usr/bin/env bash

set -e
ART_BASE="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." >/dev/null 2>&1 && pwd )"

wget --quiet -P "${ART_BASE}"/art_pieces https://upload.wikimedia.org/wikipedia/commons/d/d7/Green_Sea_Turtle_grazing_seagrass.jpg
wget --quiet -P "${ART_BASE}"/art_pieces https://upload.wikimedia.org/wikipedia/commons/0/0a/The_Great_Wave_off_Kanagawa.jpg
wget --quiet -P "${ART_BASE}"/art_pieces https://upload.wikimedia.org/wikipedia/commons/b/b4/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg
wget --quiet -P "${ART_BASE}"/art_pieces https://upload.wikimedia.org/wikipedia/commons/0/00/Tuebingen_Neckarfront.jpg
wget --quiet -P "${ART_BASE}"/art_pieces https://upload.wikimedia.org/wikipedia/commons/6/68/Pillars_of_creation_2014_HST_WFC3-UVIS_full-res_denoised.jpg
wget --quiet -P "${ART_BASE}"/art_pieces https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/1024px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg