#!/usr/bin/env bash

set -e
ART_BASE="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." >/dev/null 2>&1 && pwd )"

wget --quiet -P "${ART_BASE}"/art_pieces -i "${ART_BASE}"/scripts/download_urls.txt