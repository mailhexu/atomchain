#!/usr/bin/env bash
set -eu

rm -rf ./dist/*
uv build
uv run --with twine python -m twine upload --repository pypi dist/* --verbose
