#!/bin/bash

# Clean out the current build
jupyter-book clean site/_build
# Build the site with jupyer-book
jupyter-book build site
