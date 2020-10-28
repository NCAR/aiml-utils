#!/bin/bash

# Publish the site on GitHub
ghp-import -n -p -f site/_build/html

# Print the site domain address for convenience
echo "https://ncar.github.io/aiml-utils/home.html"
