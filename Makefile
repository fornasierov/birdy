# Birdy handy commands

.PHONY: requirements

requirements-dev:
	poetry export --with dev --without-hashes --format=requirements.txt > requirements-dev.txt