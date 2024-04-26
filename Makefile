#.python-makefile
# Makefile for managing a Python project

.PHONY: setup clean install test lint run help

# Configuration
VENV_NAME ?= venv
PYTHON = $(VENV_NAME)/bin/python
PIP = $(VENV_NAME)/bin/pip

# Default command
all: install

# Setup the Python Virtual Environment
setup: 
	@echo "Creating virtual environment..."
	python3 -m venv $(VENV_NAME)
	@echo "Virtual environment created."

# actiate env
activate:
	@echo "Activating virtual environment..."
	source $(VENV_NAME)/bin/activate
	@echo "Virtual environment activated."

# deactivate env 
deactivate:
	@echo "Deactivating virtual environment..."
	deactivate
	@echo "Virtual environment deactivated."

# Install Python Dependencies
install: setup
	@echo "Installing dependencies..."
	$(PIP) install -r requirements.txt
	@echo "Dependencies installed."

# freeze Python Dependencies
freeze: setup
	@echo "Freezing dependencies..."
	$(PIP) freeze > requirements.txt
	@echo "Dependencies frozen."

# update Python Dependencies
update: setup
	@echo "Updating dependencies..."
	$(PIP) install --upgrade -r requirements.txt
	@echo "Dependencies updated."

# Run tests with pytest
test: install
	@echo "Running tests..."
	$(PYTHON) -m pytest

# Lint the project with flake8
lint: install
	@echo "Linting the code..."
	$(PYTHON) -m flake8 --ignore=E501 src/

# Clean up the project
clean:
	@echo "Cleaning up..."
	rm -rf $(VENV_NAME)
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -exec rm -rf {} +
	@echo "Clean up done."

# run 
run: install
	@echo "Running the project..."
	$(PYTHON) src/main.py

# Help
help:
	@echo "Makefile for managing the Python project"
	@echo "Available commands:"
	@echo "   setup: Set up the Python virtual environment."
	@echo "   install: Install the Python dependencies."
	@echo "   test: Run tests."
	@echo "   lint: Check style with flake8."
	@echo "   clean: Clean up the project."
	@echo "   help: Display this help message."
	@echo "   run: Run the main application script."
