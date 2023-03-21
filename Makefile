.PHONY: lint format env

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
SHELL := /bin/bash

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Lint src directory using flake8
lint:
	flake8 EnformerCelltyping

## Format src directory using black
format: 
	isort EnformerCelltyping && black EnformerCelltyping
    
## Set up python interpreter environment and install basic dependencies
pyanlyenv:
ifeq (True,$(HAS_CONDA))
	@echo ">>> Detected conda, creating conda environment."
	
	# Create the conda environment
	conda env create -f environments/py_analysis.yml

	@echo ">>> New conda environment created successfully!."
else
	@echo ">>> No conda detected. Environment creation aborted."
endif

## Set up R interpreter environment and install basic dependencies
renv:
ifeq (True,$(HAS_CONDA))
	@echo ">>> Detected conda, creating conda environment."
	
	# Create the conda environment
	conda env create -f environments/r_bioc.yml

	@echo ">>> New conda environment created successfully!."
else
	@echo ">>> No conda detected. Environment creation aborted."
endif

## Set up sldp environment and install basic dependencies
sldpenv:
ifeq (True,$(HAS_CONDA))
        @echo ">>> Detected conda, creating conda environment."

        # Create the conda environment
        conda env create -f environments/sldp.yml

        @echo ">>> New conda environment created successfully!."
else
        @echo ">>> No conda detected. Environment creation aborted."
endif
