#!/bin/bash
#
# Generate pickle files for the CAN baseline
#

/usr/bin/time python -m datasets.main_as_images --jobs=1 --debug
