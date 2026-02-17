#!/bin/bash
cd /home/simonb/audio-cleanup
venv/bin/python3 -u pipeline.py --no-ai > pipeline_run.log 2>&1
echo "Exit code: $?" >> pipeline_run.log
