#!/usr/bin/env python3
"""Test script to process a very large file with the pipeline."""

from pipeline import IntelligentStudioPipeline, PipelineConfig

# Create config with AI disabled
config = PipelineConfig()
config.ENABLE_AI_ENHANCEMENT = False

# Initialize pipeline
pipeline = IntelligentStudioPipeline(config)

# Process the 94-minute file
input_file = "pipeline-in/Sun08Feb26-1025-auditorium_translation.mp3"
output_file = "pipeline-out/test_large_sun.opus"

print(f"Processing: {input_file}")
try:
    pipeline.process_file(input_file, output_file)
    print(f"SUCCESS: {output_file}")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
