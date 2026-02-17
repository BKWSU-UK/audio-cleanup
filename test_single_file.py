#!/usr/bin/env python3
"""Test script to process a single file with the pipeline."""

from pipeline import IntelligentStudioPipeline, PipelineConfig

# Create config with AI disabled
config = PipelineConfig()
config.ENABLE_AI_ENHANCEMENT = False

# Initialize pipeline
pipeline = IntelligentStudioPipeline(config)

# Process one large file
input_file = "pipeline-in/murli-2026-02-10.mp3"
output_file = "pipeline-out/test_murli.opus"

print(f"Processing: {input_file}")
try:
    pipeline.process_file(input_file, output_file)
    print(f"SUCCESS: {output_file}")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
