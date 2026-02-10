# Audio Cleanup Pipeline

An intelligent AI-powered audio processing pipeline for cleaning up speech recordings. Combines state-of-the-art AI enhancement with traditional DSP techniques including spectral noise reduction, noise gating, and adaptive mastering.

## Features

- **AI Enhancement**: Uses ClearVoice MossFormer2 for speech enhancement
- **Intelligent Channel Selection**: Automatically selects the best channel from stereo recordings based on SNR analysis
- **Adaptive Spectral Noise Reduction**: Self-calibrating noise profile detection and removal (similar to Audacity's noise reduction)
- **Noise Gate**: Removes residual hiss during quiet passages
- **Frequency Analysis**: Detects poor high-frequency content and applies appropriate filtering
- **Adaptive Mastering**: Analyzes clarity and applies intelligent EQ, de-essing, and loudness normalization
- **Batch Processing**: Process entire directories of audio files automatically
- **Multiple Format Support**: WAV, MP3, FLAC, M4A, AAC, OGG, Opus, WMA

## Requirements

### System Dependencies

**FFmpeg** must be installed on your system:

- **Ubuntu/Debian**: `sudo apt-get install ffmpeg`
- **macOS**: `brew install ffmpeg`
- **Windows**: Download from [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)

### Python Dependencies

- Python 3.8 or higher
- See `requirements.txt` for Python package dependencies

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd audio-cleanup
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify FFmpeg installation**:
   ```bash
   ffmpeg -version
   ```

## Usage

### Basic Usage

Place your audio files in the `pipeline-in` directory and run:

```bash
python pipeline.py
```

Processed files will appear in the `pipeline-out` directory as Opus files (64 kbps by default).

### Custom Input/Output Directories

```bash
python pipeline.py --input /path/to/input --output /path/to/output
```

### Advanced Configuration

The pipeline supports extensive customization through command-line arguments:

#### Output Settings
```bash
python pipeline.py --bitrate 96  # Output bitrate in kbps (default: 64)
```

#### Noise Reduction
```bash
python pipeline.py \
  --noise-reduction 15 \          # Noise reduction in dB (default: 12)
  --noise-sensitivity 0.2         # Detection sensitivity 0-1 (default: 0.15)
```

#### Noise Gate
```bash
python pipeline.py \
  --gate-threshold -40 \          # Gate threshold in dB FS (default: -45)
  --gate-attack 5 \               # Attack time in ms (default: 5)
  --gate-release 50               # Release time in ms (default: 50)
```

#### Frequency Processing
```bash
python pipeline.py \
  --lowpass-threshold 0.015 \     # HF detection threshold (default: 0.015)
  --lowpass-cutoff 8000           # Low-pass cutoff in Hz (default: 8000)
```

#### Mastering
```bash
python pipeline.py \
  --highpass 100 \                # Highpass filter in Hz (default: 100)
  --loudness -16 \                # Target loudness in LUFS (default: -16)
  --true-peak -1.5                # True peak limit in dB (default: -1.5)
```

### View All Options

```bash
python pipeline.py --help
```

## Processing Pipeline

The pipeline processes audio through the following stages:

1. **Channel Selection**: Analyzes stereo files and selects the channel with better SNR
2. **Preprocessing**: 
   - Clips transient peaks (mic knocks)
   - Applies low-pass filter if poor high-frequency content is detected
   - Speech normalization
   - Headroom adjustment for AI processing
3. **AI Enhancement**: ClearVoice MossFormer2 speech enhancement
4. **Spectral Noise Reduction**: Adaptive noise profile detection and removal
5. **Noise Gate**: Removes residual hiss in quiet passages
6. **Adaptive Mastering**:
   - High-pass filtering
   - Intelligent EQ based on clarity analysis
   - De-essing
   - Loudness normalization (LUFS)
   - Peak limiting

## Configuration

Default configuration can be found in the `PipelineConfig` class in `pipeline.py`. You can modify these defaults or override them via command-line arguments.

## Output Format

By default, files are output as Opus format at 64 kbps, which provides excellent quality for speech at small file sizes. The output format is optimized for:

- Speech clarity
- Small file size
- Broadcast-ready loudness (-16 LUFS)
- Controlled peaks (-1.5 dB true peak)

## Troubleshooting

- **"FFmpeg not found"**: Ensure FFmpeg is installed and in your system PATH
- **"No audio files found"**: Check that your input directory contains supported audio formats
- **Poor results**: Try adjusting noise reduction and gate settings based on your source material
- **Clipping/distortion**: Reduce `--loudness` target or increase `--true-peak` limit

## License

See `LICENSE` file for details.
