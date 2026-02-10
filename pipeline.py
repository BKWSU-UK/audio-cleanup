import os
import subprocess
import numpy as np
import scipy.io.wavfile as wav
from scipy.fft import rfft, rfftfreq, irfft
from scipy.signal import stft, istft
from pathlib import Path
import argparse
from clearvoice import ClearVoice

# ============================================================================
# CONFIGURATION DEFAULTS
# ============================================================================
class PipelineConfig:
    """Default configuration parameters for the audio processing pipeline."""
    
    # Phase control - enable/disable processing stages
    ENABLE_PREPROCESSING = True  # Channel selection, clipping, normalization
    ENABLE_AI_ENHANCEMENT = True  # ClearVoice AI enhancement
    ENABLE_SPECTRAL_DENOISE = True  # Adaptive spectral noise reduction
    ENABLE_NOISE_GATE = True  # Noise gate for quiet passages
    ENABLE_MASTERING = True  # EQ, de-essing, loudness normalization
    
    # Output settings
    OUTPUT_BITRATE = 64  # Opus bitrate in kbps (ignored for FLAC)
    OUTPUT_FORMAT = 'opus'  # Output format: 'opus' or 'flac'
    
    # Noise reduction settings
    NOISE_REDUCTION_DB = 12  # Amount of spectral noise reduction in dB
    NOISE_SENSITIVITY = 0.15  # Sensitivity for detecting noise-only regions (0-1)
    
    # Noise gate settings
    GATE_THRESHOLD_DB = -45  # Gate threshold in dB FS
    GATE_ATTACK_MS = 5  # Gate attack time in milliseconds
    GATE_RELEASE_MS = 50  # Gate release time in milliseconds
    
    # Frequency analysis settings
    LOW_PASS_THRESHOLD = 0.015  # Threshold for detecting poor high-frequency content
    LOW_PASS_CUTOFF = 8000  # Low-pass filter cutoff frequency in Hz
    
    # Preprocessing settings
    SPEECHNORM_EXPANSION = 6  # Speech normalization expansion factor (higher = more aggressive)
    
    # Mastering settings
    HIGHPASS_FREQ = 100  # Highpass filter frequency in Hz
    EQ_CENTER_FREQ = 4000  # EQ center frequency in Hz
    EQ_WIDTH = 2000  # EQ bandwidth in Hz
    DEESSER_FREQ = 0.25  # De-esser frequency (normalized)
    DEESSER_STRENGTH = 0.5  # De-esser strength
    LOUDNESS_TARGET = -16  # Target loudness in LUFS
    TRUE_PEAK = -1.5  # True peak limit in dB

class IntelligentStudioPipeline:
    def __init__(self, config=None):
        """Initialize the pipeline with optional configuration.
        
        Args:
            config: PipelineConfig object or None to use defaults
        """
        self.config = config if config is not None else PipelineConfig()
        # Initialize the AI engine only if needed
        self.ai_engine = None
        if self.config.ENABLE_AI_ENHANCEMENT:
            self.ai_engine = ClearVoice(task='speech_enhancement', model_names=['MossFormer2_SE_48K'])

    def _convert_to_wav_for_analysis(self, audio_path):
        """Convert any audio format to temporary WAV for analysis."""
        import tempfile
        temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_wav.close()
        
        cmd = f"ffmpeg -i '{audio_path}' -ar 48000 '{temp_wav.name}' -y"
        result = subprocess.run(cmd, shell=True, capture_output=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Failed to convert {audio_path} to WAV for analysis")
        
        return temp_wav.name

    def select_best_channel(self, audio_path):
        """Analyze stereo audio and select the channel with better quality.
        
        Criteria:
        - Higher RMS energy (louder = better signal)
        - Lower noise floor in quiet regions
        - Better signal-to-noise ratio
        
        Returns the best channel index (0 or 1) or None if mono.
        """
        temp_wav = None
        try:
            # Convert to WAV if not already
            if not audio_path.lower().endswith('.wav'):
                temp_wav = self._convert_to_wav_for_analysis(audio_path)
                analysis_path = temp_wav
            else:
                analysis_path = audio_path
            
            sr, data = wav.read(analysis_path)
            
            # If mono, return None
            if len(data.shape) == 1:
                return None
            
            # Analyze both channels
            left = data[:, 0].astype(np.float32)
            right = data[:, 1].astype(np.float32)
            
            # Calculate RMS energy for each channel
            rms_left = np.sqrt(np.mean(left ** 2))
            rms_right = np.sqrt(np.mean(right ** 2))
            
            # Calculate noise floor (bottom 10% of energy)
            left_sorted = np.sort(np.abs(left))
            right_sorted = np.sort(np.abs(right))
            noise_floor_left = np.mean(left_sorted[:len(left_sorted)//10])
            noise_floor_right = np.mean(right_sorted[:len(right_sorted)//10])
            
            # Calculate SNR approximation
            snr_left = rms_left / (noise_floor_left + 1e-10)
            snr_right = rms_right / (noise_floor_right + 1e-10)
            
            # Select channel with better SNR
            if snr_left > snr_right:
                print(f"Stereo detected: Using LEFT channel (SNR: {snr_left:.1f} vs {snr_right:.1f})")
                return 0
            else:
                print(f"Stereo detected: Using RIGHT channel (SNR: {snr_right:.1f} vs {snr_left:.1f})")
                return 1
        finally:
            if temp_wav and os.path.exists(temp_wav):
                os.remove(temp_wav)

    def adaptive_spectral_denoise(self, wav_path, output_path, noise_reduction_db=None, sensitivity=None):
        """Self-calibrating spectral noise reduction using automatic noise profile detection.
        
        This works similarly to Audacity's noise reduction:
        1. Detects quiet regions (likely noise-only)
        2. Builds a noise profile from those regions
        3. Applies spectral subtraction to remove the noise
        
        Args:
            wav_path: Input WAV file path
            output_path: Output WAV file path
            noise_reduction_db: Amount of noise reduction in dB (uses config default if None)
            sensitivity: Threshold for detecting noise-only regions (uses config default if None)
        """
        # Use config defaults if not specified
        if noise_reduction_db is None:
            noise_reduction_db = self.config.NOISE_REDUCTION_DB
        if sensitivity is None:
            sensitivity = self.config.NOISE_SENSITIVITY
        
        sr, data = wav.read(wav_path)
        if len(data.shape) > 1:
            # Select best channel instead of averaging
            best_channel = self.select_best_channel(wav_path)
            if best_channel is not None:
                data = data[:, best_channel]
            else:
                data = data.mean(axis=1)
        
        # Normalize to float
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
        
        # STFT parameters
        nperseg = 2048
        noverlap = nperseg // 2
        
        # Compute STFT
        f, t, Zxx = stft(data, sr, nperseg=nperseg, noverlap=noverlap)
        
        # Calculate energy per time frame
        frame_energy = np.sum(np.abs(Zxx) ** 2, axis=0)
        
        # Detect noise-only regions (bottom sensitivity% of energy)
        energy_threshold = np.percentile(frame_energy, sensitivity * 100)
        noise_frames = frame_energy < energy_threshold
        
        # Build noise profile from quiet regions across the ENTIRE file
        # This ensures we have a good profile even at the beginning
        if np.sum(noise_frames) > 0:
            noise_profile = np.median(np.abs(Zxx[:, noise_frames]), axis=1)
        else:
            # Fallback: use lowest 10% of each frequency bin across entire file
            noise_profile = np.percentile(np.abs(Zxx), 10, axis=1)
        
        # For better initial performance, also look at the first 2 seconds specifically
        # and boost the noise profile if the beginning is quieter (common in recordings)
        first_2sec_frames = int(2.0 * sr / (nperseg - noverlap))
        if first_2sec_frames < Zxx.shape[1]:
            initial_noise = np.percentile(np.abs(Zxx[:, :first_2sec_frames]), 20, axis=1)
            # Use the maximum of the two profiles to ensure we catch initial noise
            noise_profile = np.maximum(noise_profile, initial_noise * 0.8)
        
        # Apply spectral subtraction with oversubtraction factor
        reduction_factor = 10 ** (noise_reduction_db / 20)
        noise_profile_expanded = noise_profile[:, np.newaxis]
        
        # Magnitude and phase
        magnitude = np.abs(Zxx)
        phase = np.angle(Zxx)
        
        # Subtract noise profile with oversubtraction
        magnitude_cleaned = magnitude - (reduction_factor * noise_profile_expanded)
        
        # Apply spectral floor (don't reduce below -40dB of original)
        spectral_floor = magnitude * 0.01
        magnitude_cleaned = np.maximum(magnitude_cleaned, spectral_floor)
        
        # Reconstruct complex spectrogram
        Zxx_cleaned = magnitude_cleaned * np.exp(1j * phase)
        
        # Inverse STFT
        _, audio_cleaned = istft(Zxx_cleaned, sr, nperseg=nperseg, noverlap=noverlap)
        
        # Ensure same length as input
        audio_cleaned = audio_cleaned[:len(data)]
        
        # Convert back to int16
        audio_cleaned = np.clip(audio_cleaned, -1.0, 1.0)
        audio_cleaned = (audio_cleaned * 32767).astype(np.int16)
        
        # Write output
        wav.write(output_path, sr, audio_cleaned)
        
        return output_path

    def apply_noise_gate(self, wav_path, output_path, threshold_db=None, attack_ms=None, release_ms=None):
        """Apply a noise gate to remove residual hiss during quiet passages.
        
        Args:
            wav_path: Input WAV file path
            output_path: Output WAV file path
            threshold_db: Gate threshold in dB FS (uses config default if None)
            attack_ms: Attack time in milliseconds (uses config default if None)
            release_ms: Release time in milliseconds (uses config default if None)
        """
        # Use config defaults if not specified
        if threshold_db is None:
            threshold_db = self.config.GATE_THRESHOLD_DB
        if attack_ms is None:
            attack_ms = self.config.GATE_ATTACK_MS
        if release_ms is None:
            release_ms = self.config.GATE_RELEASE_MS
        
        sr, data = wav.read(wav_path)
        if len(data.shape) > 1:
            data = data.mean(axis=1)
        
        # Normalize to float
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
        
        # Convert threshold from dB to linear
        threshold_linear = 10 ** (threshold_db / 20)
        
        # Calculate envelope using RMS in small windows
        window_size = int(sr * 0.01)  # 10ms windows
        hop_size = window_size // 2
        
        # Pad data for windowing
        padded_data = np.pad(data, (window_size // 2, window_size // 2), mode='reflect')
        
        # Calculate RMS envelope
        envelope = np.array([
            np.sqrt(np.mean(padded_data[i:i+window_size]**2))
            for i in range(0, len(data), hop_size)
        ])
        
        # Create gate signal (1 = open, 0 = closed)
        gate_signal = (envelope > threshold_linear).astype(np.float32)
        
        # Apply attack and release smoothing
        attack_samples = int(sr * attack_ms / 1000)
        release_samples = int(sr * release_ms / 1000)
        
        # Smooth the gate signal
        smoothed_gate = np.copy(gate_signal)
        for i in range(1, len(gate_signal)):
            if gate_signal[i] > smoothed_gate[i-1]:
                # Attack (gate opening)
                alpha = 1.0 - np.exp(-1.0 / max(1, attack_samples / hop_size))
            else:
                # Release (gate closing)
                alpha = 1.0 - np.exp(-1.0 / max(1, release_samples / hop_size))
            smoothed_gate[i] = alpha * gate_signal[i] + (1 - alpha) * smoothed_gate[i-1]
        
        # Interpolate gate signal to match audio length
        gate_full = np.interp(
            np.arange(len(data)),
            np.arange(len(smoothed_gate)) * hop_size,
            smoothed_gate
        )
        
        # Apply gate to audio
        gated_audio = data * gate_full
        
        # Convert back to int16
        gated_audio = np.clip(gated_audio, -1.0, 1.0)
        gated_audio = (gated_audio * 32767).astype(np.int16)
        
        # Write output
        wav.write(output_path, sr, gated_audio)
        
        return output_path

    def needs_low_pass(self, audio_path):
        """Analyzes if high-frequencies are speech or codec junk."""
        temp_wav = None
        try:
            # Convert to WAV if not already
            if not audio_path.lower().endswith('.wav'):
                temp_wav = self._convert_to_wav_for_analysis(audio_path)
                analysis_path = temp_wav
            else:
                analysis_path = audio_path
            
            sr, data = wav.read(analysis_path)
            if len(data.shape) > 1: data = data.mean(axis=1)

            yf = rfft(data)
            xf = rfftfreq(len(data), 1/sr)

            # Band energy comparison
            mid_core = np.mean(np.abs(yf[(xf >= 1000) & (xf <= 4000)]))
            high_junk = np.mean(np.abs(yf[(xf >= 10000) & (xf <= 15000)]))

            ratio = high_junk / (mid_core + 1e-9)
            # Threshold: If < threshold% energy remains at 10kHz+, it's likely junk
            return ratio < self.config.LOW_PASS_THRESHOLD
        finally:
            if temp_wav and os.path.exists(temp_wav):
                os.remove(temp_wav)

    def analyze_clarity(self, audio_path):
        """Measures post-AI muffleness to determine EQ boost.
        
        Also detects heavily compressed audio (low dynamic range) and reduces
        boost to prevent clipping.
        """
        temp_wav = None
        try:
            # Convert to WAV if not already
            if not audio_path.lower().endswith('.wav'):
                temp_wav = self._convert_to_wav_for_analysis(audio_path)
                analysis_path = temp_wav
            else:
                analysis_path = audio_path
            
            sr, data = wav.read(analysis_path)
            if len(data.shape) > 1: data = data.mean(axis=1)
            
            # Normalize to float
            if data.dtype == np.int16:
                data = data.astype(np.float32) / 32768.0
            elif data.dtype == np.int32:
                data = data.astype(np.float32) / 2147483648.0
            
            # Check dynamic range (crest factor) to detect heavy compression
            peak = np.max(np.abs(data))
            rms = np.sqrt(np.mean(data ** 2))
            crest_factor_db = 20 * np.log10((peak / (rms + 1e-10)) + 1e-10)
            
            # Debug output
            print(f"  Analysis: Peak={peak:.3f}, RMS={rms:.3f}, Crest Factor={crest_factor_db:.1f}dB")
            
            # Typical speech has 12-20dB crest factor
            # Heavily compressed audio has < 8dB crest factor
            is_heavily_compressed = crest_factor_db < 8.0
            
            if is_heavily_compressed:
                print(f"  Detected heavy compression - reducing EQ boost")
            
            yf = rfft(data)
            xf = rfftfreq(len(data), 1/sr)

            low_mid = np.mean(np.abs(yf[(xf >= 200) & (xf <= 500)]))
            presence = np.mean(np.abs(yf[(xf >= 3000) & (xf <= 6000)]))

            ratio = presence / (low_mid + 1e-9)
            
            # Reduce EQ boost for compressed audio to prevent clipping
            if is_heavily_compressed:
                if ratio < 0.06: return 3   # Reduced from 8
                if ratio < 0.15: return 2   # Reduced from 4
                return 0
            else:
                if ratio < 0.06: return 8   # Boost significantly
                if ratio < 0.15: return 4   # Boost moderately
                return 0
        finally:
            if temp_wav and os.path.exists(temp_wav):
                os.remove(temp_wav)

    def process_file(self, raw_path, output_path):
        # Ensure absolute paths to prevent the empty string error
        raw_path = os.path.abspath(raw_path)
        output_path = os.path.abspath(output_path)

        filename = os.path.basename(raw_path)
        output_dir = os.path.dirname(output_path)

        # Create output directory if it doesn't exist
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Place temp files in the output directory specifically
        base_name = os.path.splitext(filename)[0]
        temp_files = []  # Track temp files for cleanup
        
        # Track the current working file through the pipeline
        current_file = raw_path
        
        # 1. PRE-PROCESSING DECISION
        if self.config.ENABLE_PREPROCESSING:
            print("[1/5] Preprocessing...")
            temp_pre = os.path.join(output_dir, f"1-preprocessed_{base_name}.wav")
            temp_files.append(temp_pre)
            
            # Check if stereo and select best channel
            best_channel = self.select_best_channel(raw_path)
            
            # Build channel selection filter
            if best_channel is not None:
                channel_filter = f"pan=mono|c0=c{best_channel},"
            else:
                channel_filter = ""
            
            # We check the original file for frequency health.
            if self.needs_low_pass(raw_path):
                print(f"Decision: Poor HF detected. Applying {self.config.LOW_PASS_CUTOFF}Hz Low-pass pre-filter.")
                # Clip transient peaks (mic knocks), low-pass, normalize speech, then reduce by 2dB for AI headroom
                cmd = f"ffmpeg -i '{raw_path}' -af '{channel_filter}alimiter=limit=0.95:attack=0.1:release=5,lowpass=f={self.config.LOW_PASS_CUTOFF},speechnorm=expansion={self.config.SPEECHNORM_EXPANSION},volume=-2dB' -ar 48000 '{temp_pre}' -y"
            else:
                print(f"Decision: HF health is good. Preserving original bandwidth.")
                # Clip transient peaks (mic knocks) first, then normalize speech, then reduce by 2dB for AI headroom
                cmd = f"ffmpeg -i '{raw_path}' -af '{channel_filter}alimiter=limit=0.95:attack=0.1:release=5,speechnorm=expansion={self.config.SPEECHNORM_EXPANSION},volume=-2dB' -ar 48000 '{temp_pre}' -y"

            subprocess.run(cmd, shell=True, capture_output=True)
            current_file = temp_pre
        else:
            print("[1/5] Preprocessing: SKIPPED")

        # 2. AI ENHANCEMENT
        if self.config.ENABLE_AI_ENHANCEMENT:
            print("[2/5] AI Enhancement...")
            temp_ai = os.path.join(output_dir, f"2-ai_enhanced_{base_name}.wav")
            temp_files.append(temp_ai)
            output_audio = self.ai_engine(input_path=current_file, online_write=False)
            self.ai_engine.write(output_audio, output_path=temp_ai)
            current_file = temp_ai
        else:
            print("[2/5] AI Enhancement: SKIPPED")

        # 3. SPECTRAL NOISE REDUCTION
        if self.config.ENABLE_SPECTRAL_DENOISE:
            print(f"[3/5] Applying adaptive spectral noise reduction ({self.config.NOISE_REDUCTION_DB}dB, sensitivity={self.config.NOISE_SENSITIVITY})...")
            temp_denoised = os.path.join(output_dir, f"3-spectral_denoised_{base_name}.wav")
            temp_files.append(temp_denoised)
            self.adaptive_spectral_denoise(current_file, temp_denoised)
            current_file = temp_denoised
        else:
            print("[3/5] Spectral Noise Reduction: SKIPPED")
        
        # 4. NOISE GATE
        if self.config.ENABLE_NOISE_GATE:
            print(f"[4/5] Applying noise gate ({self.config.GATE_THRESHOLD_DB}dB threshold)...")
            temp_gated = os.path.join(output_dir, f"4-noise_gated_{base_name}.wav")
            temp_files.append(temp_gated)
            self.apply_noise_gate(current_file, temp_gated)
            current_file = temp_gated
        else:
            print("[4/5] Noise Gate: SKIPPED")
        
        # 5. ADAPTIVE MASTERING
        if self.config.ENABLE_MASTERING:
            print("[5/5] Mastering...")
            boost_db = self.analyze_clarity(current_file)
            # Ensure temp file exists before calling FFmpeg
            if not os.path.exists(current_file):
                print(f"CRITICAL ERROR: Input file missing: {current_file}")
                return

            # Build encoding parameters based on output format
            if self.config.OUTPUT_FORMAT == 'flac':
                codec_params = "-c:a flac -compression_level 8"
            else:  # opus
                codec_params = f"-c:a libopus -b:a {self.config.OUTPUT_BITRATE}k"
            
            master_cmd = (
                f"ffmpeg -i '{current_file}' -af "
                f"'highpass=f={self.config.HIGHPASS_FREQ}, "
                f"equalizer=f={self.config.EQ_CENTER_FREQ}:width_type=h:width={self.config.EQ_WIDTH}:g={boost_db}, "
                f"deesser=f={self.config.DEESSER_FREQ}:s={self.config.DEESSER_STRENGTH}, "
                f"loudnorm=I={self.config.LOUDNESS_TARGET}:TP={self.config.TRUE_PEAK}, "
                f"alimiter=limit=0.99:attack=1:release=50:level=disabled' "
                f"-ac 1 {codec_params} '{output_path}' -y"
            )

            # Use check=True to catch FFmpeg failures
            try:
                result = subprocess.run(master_cmd, shell=True, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
                print("FFMPEG ERROR OUTPUT:")
                print(e.stderr) # This will tell you EXACTLY why the file wasn't created
                return
        else:
            print("[5/5] Mastering: SKIPPED")
            # If mastering is skipped, just copy/convert the current file to output
            if self.config.OUTPUT_FORMAT == 'flac':
                codec_params = "-c:a flac -compression_level 8"
            else:  # opus
                codec_params = f"-c:a libopus -b:a {self.config.OUTPUT_BITRATE}k"
            
            copy_cmd = f"ffmpeg -i '{current_file}' -ac 1 {codec_params} '{output_path}' -y"
            try:
                subprocess.run(copy_cmd, shell=True, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
                print("FFMPEG ERROR OUTPUT:")
                print(e.stderr)
                return

        # Cleanup intermediate files
        for f in temp_files:
            if os.path.exists(f): os.remove(f)
        print(f"Finished: {output_path}")

    def batch_process(self, input_dir, output_dir):
        """Batch process all WAV files from input_dir to output_dir."""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        # Create output directory if it doesn't exist
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all audio files in input directory
        audio_extensions = ['*.wav', '*.WAV', '*.mp3', '*.MP3', '*.flac', '*.FLAC', 
                           '*.m4a', '*.M4A', '*.aac', '*.AAC', '*.ogg', '*.OGG', 
                           '*.opus', '*.OPUS', '*.wma', '*.WMA']
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(input_path.glob(ext))
        
        if not audio_files:
            print(f"No audio files found in {input_dir}")
            return
        
        print(f"Found {len(audio_files)} file(s) to process")
        
        for i, input_file in enumerate(audio_files, 1):
            print(f"\n[{i}/{len(audio_files)}] Processing: {input_file.name}")
            
            # Generate output filename with appropriate extension
            output_file = output_path / f"{input_file.stem}.{self.config.OUTPUT_FORMAT}"
            
            try:
                self.process_file(str(input_file), str(output_file))
            except Exception as e:
                print(f"ERROR processing {input_file.name}: {e}")
                continue
        
        print(f"\nBatch processing complete! Processed {len(audio_files)} file(s)")

# Usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Intelligent Studio Pipeline - AI-powered audio enhancement',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input/Output arguments
    parser.add_argument('--input', '-i', type=str, default='pipeline-in',
                        help='Input directory containing audio files')
    parser.add_argument('--output', '-o', type=str, default='pipeline-out',
                        help='Output directory for processed files')
    
    # Phase control arguments
    parser.add_argument('--no-preprocessing', action='store_true',
                        help='Disable preprocessing (channel selection, clipping, normalization)')
    parser.add_argument('--no-ai', action='store_true',
                        help='Disable AI enhancement')
    parser.add_argument('--no-spectral-denoise', action='store_true',
                        help='Disable spectral noise reduction')
    parser.add_argument('--no-noise-gate', action='store_true',
                        help='Disable noise gate')
    parser.add_argument('--no-mastering', action='store_true',
                        help='Disable mastering (EQ, de-essing, loudness normalization)')
    
    # Output settings
    parser.add_argument('--format', type=str, default=PipelineConfig.OUTPUT_FORMAT,
                        choices=['opus', 'flac'],
                        help='Output format: opus (lossy, small) or flac (lossless, large)')
    parser.add_argument('--bitrate', type=int, default=PipelineConfig.OUTPUT_BITRATE,
                        help='Output bitrate in kbps (only for opus format)')
    
    # Noise reduction settings
    parser.add_argument('--noise-reduction', type=float, default=PipelineConfig.NOISE_REDUCTION_DB,
                        help='Spectral noise reduction amount in dB')
    parser.add_argument('--noise-sensitivity', type=float, default=PipelineConfig.NOISE_SENSITIVITY,
                        help='Noise detection sensitivity (0-1, lower = more aggressive)')
    
    # Noise gate settings
    parser.add_argument('--gate-threshold', type=float, default=PipelineConfig.GATE_THRESHOLD_DB,
                        help='Noise gate threshold in dB FS')
    parser.add_argument('--gate-attack', type=float, default=PipelineConfig.GATE_ATTACK_MS,
                        help='Noise gate attack time in milliseconds')
    parser.add_argument('--gate-release', type=float, default=PipelineConfig.GATE_RELEASE_MS,
                        help='Noise gate release time in milliseconds')
    
    # Frequency analysis settings
    parser.add_argument('--lowpass-threshold', type=float, default=PipelineConfig.LOW_PASS_THRESHOLD,
                        help='Threshold for detecting poor HF content (0-1)')
    parser.add_argument('--lowpass-cutoff', type=int, default=PipelineConfig.LOW_PASS_CUTOFF,
                        help='Low-pass filter cutoff frequency in Hz')
    
    # Preprocessing settings
    parser.add_argument('--speechnorm-expansion', type=float, default=PipelineConfig.SPEECHNORM_EXPANSION,
                        help='Speech normalization expansion factor (higher = more aggressive)')
    
    # Mastering settings
    parser.add_argument('--highpass', type=int, default=PipelineConfig.HIGHPASS_FREQ,
                        help='Highpass filter frequency in Hz')
    parser.add_argument('--loudness', type=float, default=PipelineConfig.LOUDNESS_TARGET,
                        help='Target loudness in LUFS')
    parser.add_argument('--true-peak', type=float, default=PipelineConfig.TRUE_PEAK,
                        help='True peak limit in dB')
    
    args = parser.parse_args()
    
    # Create custom config from command-line arguments
    config = PipelineConfig()
    # Phase control
    config.ENABLE_PREPROCESSING = not args.no_preprocessing
    config.ENABLE_AI_ENHANCEMENT = not args.no_ai
    config.ENABLE_SPECTRAL_DENOISE = not args.no_spectral_denoise
    config.ENABLE_NOISE_GATE = not args.no_noise_gate
    config.ENABLE_MASTERING = not args.no_mastering
    # Output settings
    config.OUTPUT_FORMAT = args.format
    config.OUTPUT_BITRATE = args.bitrate
    config.NOISE_REDUCTION_DB = args.noise_reduction
    config.NOISE_SENSITIVITY = args.noise_sensitivity
    config.GATE_THRESHOLD_DB = args.gate_threshold
    config.GATE_ATTACK_MS = args.gate_attack
    config.GATE_RELEASE_MS = args.gate_release
    config.LOW_PASS_THRESHOLD = args.lowpass_threshold
    config.LOW_PASS_CUTOFF = args.lowpass_cutoff
    config.SPEECHNORM_EXPANSION = args.speechnorm_expansion
    config.HIGHPASS_FREQ = args.highpass
    config.LOUDNESS_TARGET = args.loudness
    config.TRUE_PEAK = args.true_peak
    
    # Initialize pipeline with custom config
    studio = IntelligentStudioPipeline(config)
    
    # Print configuration
    print("=" * 70)
    print("INTELLIGENT STUDIO PIPELINE - Configuration")
    print("=" * 70)
    print(f"Input:              {args.input}")
    print(f"Output:             {args.output}")
    print(f"Format:             {config.OUTPUT_FORMAT.upper()}")
    if config.OUTPUT_FORMAT == 'opus':
        print(f"Bitrate:            {config.OUTPUT_BITRATE} kbps")
    else:
        print(f"Compression:        FLAC level 8 (lossless)")
    print()
    print("Processing Phases:")
    print(f"  [1] Preprocessing:        {'ENABLED' if config.ENABLE_PREPROCESSING else 'DISABLED'}")
    print(f"  [2] AI Enhancement:       {'ENABLED' if config.ENABLE_AI_ENHANCEMENT else 'DISABLED'}")
    print(f"  [3] Spectral Denoise:     {'ENABLED' if config.ENABLE_SPECTRAL_DENOISE else 'DISABLED'}")
    print(f"  [4] Noise Gate:           {'ENABLED' if config.ENABLE_NOISE_GATE else 'DISABLED'}")
    print(f"  [5] Mastering:            {'ENABLED' if config.ENABLE_MASTERING else 'DISABLED'}")
    print()
    if config.ENABLE_SPECTRAL_DENOISE:
        print(f"Noise Reduction:    {config.NOISE_REDUCTION_DB} dB (sensitivity: {config.NOISE_SENSITIVITY})")
    if config.ENABLE_NOISE_GATE:
        print(f"Noise Gate:         {config.GATE_THRESHOLD_DB} dB (attack: {config.GATE_ATTACK_MS}ms, release: {config.GATE_RELEASE_MS}ms)")
    if config.ENABLE_PREPROCESSING:
        print(f"Low-pass Cutoff:    {config.LOW_PASS_CUTOFF} Hz")
    if config.ENABLE_MASTERING:
        print(f"Highpass:           {config.HIGHPASS_FREQ} Hz")
        print(f"Loudness Target:    {config.LOUDNESS_TARGET} LUFS (peak: {config.TRUE_PEAK} dB)")
    print("=" * 70)
    print()
    
    studio.batch_process(args.input, args.output)

