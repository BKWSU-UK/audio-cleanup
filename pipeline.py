import os
import subprocess
import numpy as np
import scipy.io.wavfile as wav
from scipy.fft import rfft, rfftfreq, irfft
from scipy.signal import stft, istft, lfilter, get_window
from pathlib import Path
import argparse
from clearvoice import ClearVoice

try:
    import cupy as cp
    import cupy.fft
    CUPY_AVAILABLE = True
except (ImportError, Exception):
    CUPY_AVAILABLE = False

try:
    import pyfftw
    pyfftw.interfaces.cache.enable()
    PYFFTW_AVAILABLE = True
except ImportError:
    PYFFTW_AVAILABLE = False

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

    def _load_analysis_sample(self, audio_path, sample_seconds=30):
        """Load a sample of audio for analysis, converting from any format.
        
        Extracts a segment from the middle of the file to avoid intros/outros.
        Returns (sample_rate, data) where data retains its original channel layout.
        """
        import tempfile
        temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_wav.close()
        
        try:
            cmd = f"ffmpeg -i '{audio_path}' -ar 48000 '{temp_wav.name}' -y"
            result = subprocess.run(cmd, shell=True, capture_output=True)
            if result.returncode != 0:
                raise RuntimeError(f"Failed to convert {audio_path} to WAV for analysis")
            
            sr, data = wav.read(temp_wav.name)
        finally:
            if os.path.exists(temp_wav.name):
                os.remove(temp_wav.name)
        
        # Extract a sample from the middle of the file
        total_samples = data.shape[0]
        sample_len = int(sr * sample_seconds)
        
        if total_samples > sample_len:
            start = (total_samples - sample_len) // 2
            data = data[start:start + sample_len]
        
        return sr, data

    def select_best_channel(self, sr, data):
        """Analyze stereo audio and select the channel with better quality.
        
        Args:
            sr: Sample rate
            data: Audio data array (from _load_analysis_sample)
        
        Returns the best channel index (0 or 1) or None if mono.
        """
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

    def _stft(self, data, nperseg, noverlap):
        """Compute STFT with three-tier acceleration: CuPy GPU → pyfftw CPU → NumPy.
        
        Uses real-valued FFT and float32 throughout for speed.
        Returns (Zxx, window, on_gpu) where Zxx is complex64 spectrogram (freq_bins, n_frames).
        When CuPy is available, Zxx remains on GPU to avoid unnecessary transfers.
        """
        hop = nperseg - noverlap
        window = get_window('hann', nperseg).astype(np.float32)
        
        # Pad signal so all samples are covered
        n_frames = 1 + (len(data) - nperseg) // hop
        padded_len = nperseg + (n_frames - 1) * hop
        if padded_len > len(data):
            data = np.pad(data, (0, padded_len - len(data)), mode='constant')
        
        # Build frame matrix via stride tricks (zero-copy view)
        shape = (n_frames, nperseg)
        strides = (data.strides[0] * hop, data.strides[0])
        frames = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)
        
        # Apply window
        windowed = frames * window
        
        if CUPY_AVAILABLE:
            Zxx = cp.fft.rfft(cp.asarray(windowed), axis=1).astype(cp.complex64)
            return Zxx.T, window, True
        elif PYFFTW_AVAILABLE:
            Zxx = pyfftw.interfaces.numpy_fft.rfft(windowed, axis=1, threads=-1).astype(np.complex64)
        else:
            Zxx = np.fft.rfft(windowed, axis=1).astype(np.complex64)
        
        return Zxx.T, window, False  # (freq_bins, n_frames)

    def _istft(self, Zxx, window, nperseg, noverlap, original_len, on_gpu=False):
        """Compute inverse STFT with three-tier acceleration: CuPy GPU → pyfftw CPU → NumPy.
        
        Uses overlap-add reconstruction with precomputed window normalization.
        Accepts GPU arrays when on_gpu=True.
        """
        hop = nperseg - noverlap
        Zxx_T = Zxx.T  # (n_frames, freq_bins)
        
        if on_gpu and CUPY_AVAILABLE:
            frames = cp.asnumpy(cp.fft.irfft(Zxx_T, n=nperseg, axis=1).astype(cp.float32))
        elif PYFFTW_AVAILABLE:
            frames = pyfftw.interfaces.numpy_fft.irfft(Zxx_T, n=nperseg, axis=1, threads=-1).astype(np.float32)
        else:
            frames = np.fft.irfft(Zxx_T, n=nperseg, axis=1).astype(np.float32)
        
        frames *= window
        
        n_frames = frames.shape[0]
        output_len = nperseg + (n_frames - 1) * hop
        output = np.zeros(output_len, dtype=np.float32)
        
        # Precompute window normalization (constant for given window/hop)
        window_sq = window ** 2
        window_sum = np.zeros(output_len, dtype=np.float32)
        
        for i in range(n_frames):
            start = i * hop
            output[start:start + nperseg] += frames[i]
            window_sum[start:start + nperseg] += window_sq
        
        # Normalize by window overlap
        nonzero = window_sum > 1e-10
        output[nonzero] /= window_sum[nonzero]
        
        return output[:original_len]

    def adaptive_spectral_denoise(self, wav_path, output_path, noise_reduction_db=None, sensitivity=None):
        """Self-calibrating spectral noise reduction using automatic noise profile detection.
        
        This works similarly to Audacity's noise reduction:
        1. Detects quiet regions (likely noise-only)
        2. Builds a noise profile from those regions
        3. Applies spectral subtraction to remove the noise
        
        Uses pyfftw for multi-threaded FFT when available, with float32 precision
        throughout for reduced memory bandwidth.
        
        For large files (>10 minutes), processes audio in chunks to avoid memory issues.
        
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
            data = data.mean(axis=1)
        
        # Normalize to float32 (half the memory bandwidth of float64)
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
        elif data.dtype != np.float32:
            data = data.astype(np.float32)
        
        # STFT parameters
        nperseg = 2048
        noverlap = nperseg // 2
        original_len = len(data)
        
        # Check if file is large (>10 minutes) and needs chunked processing
        duration_minutes = len(data) / sr / 60
        chunk_size_samples = sr * 300  # 5 minutes per chunk
        
        if duration_minutes > 10:
            print(f"  Large file detected ({duration_minutes:.1f} minutes) - using chunked processing...")
            
            # Step 1: Build noise profile from a representative sample (first 5 minutes)
            sample_data = data[:min(chunk_size_samples, len(data))]
            Zxx_sample, window, on_gpu = self._stft(sample_data, nperseg, noverlap)
            xp = cp if on_gpu else np
            
            magnitude_sample = xp.abs(Zxx_sample)
            frame_energy = xp.sum(magnitude_sample ** 2, axis=0)
            energy_threshold = xp.percentile(frame_energy, sensitivity * 100)
            noise_frames = frame_energy < energy_threshold
            
            noise_count = int(xp.sum(noise_frames))
            if noise_count > 0:
                noise_profile = xp.mean(magnitude_sample[:, noise_frames], axis=1)
            else:
                noise_profile = xp.percentile(magnitude_sample, 10, axis=1)
            
            # Convert noise profile to CPU for reuse across chunks
            if on_gpu:
                noise_profile = cp.asnumpy(noise_profile)
            
            # Step 2: Process audio in chunks
            output_audio = np.zeros(original_len, dtype=np.float32)
            overlap_samples = nperseg * 2  # Overlap between chunks to avoid artifacts
            
            num_chunks = int(np.ceil(len(data) / chunk_size_samples))
            for i in range(num_chunks):
                start_idx = max(0, i * chunk_size_samples - overlap_samples)
                end_idx = min(len(data), (i + 1) * chunk_size_samples + overlap_samples)
                chunk = data[start_idx:end_idx]
                
                # Process chunk
                Zxx_chunk, _, on_gpu = self._stft(chunk, nperseg, noverlap)
                xp = cp if on_gpu else np
                
                # Upload noise profile to GPU if needed
                noise_profile_xp = xp.asarray(noise_profile) if on_gpu else noise_profile
                
                magnitude = xp.abs(Zxx_chunk)
                reduction_factor = xp.float32(10 ** (noise_reduction_db / 20))
                noise_scaled = (noise_profile_xp * reduction_factor)[:, xp.newaxis]
                
                magnitude -= noise_scaled
                spectral_floor = xp.abs(Zxx_chunk) * xp.float32(0.01)
                xp.maximum(magnitude, spectral_floor, out=magnitude)
                
                abs_Zxx = xp.abs(Zxx_chunk)
                abs_Zxx[abs_Zxx < 1e-10] = 1e-10
                Zxx_chunk *= (magnitude / abs_Zxx)
                
                chunk_cleaned = self._istft(Zxx_chunk, window, nperseg, noverlap, len(chunk), on_gpu)
                
                # Calculate which portion of the cleaned chunk to use
                write_start = i * chunk_size_samples
                write_end = min(original_len, write_start + chunk_size_samples)
                write_length = write_end - write_start
                
                if i == 0:
                    # First chunk: skip initial overlap (it doesn't exist), take from start
                    chunk_start = 0
                    chunk_end = min(write_length, len(chunk_cleaned))
                elif i == num_chunks - 1:
                    # Last chunk: skip the overlap region at the start
                    chunk_start = overlap_samples
                    chunk_end = min(chunk_start + write_length, len(chunk_cleaned))
                else:
                    # Middle chunks: skip overlap at start
                    chunk_start = overlap_samples
                    chunk_end = min(chunk_start + write_length, len(chunk_cleaned))
                
                # Copy the valid portion
                valid_length = min(write_length, chunk_end - chunk_start)
                output_audio[write_start:write_start + valid_length] = chunk_cleaned[chunk_start:chunk_start + valid_length]
                
                print(f"    Processed chunk {i+1}/{num_chunks}")
            
            audio_cleaned = output_audio
        else:
            # Original single-pass processing for smaller files
            Zxx, window, on_gpu = self._stft(data, nperseg, noverlap)
            xp = cp if on_gpu else np
            
            magnitude = xp.abs(Zxx)
            frame_energy = xp.sum(magnitude ** 2, axis=0)
            energy_threshold = xp.percentile(frame_energy, sensitivity * 100)
            noise_frames = frame_energy < energy_threshold
            
            noise_count = int(xp.sum(noise_frames))
            if noise_count > 0:
                noise_profile = xp.mean(magnitude[:, noise_frames], axis=1)
            else:
                noise_profile = xp.percentile(magnitude, 10, axis=1)
            
            first_2sec_frames = int(2.0 * sr / (nperseg - noverlap))
            if first_2sec_frames < Zxx.shape[1]:
                initial_noise = xp.percentile(magnitude[:, :first_2sec_frames], 20, axis=1)
                xp.maximum(noise_profile, initial_noise * 0.8, out=noise_profile)
            
            reduction_factor = xp.float32(10 ** (noise_reduction_db / 20))
            noise_scaled = (noise_profile * reduction_factor)[:, xp.newaxis]
            
            magnitude -= noise_scaled
            spectral_floor = xp.abs(Zxx) * xp.float32(0.01)
            xp.maximum(magnitude, spectral_floor, out=magnitude)
            
            abs_Zxx = xp.abs(Zxx)
            abs_Zxx[abs_Zxx < 1e-10] = 1e-10
            Zxx *= (magnitude / abs_Zxx)
            
            audio_cleaned = self._istft(Zxx, window, nperseg, noverlap, original_len, on_gpu)
        
        # Convert back to int16
        np.clip(audio_cleaned, -1.0, 1.0, out=audio_cleaned)
        wav.write(output_path, sr, (audio_cleaned * 32767).astype(np.int16))
        
        return output_path

    def apply_noise_gate(self, wav_path, output_path, threshold_db=None, attack_ms=None, release_ms=None):
        """Apply a noise gate to remove residual hiss during quiet passages.
        
        Uses streaming processing for large files to avoid memory issues.
        
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
        
        # Normalize to float32
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
        elif data.dtype != np.float32:
            data = data.astype(np.float32)
        
        # Convert threshold from dB to linear
        threshold_linear = np.float32(10 ** (threshold_db / 20))
        
        # Calculate envelope using simple moving RMS
        window_size = int(sr * 0.01)  # 10ms windows
        hop_size = window_size // 2
        
        # For large files, compute envelope in chunks to avoid memory issues
        duration_minutes = len(data) / sr / 60
        if duration_minutes > 30:
            print(f"  Large file ({duration_minutes:.1f} minutes) - using streaming gate calculation...")
            
            # Compute envelope in chunks
            chunk_size = sr * 60  # 1 minute chunks
            envelope_list = []
            frame_starts_list = []
            
            for chunk_start in range(0, len(data), chunk_size):
                chunk_end = min(chunk_start + chunk_size + window_size, len(data))
                chunk = data[chunk_start:chunk_end]
                
                # Pad chunk
                padded_chunk = np.pad(chunk, (window_size // 2, window_size // 2), mode='reflect')
                
                # Compute envelope for this chunk using strided view (memory efficient)
                n_frames = (len(chunk) + hop_size - 1) // hop_size
                envelope_chunk = np.zeros(n_frames, dtype=np.float32)
                
                for i in range(n_frames):
                    start_idx = i * hop_size
                    end_idx = min(start_idx + window_size, len(padded_chunk))
                    window_data = padded_chunk[start_idx:end_idx]
                    envelope_chunk[i] = np.sqrt(np.mean(window_data ** 2))
                
                envelope_list.append(envelope_chunk)
                frame_starts_list.append(np.arange(chunk_start, chunk_start + len(chunk), hop_size))
            
            envelope = np.concatenate(envelope_list)
            frame_starts = np.concatenate(frame_starts_list)
        else:
            # Original fast method for smaller files
            padded_data = np.pad(data, (window_size // 2, window_size // 2), mode='reflect')
            squared = padded_data.astype(np.float32) ** 2
            cumsum = np.concatenate(([0], np.cumsum(squared)))
            frame_starts = np.arange(0, len(data), hop_size)
            frame_ends = np.minimum(frame_starts + window_size, len(cumsum) - 1)
            envelope = np.sqrt((cumsum[frame_ends] - cumsum[frame_starts]) / window_size)
        
        # Create gate signal (1 = open, 0 = closed)
        gate_signal = (envelope > threshold_linear).astype(np.float32)
        
        # Vectorized attack/release smoothing using lfilter
        attack_alpha = np.float32(1.0 - np.exp(-1.0 / max(1, attack_ms * sr / (1000 * hop_size))))
        release_alpha = np.float32(1.0 - np.exp(-1.0 / max(1, release_ms * sr / (1000 * hop_size))))
        
        # Apply release (slow decay) first, then attack (fast rise)
        release_smoothed = lfilter([release_alpha], [1, -(1 - release_alpha)], gate_signal).astype(np.float32)
        smoothed_gate = lfilter([attack_alpha], [1, -(1 - attack_alpha)], 
                                np.maximum(gate_signal, release_smoothed)).astype(np.float32)
        
        # Interpolate gate signal to match audio length
        gate_full = np.interp(
            np.arange(len(data)),
            frame_starts[:len(smoothed_gate)],
            smoothed_gate
        ).astype(np.float32)
        
        # Apply gate to audio
        data *= gate_full
        
        # Convert back to int16
        np.clip(data, -1.0, 1.0, out=data)
        wav.write(output_path, sr, (data * 32767).astype(np.int16))
        
        return output_path

    def needs_low_pass(self, sr, data):
        """Analyzes if high-frequencies are speech or codec junk.
        
        Args:
            sr: Sample rate
            data: Audio data array (from _load_analysis_sample), mono or stereo
        """
        if len(data.shape) > 1:
            data = data.mean(axis=1)

        yf = rfft(data)
        xf = rfftfreq(len(data), 1/sr)

        # Band energy comparison
        mid_core = np.mean(np.abs(yf[(xf >= 1000) & (xf <= 4000)]))
        high_junk = np.mean(np.abs(yf[(xf >= 10000) & (xf <= 15000)]))

        ratio = high_junk / (mid_core + 1e-9)
        # Threshold: If < threshold% energy remains at 10kHz+, it's likely junk
        return ratio < self.config.LOW_PASS_THRESHOLD

    def analyze_clarity(self, wav_path):
        """Measures post-AI muffleness to determine EQ boost.
        
        Also detects heavily compressed audio (low dynamic range) and reduces
        boost to prevent clipping.
        
        Args:
            wav_path: Path to a WAV file (always WAV at this pipeline stage)
        """
        sr, data = wav.read(wav_path)
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
            
            # Load a 30-second sample once for all pre-processing analysis
            sr_sample, data_sample = self._load_analysis_sample(raw_path)
            
            # Check if stereo and select best channel
            best_channel = self.select_best_channel(sr_sample, data_sample)
            
            # Build channel selection filter
            if best_channel is not None:
                channel_filter = f"pan=mono|c0=c{best_channel},"
            else:
                channel_filter = ""
            
            # We check the original file for frequency health.
            if self.needs_low_pass(sr_sample, data_sample):
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
    fft_backend = "GPU (CuPy/CUDA)" if CUPY_AVAILABLE else "CPU (pyfftw multi-threaded)" if PYFFTW_AVAILABLE else "CPU (NumPy single-threaded)"
    print(f"FFT Acceleration:   {fft_backend}")
    print("=" * 70)
    print()
    
    studio.batch_process(args.input, args.output)

