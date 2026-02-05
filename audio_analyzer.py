# Audio Analyzer v1.2.0 - PySide6
# Date: 05/02/2026
# Email: cs@abaye.co
# GitHub: github.com/abaye123

import os
import sys
from datetime import timedelta, datetime

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QMessageBox, QTextEdit,
    QGroupBox, QStatusBar, QDialog, QGridLayout, QSlider
)
from PySide6.QtCore import Qt, QThread, Signal, Slot, QUrl
from PySide6.QtGui import QFont
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput

import librosa
import soundfile as sf
from mutagen import File as MutagenFile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

VERSION = "v1.2.0"


def format_duration(seconds):
    """Format duration in seconds to HH:MM:SS"""
    td = timedelta(seconds=int(seconds))
    hours = td.seconds // 3600
    minutes = (td.seconds % 3600) // 60
    secs = td.seconds % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"


def format_file_size(size_bytes):
    """Format file size in bytes to human readable format"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def get_key_from_chroma(chroma):
    """Estimate musical key from chromagram"""
    # Keys mapping (0=C, 1=C#, 2=D, etc.)
    keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    # Sum across time to get average chroma
    chroma_sum = np.sum(chroma, axis=1)
    
    # Find the dominant note
    key_index = np.argmax(chroma_sum)
    
    # Determine if major or minor based on third interval
    # This is a simplified approach
    third_major = chroma_sum[(key_index + 4) % 12]
    third_minor = chroma_sum[(key_index + 3) % 12]
    
    mode = "Major" if third_major > third_minor else "Minor"
    
    return f"{keys[key_index]} {mode}"


class AnalyzerThread(QThread):
    """Thread for analyzing audio files"""
    update_console = Signal(str)
    analysis_complete = Signal(dict)
    analysis_error = Signal(str)
    
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
    
    def run(self):
        try:
            self.update_console.emit("מתחיל ניתוח קובץ...")
            
            # Load audio file
            self.update_console.emit("טוען קובץ אודיו...")
            y, sr = librosa.load(self.file_path, sr=None)
            
            results = {}
            
            # File info
            file_stats = os.stat(self.file_path)
            results['file_name'] = os.path.basename(self.file_path)
            results['file_path'] = self.file_path
            results['file_size'] = format_file_size(file_stats.st_size)
            results['file_size_bytes'] = file_stats.st_size
            
            # Duration
            duration = librosa.get_duration(y=y, sr=sr)
            results['duration'] = format_duration(duration)
            results['duration_seconds'] = duration
            
            # Sample rate
            results['sample_rate'] = f"{sr} Hz"
            
            # Channels
            sf_info = sf.info(self.file_path)
            results['channels'] = sf_info.channels
            results['channel_text'] = "Mono" if sf_info.channels == 1 else f"Stereo ({sf_info.channels} channels)"
            
            # Bit depth
            results['bit_depth'] = f"{sf_info.subtype}"
            
            # BPM (Tempo) - Analyze in segments to detect tempo changes
            self.update_console.emit("מחשב BPM...")
            
            # Analyze the entire song first
            tempo_global, beats = librosa.beat.beat_track(y=y, sr=sr)
            if isinstance(tempo_global, np.ndarray):
                tempo_global = float(tempo_global[0]) if len(tempo_global) > 0 else float(tempo_global)
            else:
                tempo_global = float(tempo_global)
            
            # Divide song into segments to detect tempo changes
            segment_duration = 30  # seconds per segment
            segment_samples = int(segment_duration * sr)
            num_segments = max(1, int(np.ceil(len(y) / segment_samples)))
            
            tempos = []
            tempo_timeline = []  # Store (time, tempo) pairs for visualization
            
            for i in range(num_segments):
                start_sample = i * segment_samples
                end_sample = min((i + 1) * segment_samples, len(y))
                segment = y[start_sample:end_sample]
                
                # Calculate time in seconds for this segment
                segment_time = i * segment_duration
                
                # Skip very short segments
                if len(segment) < sr * 5:  # At least 5 seconds
                    continue
                
                try:
                    tempo_seg, _ = librosa.beat.beat_track(y=segment, sr=sr)
                    if isinstance(tempo_seg, np.ndarray):
                        tempo_seg = float(tempo_seg[0]) if len(tempo_seg) > 0 else float(tempo_seg)
                    else:
                        tempo_seg = float(tempo_seg)
                    
                    if 40 <= tempo_seg <= 240:  # Valid BPM range
                        tempos.append(tempo_seg)
                        tempo_timeline.append((segment_time, tempo_seg))
                except:
                    continue
            
            # Detect if there are multiple distinct tempos
            if len(tempos) > 1:
                # Use clustering to find distinct tempo groups
                tempos_array = np.array(tempos)
                tempo_std = np.std(tempos_array)
                tempo_mean = np.mean(tempos_array)
                
                # If standard deviation is high, we likely have tempo changes
                if tempo_std > 10:  # Threshold for tempo variation
                    # Find unique tempos (within tolerance)
                    unique_tempos = []
                    tolerance = 5  # BPM tolerance
                    
                    for tempo in tempos_array:
                        is_unique = True
                        for unique_tempo in unique_tempos:
                            if abs(tempo - unique_tempo) < tolerance:
                                is_unique = False
                                break
                        if is_unique:
                            unique_tempos.append(tempo)
                    
                    # Sort and format
                    unique_tempos = sorted(unique_tempos)
                    
                    if len(unique_tempos) > 1:
                        # Multiple distinct tempos detected
                        bpm_str = " / ".join([f"{t:.1f}" for t in unique_tempos])
                        results['bpm'] = bpm_str
                        results['bpm_note'] = f"זוהו {len(unique_tempos)} מקצבים שונים"
                        results['has_tempo_changes'] = True
                        self.update_console.emit(f"זוהו מספר מקצבים: {bpm_str}")
                    else:
                        results['bpm'] = f"{tempo_mean:.2f}"
                        results['bpm_note'] = ""
                        results['has_tempo_changes'] = False
                else:
                    # Relatively stable tempo
                    results['bpm'] = f"{tempo_mean:.2f}"
                    results['bpm_note'] = ""
                    results['has_tempo_changes'] = False
            else:
                # Use global tempo
                results['bpm'] = f"{tempo_global:.2f}"
                results['bpm_note'] = ""
                results['has_tempo_changes'] = False
            
            results['beats_count'] = len(beats)
            results['tempo_timeline'] = tempo_timeline  # Save for visualization
            
            # Key estimation
            self.update_console.emit("מזהה מפתח מוזיקלי...")
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
            key = get_key_from_chroma(chroma)
            results['key'] = key
            
            # Extract scale (major/minor) separately for display
            key_parts = key.split(' ')
            results['note'] = key_parts[0] if len(key_parts) > 0 else key
            results['scale'] = key_parts[1] if len(key_parts) > 1 else ''
            
            # Spectral features
            self.update_console.emit("מחשב מאפיינים ספקטרליים...")
            
            # Spectral centroid (brightness)
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            results['spectral_centroid'] = f"{np.mean(spectral_centroids):.2f} Hz"
            
            # Spectral rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            results['spectral_rolloff'] = f"{np.mean(spectral_rolloff):.2f} Hz"
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            results['zero_crossing_rate'] = f"{np.mean(zcr):.4f}"
            
            # RMS energy
            rms = librosa.feature.rms(y=y)[0]
            results['rms_energy'] = f"{np.mean(rms):.4f}"
            results['rms_db'] = f"{20 * np.log10(np.mean(rms)):.2f} dB"
            
            # Metadata from file
            self.update_console.emit("קורא מטא-דאטה...")
            try:
                audio_file = MutagenFile(self.file_path)
                if audio_file is not None and audio_file.tags is not None:
                    # Try to get common tags
                    results['artist'] = str(audio_file.tags.get('artist', [''])[0]) if hasattr(audio_file.tags.get('artist', ['']), '__getitem__') else str(audio_file.tags.get('artist', ''))
                    results['title'] = str(audio_file.tags.get('title', [''])[0]) if hasattr(audio_file.tags.get('title', ['']), '__getitem__') else str(audio_file.tags.get('title', ''))
                    results['album'] = str(audio_file.tags.get('album', [''])[0]) if hasattr(audio_file.tags.get('album', ['']), '__getitem__') else str(audio_file.tags.get('album', ''))
                    results['genre'] = str(audio_file.tags.get('genre', [''])[0]) if hasattr(audio_file.tags.get('genre', ['']), '__getitem__') else str(audio_file.tags.get('genre', ''))
                    
                    # Handle different tag formats
                    if not results.get('artist'):
                        results['artist'] = str(audio_file.tags.get('TPE1', '')) or str(audio_file.tags.get('©ART', ''))
                    if not results.get('title'):
                        results['title'] = str(audio_file.tags.get('TIT2', '')) or str(audio_file.tags.get('©nam', ''))
                    if not results.get('album'):
                        results['album'] = str(audio_file.tags.get('TALB', '')) or str(audio_file.tags.get('©alb', ''))
                    if not results.get('genre'):
                        results['genre'] = str(audio_file.tags.get('TCON', '')) or str(audio_file.tags.get('©gen', ''))
                else:
                    results['artist'] = ''
                    results['title'] = ''
                    results['album'] = ''
                    results['genre'] = ''
            except:
                results['artist'] = ''
                results['title'] = ''
                results['album'] = ''
                results['genre'] = ''
            
            self.update_console.emit("הניתוח הושלם בהצלחה!")
            self.analysis_complete.emit(results)
            
        except Exception as e:
            self.update_console.emit(f"שגיאה בניתוח: {str(e)}")
            self.analysis_error.emit(str(e))


class TempoTimelineDialog(QDialog):
    """Dialog to display tempo timeline visualization"""
    def __init__(self, tempo_timeline, song_name, parent=None):
        super().__init__(parent)
        self.tempo_timeline = tempo_timeline
        self.song_name = song_name
        self.setWindowTitle(f"Timeline של קצב - {song_name}")
        self.setMinimumSize(900, 500)
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Create matplotlib figure
        fig = Figure(figsize=(10, 5), dpi=100)
        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)
        
        # Plot the tempo timeline
        ax = fig.add_subplot(111)
        
        if self.tempo_timeline and len(self.tempo_timeline) > 0:
            times = [t[0] for t in self.tempo_timeline]
            tempos = [t[1] for t in self.tempo_timeline]
            
            # Plot as step function to show clear transitions
            ax.step(times, tempos, where='post', linewidth=2, color='#0d6efd', label='BPM')
            ax.plot(times, tempos, 'o', color='#0d6efd', markersize=6)
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
            # Labels
            ax.set_xlabel('זמן (שניות)', fontsize=12, fontweight='bold')
            ax.set_ylabel('BPM', fontsize=12, fontweight='bold')
            ax.set_title(f'Timeline של שינויי קצב - {self.song_name}', fontsize=14, fontweight='bold', pad=15)
            
            # Format
            ax.set_xlim(left=0)
            y_min, y_max = ax.get_ylim()
            ax.set_ylim(y_min - 5, y_max + 5)
            
            # Add legend
            ax.legend(loc='upper right')
        else:
            ax.text(0.5, 0.5, 'אין מספיק נתונים להצגה', 
                   ha='center', va='center', fontsize=14)
        
        fig.tight_layout()
        
        # Close button
        close_btn = QPushButton("סגור")
        close_btn.clicked.connect(self.accept)
        close_btn.setMinimumHeight(40)
        layout.addWidget(close_btn)


class AboutDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("אודות")
        self.setFixedSize(400, 350)
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        title_label = QLabel("מנתח אודיו מתקדם - Audio Analyzer")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        version_label = QLabel(VERSION)
        version_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(version_label)
        
        desc_label = QLabel("ניתוח קבצי אודיו מתקדם עם זיהוי BPM, מפתח מוזיקלי ועוד")
        desc_label.setWordWrap(True)
        desc_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(desc_label)
        
        layout.addSpacing(20)
        
        info_group = QGroupBox()
        info_layout = QVBoxLayout()
        info_group.setLayout(info_layout)
        
        info_label = QLabel(
            "<b>תכונות:</b><br>"
            "• ניתוח BPM (קצב)<br>"
            "• זיהוי מפתח מוזיקלי וסולם<br>"
            "• ניתוח ספקטרלי<br>"
            "• נגן אודיו מובנה<br>"
            "• ייצוא תוצאות לקובץ MD<br>"
            "• מידע על קובץ ומטא-דאטה<br><br>"
            "נבנה עם PySide6 ו-Librosa<br><br>"
            'abaye © 2026'
        )
        info_label.setWordWrap(True)
        info_label.setAlignment(Qt.AlignCenter)
        info_layout.addWidget(info_label)
        
        layout.addWidget(info_group)
        
        layout.addStretch()
        
        close_btn = QPushButton("סגור")
        close_btn.clicked.connect(self.accept)
        close_btn.setDefault(True)
        
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.addWidget(close_btn)
        btn_layout.addStretch()
        
        layout.addLayout(btn_layout)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("מנתח אודיו מתקדם - Audio Analyzer")
        self.setMinimumSize(1000, 900)
        
        # Apply light theme
        self.set_light_mode()
        
        self.setup_ui()
        self.analyzer_thread = None
        self.current_results = None
        
        # Audio player setup
        self.audio_player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.audio_player.setAudioOutput(self.audio_output)
        self.audio_player.positionChanged.connect(self.update_position)
        self.audio_player.durationChanged.connect(self.update_duration)
        
        # Set default volume
        self.audio_output.setVolume(0.7)
    
    def set_light_mode(self):
        """Set light theme"""
        light_style = """
        QMainWindow, QDialog {
            background-color: #f5f5f5;
        }
        QWidget {
            background-color: #f5f5f5;
        }
        QGroupBox {
            border: 1px solid #cccccc;
            border-radius: 5px;
            margin-top: 1ex;
            font-weight: bold;
            padding-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top center;
            padding: 0 3px;
        }
        QPushButton {
            background-color: #0d6efd;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 8px 15px;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #0b5ed7;
        }
        QPushButton:pressed {
            background-color: #0a58ca;
        }
        QPushButton:disabled {
            background-color: #cccccc;
            color: #666666;
        }
        QTextEdit {
            background-color: white;
            border: 1px solid #cccccc;
            border-radius: 4px;
            padding: 5px;
        }
        QLabel {
            background-color: transparent;
        }
        QStatusBar {
            background-color: #e9ecef;
        }
        """
        self.setStyleSheet(light_style)
    
    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Header
        header_layout = QHBoxLayout()
        main_layout.addLayout(header_layout)
        
        title_label = QLabel("מנתח אודיו מתקדם - Audio Analyzer")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        header_layout.addWidget(title_label)
        
        header_layout.addStretch()
        
        about_btn = QPushButton("אודות")
        about_btn.clicked.connect(self.show_about)
        header_layout.addWidget(about_btn)
        
        main_layout.addSpacing(10)
        
        # File selection group
        file_group = QGroupBox("בחירת קובץ אודיו")
        file_layout = QVBoxLayout()
        file_group.setLayout(file_layout)
        
        file_btn_layout = QHBoxLayout()
        file_layout.addLayout(file_btn_layout)
        
        self.select_file_btn = QPushButton("בחר קובץ אודיו")
        self.select_file_btn.clicked.connect(self.select_audio_file)
        self.select_file_btn.setMinimumHeight(50)
        file_btn_layout.addWidget(self.select_file_btn)
        
        self.analyze_btn = QPushButton("נתח קובץ")
        self.analyze_btn.clicked.connect(self.analyze_file)
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.setMinimumHeight(50)
        file_btn_layout.addWidget(self.analyze_btn)
        
        self.selected_file_label = QLabel("לא נבחר קובץ")
        self.selected_file_label.setStyleSheet("color: #666666; padding: 5px;")
        file_layout.addWidget(self.selected_file_label)
        
        main_layout.addWidget(file_group)
        
        # Results area - two columns
        results_layout = QHBoxLayout()
        main_layout.addLayout(results_layout)
        
        # Left column - File info
        file_info_group = QGroupBox("מידע על הקובץ")
        file_info_layout = QGridLayout()
        file_info_group.setLayout(file_info_layout)
        
        row = 0
        # File name
        file_info_layout.addWidget(QLabel("<b>שם קובץ:</b>"), row, 0)
        self.info_filename = QLabel("-")
        file_info_layout.addWidget(self.info_filename, row, 1)
        row += 1
        
        # File size
        file_info_layout.addWidget(QLabel("<b>גודל:</b>"), row, 0)
        self.info_filesize = QLabel("-")
        file_info_layout.addWidget(self.info_filesize, row, 1)
        row += 1
        
        # Duration
        file_info_layout.addWidget(QLabel("<b>משך:</b>"), row, 0)
        self.info_duration = QLabel("-")
        file_info_layout.addWidget(self.info_duration, row, 1)
        row += 1
        
        # Sample rate
        file_info_layout.addWidget(QLabel("<b>תדר דגימה:</b>"), row, 0)
        self.info_sample_rate = QLabel("-")
        file_info_layout.addWidget(self.info_sample_rate, row, 1)
        row += 1
        
        # Channels
        file_info_layout.addWidget(QLabel("<b>ערוצים:</b>"), row, 0)
        self.info_channels = QLabel("-")
        file_info_layout.addWidget(self.info_channels, row, 1)
        row += 1
        
        # Bit depth
        file_info_layout.addWidget(QLabel("<b>עומק סיביות:</b>"), row, 0)
        self.info_bit_depth = QLabel("-")
        file_info_layout.addWidget(self.info_bit_depth, row, 1)
        row += 1
        
        file_info_layout.setRowStretch(row, 1)
        results_layout.addWidget(file_info_group)
        
        # Right column - Musical analysis
        musical_group = QGroupBox("ניתוח מוזיקלי")
        musical_layout = QGridLayout()
        musical_group.setLayout(musical_layout)
        
        row = 0
        # BPM
        musical_layout.addWidget(QLabel("<b>BPM (קצב):</b>"), row, 0)
        self.info_bpm = QLabel("-")
        self.info_bpm.setStyleSheet("color: #0d6efd; font-size: 14px; font-weight: bold;")
        musical_layout.addWidget(self.info_bpm, row, 1)
        row += 1
        
        # BPM Note (for multiple tempos)
        self.info_bpm_note = QLabel("")
        self.info_bpm_note.setStyleSheet("color: #666666; font-size: 11px;")
        self.info_bpm_note.setWordWrap(True)
        musical_layout.addWidget(self.info_bpm_note, row, 0, 1, 2)
        row += 1
        
        # Note
        musical_layout.addWidget(QLabel("<b>מפתח:</b>"), row, 0)
        self.info_note = QLabel("-")
        self.info_note.setStyleSheet("color: #0d6efd; font-size: 14px; font-weight: bold;")
        musical_layout.addWidget(self.info_note, row, 1)
        row += 1
        
        # Scale
        musical_layout.addWidget(QLabel("<b>סולם:</b>"), row, 0)
        self.info_scale = QLabel("-")
        self.info_scale.setStyleSheet("color: #0d6efd; font-size: 14px; font-weight: bold;")
        musical_layout.addWidget(self.info_scale, row, 1)
        row += 1
        
        # Beats count
        musical_layout.addWidget(QLabel("<b>מספר פעימות:</b>"), row, 0)
        self.info_beats = QLabel("-")
        musical_layout.addWidget(self.info_beats, row, 1)
        row += 1
        
        # Spectral centroid
        musical_layout.addWidget(QLabel("<b>Spectral Centroid:</b>"), row, 0)
        self.info_spectral_centroid = QLabel("-")
        musical_layout.addWidget(self.info_spectral_centroid, row, 1)
        row += 1
        
        # RMS Energy
        musical_layout.addWidget(QLabel("<b>RMS Energy:</b>"), row, 0)
        self.info_rms = QLabel("-")
        musical_layout.addWidget(self.info_rms, row, 1)
        row += 1
        
        # Zero crossing rate
        musical_layout.addWidget(QLabel("<b>Zero Crossing Rate:</b>"), row, 0)
        self.info_zcr = QLabel("-")
        musical_layout.addWidget(self.info_zcr, row, 1)
        row += 1
        
        musical_layout.setRowStretch(row, 1)
        results_layout.addWidget(musical_group)
        
        # Metadata group
        metadata_group = QGroupBox("מטא-דאטה")
        metadata_layout = QGridLayout()
        metadata_group.setLayout(metadata_layout)
        
        row = 0
        # Artist
        metadata_layout.addWidget(QLabel("<b>אמן:</b>"), row, 0)
        self.info_artist = QLabel("-")
        metadata_layout.addWidget(self.info_artist, row, 1)
        row += 1
        
        # Title
        metadata_layout.addWidget(QLabel("<b>כותרת:</b>"), row, 0)
        self.info_title = QLabel("-")
        metadata_layout.addWidget(self.info_title, row, 1)
        row += 1
        
        # Album
        metadata_layout.addWidget(QLabel("<b>אלבום:</b>"), row, 0)
        self.info_album = QLabel("-")
        metadata_layout.addWidget(self.info_album, row, 1)
        row += 1
        
        # Genre
        metadata_layout.addWidget(QLabel("<b>ז'אנר:</b>"), row, 0)
        self.info_genre = QLabel("-")
        metadata_layout.addWidget(self.info_genre, row, 1)
        row += 1
        
        metadata_layout.setRowStretch(row, 1)
        main_layout.addWidget(metadata_group)
        
        # Audio player controls
        player_group = QGroupBox("נגן אודיו")
        player_layout = QVBoxLayout()
        player_group.setLayout(player_layout)
        
        # Control buttons
        control_btn_layout = QHBoxLayout()
        player_layout.addLayout(control_btn_layout)
        
        self.play_btn = QPushButton("▶ נגן")
        self.play_btn.clicked.connect(self.play_audio)
        self.play_btn.setEnabled(False)
        control_btn_layout.addWidget(self.play_btn)
        
        self.pause_btn = QPushButton("⏸ השהה")
        self.pause_btn.clicked.connect(self.pause_audio)
        self.pause_btn.setEnabled(False)
        control_btn_layout.addWidget(self.pause_btn)
        
        self.stop_btn = QPushButton("⏹ עצור")
        self.stop_btn.clicked.connect(self.stop_audio)
        self.stop_btn.setEnabled(False)
        control_btn_layout.addWidget(self.stop_btn)
        
        # Position slider and time labels
        position_layout = QHBoxLayout()
        player_layout.addLayout(position_layout)
        
        self.current_time_label = QLabel("00:00")
        position_layout.addWidget(self.current_time_label)
        
        self.position_slider = QSlider(Qt.Horizontal)
        self.position_slider.sliderMoved.connect(self.set_position)
        self.position_slider.setEnabled(False)
        position_layout.addWidget(self.position_slider)
        
        self.total_time_label = QLabel("00:00")
        position_layout.addWidget(self.total_time_label)
        
        # Volume control
        volume_layout = QHBoxLayout()
        player_layout.addLayout(volume_layout)
        
        volume_layout.addWidget(QLabel("🔊 עוצמה:"))
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setMinimum(0)
        self.volume_slider.setMaximum(100)
        self.volume_slider.setValue(70)
        self.volume_slider.valueChanged.connect(self.set_volume)
        volume_layout.addWidget(self.volume_slider)
        self.volume_label = QLabel("70%")
        volume_layout.addWidget(self.volume_label)
        
        main_layout.addWidget(player_group)
        
        # Console output group
        console_group = QGroupBox("לוג ניתוח")
        console_layout = QVBoxLayout()
        console_group.setLayout(console_layout)
        
        console_header = QHBoxLayout()
        console_layout.addLayout(console_header)
        
        clear_btn = QPushButton("נקה")
        clear_btn.clicked.connect(self.clear_console)
        console_header.addWidget(clear_btn)
        
        self.timeline_btn = QPushButton("📊 הצג Timeline של קצב")
        self.timeline_btn.clicked.connect(self.show_tempo_timeline)
        self.timeline_btn.setEnabled(False)
        console_header.addWidget(self.timeline_btn)
        
        export_btn = QPushButton("ייצוא לקובץ MD")
        export_btn.clicked.connect(self.export_results)
        console_header.addWidget(export_btn)
        
        console_header.addStretch()
        
        self.console_text = QTextEdit()
        self.console_text.setReadOnly(True)
        self.console_text.setFont(QFont("Consolas", 10))
        self.console_text.setMaximumHeight(150)
        console_layout.addWidget(self.console_text)
        
        main_layout.addWidget(console_group)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("מוכן לניתוח קבצי אודיו")
    
    def select_audio_file(self):
        """Open file dialog to select audio file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "בחר קובץ אודיו",
            os.path.expanduser("~"),
            "Audio Files (*.mp3 *.wav *.flac *.ogg *.m4a *.wma *.aac);;All Files (*.*)"
        )
        
        if file_path:
            self.selected_file_path = file_path
            self.selected_file_label.setText(f"נבחר: {os.path.basename(file_path)}")
            self.analyze_btn.setEnabled(True)
            self.update_log(f"נבחר קובץ: {file_path}")
            
            # Clear previous results
            self.clear_results()
            
            # Load audio file into player
            self.audio_player.setSource(QUrl.fromLocalFile(file_path))
            self.play_btn.setEnabled(True)
            self.stop_btn.setEnabled(True)
            self.position_slider.setEnabled(True)
    
    def analyze_file(self):
        """Start analyzing the selected audio file"""
        if not hasattr(self, 'selected_file_path'):
            QMessageBox.warning(self, "שגיאה", "נא לבחור קובץ אודיו תחילה")
            return
        
        self.analyze_btn.setEnabled(False)
        self.select_file_btn.setEnabled(False)
        
        self.update_log("מתחיל ניתוח...")
        self.status_bar.showMessage("מנתח קובץ אודיו...")
        
        self.analyzer_thread = AnalyzerThread(self.selected_file_path)
        self.analyzer_thread.update_console.connect(self.update_log)
        self.analyzer_thread.analysis_complete.connect(self.display_results)
        self.analyzer_thread.analysis_error.connect(self.handle_error)
        self.analyzer_thread.start()
    
    @Slot(dict)
    def display_results(self, results):
        """Display analysis results"""
        self.current_results = results
        
        # File info
        self.info_filename.setText(results.get('file_name', '-'))
        self.info_filesize.setText(results.get('file_size', '-'))
        self.info_duration.setText(results.get('duration', '-'))
        self.info_sample_rate.setText(results.get('sample_rate', '-'))
        self.info_channels.setText(results.get('channel_text', '-'))
        self.info_bit_depth.setText(results.get('bit_depth', '-'))
        
        # Musical analysis
        self.info_bpm.setText(results.get('bpm', '-'))
        self.info_bpm_note.setText(results.get('bpm_note', ''))
        self.info_note.setText(results.get('note', '-'))
        self.info_scale.setText(results.get('scale', '-'))
        self.info_beats.setText(str(results.get('beats_count', '-')))
        self.info_spectral_centroid.setText(results.get('spectral_centroid', '-'))
        self.info_rms.setText(results.get('rms_db', '-'))
        self.info_zcr.setText(results.get('zero_crossing_rate', '-'))
        
        # Metadata
        self.info_artist.setText(results.get('artist', '-') or '-')
        self.info_title.setText(results.get('title', '-') or '-')
        self.info_album.setText(results.get('album', '-') or '-')
        self.info_genre.setText(results.get('genre', '-') or '-')
        
        # Enable timeline button if we have tempo data
        if results.get('tempo_timeline'):
            self.timeline_btn.setEnabled(True)
        
        self.analyze_btn.setEnabled(True)
        self.select_file_btn.setEnabled(True)
        self.status_bar.showMessage("הניתוח הושלם בהצלחה")
    
    def show_tempo_timeline(self):
        """Show tempo timeline visualization"""
        if not self.current_results or not self.current_results.get('tempo_timeline'):
            QMessageBox.warning(self, "שגיאה", "אין נתוני timeline זמינים")
            return
        
        song_name = self.current_results.get('file_name', 'Unknown')
        tempo_timeline = self.current_results.get('tempo_timeline', [])
        
        dialog = TempoTimelineDialog(tempo_timeline, song_name, self)
        dialog.exec()
    
    @Slot(str)
    def handle_error(self, error_msg):
        """Handle analysis error"""
        QMessageBox.critical(self, "שגיאת ניתוח", f"אירעה שגיאה בניתוח הקובץ:\n\n{error_msg}")
        self.analyze_btn.setEnabled(True)
        self.select_file_btn.setEnabled(True)
        self.status_bar.showMessage("ניתוח נכשל")
    
    def clear_results(self):
        """Clear all result labels"""
        self.info_filename.setText("-")
        self.info_filesize.setText("-")
        self.info_duration.setText("-")
        self.info_sample_rate.setText("-")
        self.info_channels.setText("-")
        self.info_bit_depth.setText("-")
        self.info_bpm.setText("-")
        self.info_bpm_note.setText("")
        self.info_note.setText("-")
        self.info_scale.setText("-")
        self.info_beats.setText("-")
        self.info_spectral_centroid.setText("-")
        self.info_rms.setText("-")
        self.info_zcr.setText("-")
        self.info_artist.setText("-")
        self.info_title.setText("-")
        self.info_album.setText("-")
        self.info_genre.setText("-")
    
    def clear_console(self):
        """Clear console output"""
        self.console_text.clear()
    
    def play_audio(self):
        """Play audio file"""
        self.audio_player.play()
        self.play_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.update_log("מנגן אודיו...")
    
    def pause_audio(self):
        """Pause audio playback"""
        self.audio_player.pause()
        self.play_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.update_log("השהיית נגינה")
    
    def stop_audio(self):
        """Stop audio playback"""
        self.audio_player.stop()
        self.play_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.position_slider.setValue(0)
        self.update_log("עצירת נגינה")
    
    def set_position(self, position):
        """Set playback position"""
        self.audio_player.setPosition(position)
    
    def update_position(self, position):
        """Update position slider and time label"""
        self.position_slider.blockSignals(True)
        self.position_slider.setValue(position)
        self.position_slider.blockSignals(False)
        
        # Update time label
        seconds = position // 1000
        minutes = seconds // 60
        seconds = seconds % 60
        self.current_time_label.setText(f"{minutes:02d}:{seconds:02d}")
    
    def update_duration(self, duration):
        """Update duration slider and time label"""
        self.position_slider.setRange(0, duration)
        
        # Update time label
        seconds = duration // 1000
        minutes = seconds // 60
        seconds = seconds % 60
        self.total_time_label.setText(f"{minutes:02d}:{seconds:02d}")
    
    def set_volume(self, value):
        """Set audio volume"""
        self.audio_output.setVolume(value / 100.0)
        self.volume_label.setText(f"{value}%")
    
    def export_results(self):
        """Export analysis results to MD file"""
        if not self.current_results:
            QMessageBox.warning(self, "שגיאה", "אין תוצאות לייצוא. נא לנתח קובץ תחילה.")
            return
        
        # Set default save location to Downloads folder
        downloads_folder = os.path.join(os.path.expanduser("~"), "Downloads")
        if not os.path.exists(downloads_folder):
            downloads_folder = os.path.expanduser("~")
        
        default_filename = f"{os.path.splitext(self.current_results.get('file_name', 'results'))[0]}_analysis.md"
        default_path = os.path.join(downloads_folder, default_filename)
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "שמור תוצאות",
            default_path,
            "Markdown Files (*.md);;All Files (*.*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write("# ניתוח קובץ אודיו\n\n")
                    f.write(f"**תאריך ניתוח:** {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n\n")
                    
                    f.write("## מידע על הקובץ\n\n")
                    f.write(f"- **שם קובץ:** {self.current_results.get('file_name', '-')}\n")
                    f.write(f"- **נתיב:** {self.current_results.get('file_path', '-')}\n")
                    f.write(f"- **גודל:** {self.current_results.get('file_size', '-')}\n")
                    f.write(f"- **משך:** {self.current_results.get('duration', '-')}\n")
                    f.write(f"- **תדר דגימה:** {self.current_results.get('sample_rate', '-')}\n")
                    f.write(f"- **ערוצים:** {self.current_results.get('channel_text', '-')}\n")
                    f.write(f"- **עומק סיביות:** {self.current_results.get('bit_depth', '-')}\n\n")
                    
                    f.write("## ניתוח מוזיקלי\n\n")
                    f.write(f"- **BPM (קצב):** {self.current_results.get('bpm', '-')}\n")
                    if self.current_results.get('bpm_note'):
                        f.write(f"  - *{self.current_results.get('bpm_note')}*\n")
                    f.write(f"- **מפתח:** {self.current_results.get('note', '-')}\n")
                    f.write(f"- **סולם:** {self.current_results.get('scale', '-')}\n")
                    f.write(f"- **מספר פעימות:** {self.current_results.get('beats_count', '-')}\n")
                    f.write(f"- **Spectral Centroid:** {self.current_results.get('spectral_centroid', '-')}\n")
                    f.write(f"- **RMS Energy:** {self.current_results.get('rms_db', '-')}\n")
                    f.write(f"- **Zero Crossing Rate:** {self.current_results.get('zero_crossing_rate', '-')}\n\n")
                    
                    f.write("## מטא-דאטה\n\n")
                    artist = self.current_results.get('artist', '-') or '-'
                    title = self.current_results.get('title', '-') or '-'
                    album = self.current_results.get('album', '-') or '-'
                    genre = self.current_results.get('genre', '-') or '-'
                    
                    f.write(f"- **אמן:** {artist}\n")
                    f.write(f"- **כותרת:** {title}\n")
                    f.write(f"- **אלבום:** {album}\n")
                    f.write(f"- **ז'אנר:** {genre}\n\n")
                    
                    f.write("---\n\n")
                    f.write("*נוצר באמצעות מנתח אודיו מתקדם - Audio Analyzer*\n")
                
                self.update_log(f"התוצאות יוצאו בהצלחה: {file_path}")
                QMessageBox.information(self, "הצלחה", "התוצאות יוצאו בהצלחה לקובץ MD")
            except Exception as e:
                self.update_log(f"שגיאה בייצוא: {str(e)}")
                QMessageBox.critical(self, "שגיאה", f"שגיאה בייצוא התוצאות:\n\n{str(e)}")
    
    @Slot(str)
    def update_log(self, message):
        """Update console log"""
        self.console_text.append(message)
        scrollbar = self.console_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def show_about(self):
        """Show about dialog"""
        dialog = AboutDialog(self)
        dialog.exec()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setLayoutDirection(Qt.RightToLeft)
    app.setStyle("Fusion")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())
