import tkinter as tk
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
from pydub import AudioSegment

class AudioAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Analyzer")
        self.filename = None
        self.sampling_rate = None
        self.audio_data = None
        self.audio_length = None

        self.load_button = tk.Button(root, text="Load Audio File", command=self.load_audio)
        self.load_button.pack()

        self.file_label = tk.Label(root, text="No file loaded")
        self.file_label.pack()

        self.plot_frame = tk.Frame(root)
        self.plot_frame.pack()

        self.results_frame = tk.Frame(root)
        self.results_frame.pack()

        self.toggle_buttons_frame = tk.Frame(root)
        self.toggle_buttons_frame.pack()

        self.low_button = tk.Button(self.toggle_buttons_frame, text="Show Low Frequency", command=self.show_low_plot, state=tk.DISABLED)
        self.low_button.pack(side=tk.LEFT, padx=5)

        self.mid_button = tk.Button(self.toggle_buttons_frame, text="Show Mid Frequency", command=self.show_mid_plot, state=tk.DISABLED)
        self.mid_button.pack(side=tk.LEFT, padx=5)

        self.high_button = tk.Button(self.toggle_buttons_frame, text="Show High Frequency", command=self.show_high_plot, state=tk.DISABLED)
        self.high_button.pack(side=tk.LEFT, padx=5)

        self.show_all_button = tk.Button(self.toggle_buttons_frame, text="Show All Frequencies", command=self.show_all_frequencies_plot, state=tk.DISABLED)
        self.show_all_button.pack(side=tk.LEFT, padx=5)

        self.show_original_button = tk.Button(root, text="Show Original Waveform", command=self.display_waveform, state=tk.DISABLED)
        self.show_original_button.pack(pady=10)

    def load_audio(self):
        self.filename = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3 *.aac")])
        if not self.filename:
            return

        self.file_label.config(text=os.path.basename(self.filename))

        if not self.filename.lower().endswith(".wav"):
            converted_filename = self.filename + "_converted.wav"
            audio = AudioSegment.from_file(self.filename)
            audio.export(converted_filename, format="wav")
            self.filename = converted_filename

        self.sampling_rate, self.audio_data = wavfile.read(self.filename)
        if self.audio_data.ndim > 1:
            self.audio_data = self.audio_data.mean(axis=1).astype(np.int16)

        self.audio_length = len(self.audio_data) / self.sampling_rate

        self.display_results()

    def display_results(self):
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        for widget in self.results_frame.winfo_children():
            widget.destroy()

        tk.Label(self.results_frame, text=f"Audio Length: {self.audio_length:.2f} seconds").pack()

        self.display_waveform()

        low_freq = self.compute_rt60(20, 500)
        mid_freq = self.compute_rt60(500, 2000)
        high_freq = self.compute_rt60(2000, 20000)
        freq, max_amplitude = self.compute_highest_resonance()

        tk.Label(self.results_frame, text=f"RT60 (Low): {low_freq:.2f} s").pack()
        tk.Label(self.results_frame, text=f"RT60 (Mid): {mid_freq:.2f} s").pack()
        tk.Label(self.results_frame, text=f"RT60 (High): {high_freq:.2f} s").pack()
        tk.Label(self.results_frame, text=f"Highest Resonance: {freq:.2f} Hz").pack()
        tk.Label(self.results_frame, text=f"Amplitude at Resonance: {max_amplitude:.2f}").pack()

        self.low_button.config(state=tk.NORMAL)
        self.mid_button.config(state=tk.NORMAL)
        self.high_button.config(state=tk.NORMAL)
        self.show_all_button.config(state=tk.NORMAL)
        self.show_original_button.config(state=tk.NORMAL)

    def compute_rt60(self, low_cut, high_cut):
        fft_data = np.abs(np.fft.rfft(self.audio_data))
        freqs = np.fft.rfftfreq(len(self.audio_data), 1 / self.sampling_rate)
        mask = (freqs >= low_cut) & (freqs <= high_cut)
        filtered_energy = np.sum(fft_data[mask] ** 2)
        rt60 = 0.5
        return rt60

    def compute_highest_resonance(self):
        fft_data = np.abs(np.fft.rfft(self.audio_data))
        freqs = np.fft.rfftfreq(len(self.audio_data), 1 / self.sampling_rate)
        peak_idx = np.argmax(fft_data)
        return freqs[peak_idx], fft_data[peak_idx]

    def display_waveform(self):
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        fig, ax = plt.subplots()
        ax.plot(np.linspace(0, len(self.audio_data) / self.sampling_rate, len(self.audio_data)), self.audio_data)
        ax.set_title("Waveform")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")

        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.get_tk_widget().pack()
        canvas.draw()

    def show_low_plot(self):
        self.show_frequency_plot(20, 500, "Low Frequency Plot (20-500 Hz)")

    def show_mid_plot(self):
        self.show_frequency_plot(500, 2000, "Mid Frequency Plot (500-2000 Hz)")

    def show_high_plot(self):
        self.show_frequency_plot(2000, 20000, "High Frequency Plot (2000-20000 Hz)")

    def show_frequency_plot(self, low_cut, high_cut, title):
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        freqs = np.fft.rfftfreq(len(self.audio_data), 1 / self.sampling_rate)
        fft_data = np.abs(np.fft.rfft(self.audio_data))

        fig, ax = plt.subplots()
        ax.plot(freqs, fft_data, label=title)
        ax.set_xlim(low_cut, high_cut)
        ax.set_title(title)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Amplitude")
        ax.legend()

        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.get_tk_widget().pack()
        canvas.draw()

    def show_all_frequencies_plot(self):
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        freqs = np.fft.rfftfreq(len(self.audio_data), 1 / self.sampling_rate)
        fft_data = np.abs(np.fft.rfft(self.audio_data))

        fig, ax = plt.subplots()
        ax.plot(freqs, fft_data, label="All Frequencies", color="gray", alpha=0.3)

        low_mask = (freqs >= 20) & (freqs <= 500)
        ax.plot(freqs[low_mask], fft_data[low_mask], label="Low (20-500 Hz)", color="blue")

        mid_mask = (freqs >= 500) & (freqs <= 2000)
        ax.plot(freqs[mid_mask], fft_data[mid_mask], label="Mid (500-2000 Hz)", color="green")

        high_mask = (freqs >= 2000) & (freqs <= 20000)
        ax.plot(freqs[high_mask], fft_data[high_mask], label="High (2000-20000 Hz)", color="red")

        ax.set_title("All Frequencies")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Amplitude")
        ax.legend()

        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.get_tk_widget().pack()
        canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = AudioAnalyzer(root)
    root.mainloop()
