import os
import sys
import threading
import queue
import time
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Use absolute package imports so the script runs from repo root
from stemalyze.separate import separate
from stemalyze.analysis import (
    analyze_vocals_melody,
    analyze_drums,
    bars_from_beats,
    analyze_harmony_other,
    quantize_melody_to_bars,
    analyze_drum_onsets,
)
from stemalyze.report import save_all

def _open_path(p: str):
    p = str(p)
    if sys.platform.startswith("win"):
        os.startfile(p)  # type: ignore
    elif sys.platform == "darwin":
        os.system(f'open "{p}"')
    else:
        os.system(f'xdg-open "{p}"')

class StemalyzeGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("STEMALYZE — Stems + Analysis GUI")
        self.geometry("820x620")
        self.minsize(760, 560)

        self.log_queue = queue.Queue()
        self.worker = None
        self.stop_flag = threading.Event()

        # Top frame: Inputs
        frm = ttk.Frame(self, padding=10)
        frm.pack(fill="x")

        # Audio file
        ttk.Label(frm, text="Audio file:").grid(row=0, column=0, sticky="w")
        self.var_audio = tk.StringVar()
        e_audio = ttk.Entry(frm, textvariable=self.var_audio, width=70)
        e_audio.grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(frm, text="Browse...", command=self.browse_audio).grid(row=0, column=2, padx=2)

        # Output dir
        ttk.Label(frm, text="Output dir:").grid(row=1, column=0, sticky="w")
        self.var_out = tk.StringVar(value=str(Path.cwd() / "stemalyze_out"))
        e_out = ttk.Entry(frm, textvariable=self.var_out, width=70)
        e_out.grid(row=1, column=1, sticky="ew", padx=6)
        ttk.Button(frm, text="Choose...", command=self.browse_out).grid(row=1, column=2, padx=2)

        # Options row
        opt = ttk.Frame(frm)
        opt.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(10,0))
        for i in range(8):
            opt.grid_columnconfigure(i, weight=1)

        ttk.Label(opt, text="Backend:").grid(row=0, column=0, sticky="w")
        self.var_backend = tk.StringVar(value="auto")
        ttk.Combobox(opt, textvariable=self.var_backend, values=["auto","demucs","spleeter","none"], width=10, state="readonly").grid(row=0, column=1, sticky="w")

        ttk.Label(opt, text="Sample rate:").grid(row=0, column=2, sticky="e")
        self.var_sr = tk.IntVar(value=22050)
        ttk.Entry(opt, textvariable=self.var_sr, width=8).grid(row=0, column=3, sticky="w")

        ttk.Label(opt, text="Time sig:").grid(row=0, column=4, sticky="e")
        self.var_barsig = tk.StringVar(value="auto")
        ttk.Combobox(opt, textvariable=self.var_barsig, values=["auto","3","4"], width=6, state="readonly").grid(row=0, column=5, sticky="w")

        self.var_overwrite = tk.BooleanVar(value=False)
        ttk.Checkbutton(opt, text="Overwrite", variable=self.var_overwrite).grid(row=0, column=6, sticky="w")

        # Action buttons
        btns = ttk.Frame(frm)
        btns.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(10,0))
        self.btn_run = ttk.Button(btns, text="Run Analysis", command=self.run_clicked)
        self.btn_run.pack(side="left")
        ttk.Button(btns, text="Open Output Folder", command=self.open_out).pack(side="left", padx=8)

        # Progress
        self.prog = ttk.Progressbar(frm, mode="indeterminate")
        self.prog.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(10,0))

        # Log and results
        lower = ttk.Panedwindow(self, orient="vertical")
        lower.pack(fill="both", expand=True, padx=10, pady=10)

        log_frame = ttk.LabelFrame(lower, text="Log")
        res_frame = ttk.LabelFrame(lower, text="Results")
        lower.add(log_frame, weight=3)
        lower.add(res_frame, weight=2)

        self.txt = tk.Text(log_frame, wrap="word", height=12)
        self.txt.pack(fill="both", expand=True)
        self.txt.configure(state="disabled")

        res_grid = ttk.Frame(res_frame)
        res_grid.pack(fill="both", expand=True, padx=8, pady=8)
        for i in range(4): res_grid.grid_columnconfigure(i, weight=1)

        ttk.Label(res_grid, text="Tempo (BPM):").grid(row=0, column=0, sticky="e")
        self.var_bpm = tk.StringVar(value="-")
        ttk.Label(res_grid, textvariable=self.var_bpm).grid(row=0, column=1, sticky="w")

        ttk.Label(res_grid, text="Time Sig:").grid(row=0, column=2, sticky="e")
        self.var_sig = tk.StringVar(value="-")
        ttk.Label(res_grid, textvariable=self.var_sig).grid(row=0, column=3, sticky="w")

        ttk.Label(res_grid, text="Bars:").grid(row=1, column=0, sticky="e")
        self.var_bars = tk.StringVar(value="-")
        ttk.Label(res_grid, textvariable=self.var_bars).grid(row=1, column=1, sticky="w")

        ttk.Label(res_grid, text="Report:").grid(row=2, column=0, sticky="e")
        self.var_report = tk.StringVar(value="-")
        ttk.Label(res_grid, textvariable=self.var_report).grid(row=2, column=1, columnspan=2, sticky="w")
        ttk.Button(res_grid, text="Open Report", command=self.open_report).grid(row=2, column=3, sticky="e")

        # Periodic UI update
        self.after(100, self._drain_log_queue)

    def log(self, msg: str):
        self.log_queue.put(msg.rstrip() + "\n")

    def _drain_log_queue(self):
        while True:
            try:
                line = self.log_queue.get_nowait()
            except queue.Empty:
                break
            self.txt.configure(state="normal")
            self.txt.insert("end", line)
            self.txt.see("end")
            self.txt.configure(state="disabled")
        self.after(120, self._drain_log_queue)

    def browse_audio(self):
        f = filedialog.askopenfilename(title="Choose audio file",
                                       filetypes=[("Audio","*.wav *.mp3 *.flac *.ogg *.m4a"), ("All","*.*")])
        if f:
            self.var_audio.set(f)

    def browse_out(self):
        d = filedialog.askdirectory(title="Choose output folder")
        if d:
            self.var_out.set(d)

    def open_out(self):
        p = self.var_out.get().strip()
        if p:
            _open_path(p)

    def open_report(self):
        rp = self.var_report.get().strip()
        if rp and rp not in ("-", "") and os.path.exists(rp):
            _open_path(rp)
        else:
            messagebox.showinfo("Report", "No report yet. Run analysis first.")

    def run_clicked(self):
        if self.worker and self.worker.is_alive():
            messagebox.showwarning("Busy", "Analysis is already running.")
            return
        audio = self.var_audio.get().strip()
        outdir = self.var_out.get().strip()
        if not audio or not os.path.exists(audio):
            messagebox.showerror("Missing file", "Please choose a valid audio file.")
            return
        if not outdir:
            messagebox.showerror("Missing folder", "Please choose an output folder.")
            return
        Path(outdir).mkdir(parents=True, exist_ok=True)
        backend = self.var_backend.get()
        sr = int(self.var_sr.get())
        barsig = self.var_barsig.get()
        barsig_val = None if barsig == "auto" else int(barsig)
        overwrite = bool(self.var_overwrite.get())

        self.btn_run.configure(state="disabled")
        self.prog.start(10)

        self.worker = threading.Thread(target=self._run_pipeline, args=(audio, outdir, backend, sr, barsig_val, overwrite), daemon=True)
        self.worker.start()

    def _run_pipeline(self, audio, outdir, backend, sr, barsig_val, overwrite):
        try:
            self.log(f"Audio: {audio}")
            self.log(f"Out: {outdir}  | Backend: {backend}  | SR: {sr}  | TimeSig: {barsig_val or 'auto'}  | Overwrite: {overwrite}")
            # 1) Separate
            self.log("Separating stems...")
            stems = separate(audio, outdir, backend=backend, overwrite=overwrite)
            for k,v in stems.items():
                self.log(f" - {k}: {v}")

            # 2) Drums → tempo/beatgrid/tsig
            self.log("Analyzing drums...")
            drums = analyze_drums(stems["drums"], sr=sr, force_time_sig_beats=barsig_val)
            self.log(f"Tempo: {drums.tempo_bpm:.2f}  | TimeSig: {drums.time_sig_beats}/4  | Beats: {len(drums.beat_times)}")

            # 3) Bars
            import soundfile as sf
            dur = sf.info(stems["drums"]).duration
            bars = bars_from_beats(drums, audio_duration=dur)
            self.log(f"Bars detected: {len(bars)}")

            # 4) Vocals melody
            self.log("Extracting vocal melody...")
            notes = analyze_vocals_melody(stems["vocals"], sr=sr)
            bar_melody = quantize_melody_to_bars(notes, bars)
            self.log(f"Vocal notes: {sum(len(v) for v in bar_melody.values())} segments")

            # 5) Harmony from 'other'
            self.log("Estimating chords per bar...")
            chords = analyze_harmony_other(stems["other"], bars, sr=sr, bass_path=stems.get("bass"))

            # 5b) Drums → per-bar kick/snare pattern (CPU-friendly)
            self.log("Extracting drum onset patterns...")
            drum_patterns = analyze_drum_onsets(stems["drums"], bars, sr=sr)

            # 6) Save
            meta = {"source": str(Path(audio).resolve()), "backend": backend, "sr": sr,
                    "tempo_bpm": drums.tempo_bpm, "time_sig_beats": drums.time_sig_beats}
            report_path = save_all(outdir, meta, bars, chords, bar_melody, drum_patterns)
            self.log(f"Report written to: {report_path}")

            # Update result labels
            self.var_bpm.set(f"{drums.tempo_bpm:.2f}")
            self.var_sig.set(f"{drums.time_sig_beats}/4")
            self.var_bars.set(str(len(bars)))
            self.var_report.set(report_path)

        except Exception as e:
            self.log(f"[ERROR] {e}")
            messagebox.showerror("Error", str(e))
        finally:
            self.prog.stop()
            self.btn_run.configure(state="normal")

def main():
    app = StemalyzeGUI()
    app.mainloop()

if __name__ == "__main__":
    main()
