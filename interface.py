import math
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import os

# --- importă modulele cu scripturi
import lab1 # type: ignore
import lab2 # type: ignore
import lab3 # type: ignore
import lab4 # type: ignore
import lab5 # type: ignore
import lab6 # type: ignore
import lab7 # type: ignore
import tema1 # type: ignore
import tema2 # type: ignore
import tema3 # type: ignore
import tema4 # type: ignore
import tema5 # type: ignore

scripts_dict = {
    "lab1": lab1,
    "lab2": lab2,
    "lab3": lab3,
    "lab4": lab4,
    "lab5": lab5,
    "lab6": lab6,
    "lab7": lab7,
    "tema1": tema1,
    "tema2": tema2,
    "tema3": tema3,
    "tema4": tema4,
    "tema5": tema5
}

class ImageProcessorGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Interfață Procesare Imagini")
        self.master.geometry("1200x800")

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("App.TFrame", background="#DDE")
        style.configure("Top.TFrame", background="#CCD")
        style.configure("Result.TFrame", background="#EEE")
        style.configure("TButton",
                        padding=6,
                        relief="raised",
                        foreground="#333",
                        background="#ABC",
                        font=("Arial", 10, "bold"))
        style.map("TButton",
                  foreground=[("active", "#000")],
                  background=[("active", "#FFF")])
        style.configure("Title.TLabel",
                        background="#CCD",
                        foreground="#000",
                        font=("Helvetica", 14, "bold"))
        style.configure("TLabel",
                        background="#EEE",
                        foreground="#111",
                        font=("Arial", 10))
 
        # 1) Frame sus: titlu + intrare fișier
        frame_top = ttk.Frame(master, style="Top.TFrame", padding=10)
        frame_top.pack(side=tk.TOP, fill=tk.X)

        self.label_title = ttk.Label(frame_top, text="Procesare Imagini", style="Title.TLabel")
        self.label_title.pack(side=tk.LEFT, padx=10)

        btn_browse = ttk.Button(frame_top, text="Alege Imagine", command=self.load_image_dialog)
        btn_browse.pack(side=tk.RIGHT, padx=10)

        self.image_path_var = tk.StringVar()
        self.entry_path = ttk.Entry(frame_top, textvariable=self.image_path_var, width=70)
        self.entry_path.pack(side=tk.RIGHT, padx=5)
        # 2) Frame mijloc: butoane pe mai multe rânduri
        frame_mid = ttk.Frame(master, style="App.TFrame", padding=5)
        frame_mid.pack(side=tk.TOP, fill=tk.X)

        max_cols = 7
        row = 0
        col = 0
        for script_name in scripts_dict.keys():
            btn = ttk.Button(frame_mid, text=script_name, command=lambda sn=script_name: self.run_script(sn))
            btn.grid(row=row, column=col, padx=5, pady=5, sticky="w")
            col += 1
            if col >= max_cols:
                col = 0
                row += 1

        # 3) Frame mare pt rezultate
        self.frame_results = ttk.Frame(master, style="Result.TFrame", padding=10)
        self.frame_results.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def load_image_dialog(self):
        file_path = filedialog.askopenfilename(
            filetypes=[
                ("Imagini", "*.png *.jpg *.jpeg *.bmp *.tiff *.gif"), 
                ("Toate", "*.*")]
        )
        if file_path:
            self.image_path_var.set(file_path)

    def run_script(self, script_name):
        input_path = self.image_path_var.get().strip()
        if not os.path.isfile(input_path):
            messagebox.showerror("Eroare", "Selectează un fișier imagine valid.")
            return

        script_module = scripts_dict.get(script_name)
        if not script_module:
            messagebox.showerror("Eroare", f"Script inexistent: {script_name}")
            return

        # Curățăm frame-ul
        for widget in self.frame_results.winfo_children():
            widget.destroy()

        try:
            results = script_module.process_image(input_path)
            if not results or not isinstance(results, list):
                raise ValueError("Scriptul nu a returnat o listă validă de imagini.")

            # -- forțăm update la geometrie, ca să știm dimensiunea frame_results
            self.frame_results.update_idletasks()
            w = self.frame_results.winfo_width()
            h = self.frame_results.winfo_height()
            # fallback minim - dacă e 1x1, poate fereastra nu e încă desenată
            if w < 100: w = 1000
            if h < 100: h = 600

            max_col = 2
            n = len(results)
            rows = math.ceil(n / max_col)

            margin = 20  

            # dimensiunea maximă pentru fiecare "celulă"
            cell_w = (w // max_col) - margin
            cell_h = (h // rows) - margin

            for i, (title, pil_img) in enumerate(results):
                row = i // max_col
                col = i % max_col

                # Redimensionăm imaginea
                # Notă: "thumbnail" păstrează aspect ratio. 
                pil_img.thumbnail((cell_w, cell_h), Image.Resampling.LANCZOS)

                frame_img = ttk.LabelFrame(self.frame_results, text=title, padding=5)
                frame_img.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")

                tk_img = ImageTk.PhotoImage(pil_img)
                lbl_img = ttk.Label(frame_img, image=tk_img)
                lbl_img.image = tk_img
                lbl_img.pack()

            # Ca să se extindă frumos
            for r in range(rows):
                self.frame_results.rowconfigure(r, weight=1)
            for c in range(max_col):
                self.frame_results.columnconfigure(c, weight=1)

        except Exception as e:
            messagebox.showerror("Eroare la procesare", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorGUI(root)
    root.mainloop()
