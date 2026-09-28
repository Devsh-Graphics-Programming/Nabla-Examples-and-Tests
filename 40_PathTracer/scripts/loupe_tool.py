import os, json
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2, numpy as np
import tkinter as tk
from tkinter import filedialog, simpledialog, ttk, messagebox
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Circle, Rectangle, ConnectionPatch

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.join(HERE, "loupe_project.json")
PALETTE = ["#ffcc00", "#00e5ff", "#ff4d4d", "#7CFC00", "#ff66ff", "#ffffff", "#ff9900"]


def load_lin(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
    if img is None:
        raise RuntimeError(f"cannot read {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    if img.max() > 2.0:                     # 8-bit LDR (png/jpg) -> normalize
        img = img / 255.0
    return np.where(np.isfinite(img), img, 0.0)


def enc(lin, ev):
    return np.clip(lin * (2.0 ** ev), 0, 1) ** (1 / 2.2)


class LoupeTool:
    def __init__(self, root):
        self.root = root
        root.title("loupe figure tool")
        self.images = []        # {title, path, lin}
        self.loupes = []        # {cx,cy,r,color,anchor}
        self.base = 0
        self.pending = None
        self.drag = None        # ("anchor"|"roi", index)
        self.sel = None         # selected loupe index
        self.ev = tk.DoubleVar(value=0.0)
        self.shape = tk.StringVar(value="circle")
        self.stroke = tk.DoubleVar(value=0.7)
        self.inset_fr = tk.DoubleVar(value=0.26)
        self.connect = tk.BooleanVar(value=True)
        self._extra = []

        # ---- left panel ----
        left = tk.Frame(root); left.pack(side="left", fill="y", padx=6, pady=6)
        tk.Label(left, text="Images (compared, left->right)").pack(anchor="w")
        self.listbox = tk.Listbox(left, width=34, height=10, exportselection=False)
        self.listbox.pack(fill="y")
        self.listbox.bind("<<ListboxSelect>>", self.on_pick_base)
        b = tk.Frame(left); b.pack(fill="x", pady=4)
        tk.Button(b, text="Add", command=self.add_image).pack(side="left")
        tk.Button(b, text="Remove", command=self.remove_image).pack(side="left")
        tk.Button(b, text="Rename", command=self.rename_image).pack(side="left")
        tk.Button(b, text="Up", command=lambda: self.move(-1)).pack(side="left")
        tk.Button(b, text="Dn", command=lambda: self.move(1)).pack(side="left")

        opt = tk.LabelFrame(left, text="style"); opt.pack(fill="x", pady=6)
        row = tk.Frame(opt); row.pack(fill="x")
        tk.Label(row, text="shape").pack(side="left")
        ttk.Combobox(row, textvariable=self.shape, values=["circle", "square"],
                     width=8, state="readonly").pack(side="left")
        self.shape.trace_add("write", lambda *_: self.redraw())
        for lbl, var, frm, to, inc in [("EV", self.ev, -5, 5, 0.1),
                                       ("stroke", self.stroke, 0.2, 4, 0.1),
                                       ("inset", self.inset_fr, 0.1, 0.5, 0.02)]:
            r = tk.Frame(opt); r.pack(fill="x")
            tk.Label(r, text=lbl, width=6, anchor="w").pack(side="left")
            tk.Scale(r, from_=frm, to=to, resolution=inc, orient="horizontal",
                     variable=var, command=lambda *_: self.redraw(), length=150).pack(side="left")
        tk.Checkbutton(opt, text="connector line", variable=self.connect,
                       command=self.redraw).pack(anchor="w")

        lf = tk.LabelFrame(left, text="loupes (insets)"); lf.pack(fill="x", pady=6)
        self.loupebox = tk.Listbox(lf, width=34, height=6, exportselection=False)
        self.loupebox.pack(fill="x")
        self.loupebox.bind("<<ListboxSelect>>", self.on_pick_loupe)
        lb = tk.Frame(lf); lb.pack(fill="x")
        tk.Button(lb, text="Remove", command=self.remove_loupe).pack(side="left")
        tk.Button(lb, text="Recolor", command=self.recolor_loupe).pack(side="left")

        tk.Button(left, text="Export PNG + PDF", command=self.export,
                  height=2, bg="#3a7").pack(fill="x", pady=6)
        tk.Label(left, text="click region -> click to place zoom\n"
                           "drag ROI circle = move zoom target\n"
                           "drag inset = move zoom  |  scroll = resize\n"
                           "right-click = remove",
                 justify="left", fg="#666").pack(anchor="w")

        # ---- canvas ----
        self.fig = Figure(figsize=(9, 6))
        self.ax = self.fig.add_axes([0, 0, 1, 1]); self.ax.axis("off")
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(side="left", fill="both", expand=True)
        self.canvas.mpl_connect("button_press_event", self.on_press)
        self.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.canvas.mpl_connect("button_release_event", self.on_release)
        self.canvas.mpl_connect("scroll_event", self.on_scroll)

        self.load_project()
        self.refresh_list()
        self.refresh_loupes()
        self.redraw()

    # ---- persistence ----
    def save_project(self):
        json.dump({"images": [{"title": im["title"], "path": im["path"]} for im in self.images],
                   "loupes": self.loupes,
                   "style": {"shape": self.shape.get(), "ev": self.ev.get(),
                             "stroke": self.stroke.get(), "inset_fr": self.inset_fr.get(),
                             "connect": self.connect.get()}},
                  open(PROJECT, "w"), indent=2)

    def load_project(self):
        if not os.path.exists(PROJECT):
            return
        d = json.load(open(PROJECT))
        missing = []
        for im in d.get("images", []):
            try:
                self.images.append({"title": im["title"], "path": im["path"],
                                    "lin": load_lin(im["path"])})
            except Exception as e:
                missing.append(im.get("path")); print("skip", im.get("path"), e)
        self.loupes = d.get("loupes", [])
        s = d.get("style", {})
        self.shape.set(s.get("shape", "circle")); self.ev.set(s.get("ev", 0.0))
        self.stroke.set(s.get("stroke", 0.7)); self.inset_fr.set(s.get("inset_fr", 0.26))
        self.connect.set(s.get("connect", True))
        if missing:
            self.root.after(300, lambda: messagebox.showwarning(
                "missing images", "could not load:\n" + "\n".join(missing)))

    # ---- image list ----
    def refresh_list(self):
        self.listbox.delete(0, "end")
        for im in self.images:
            self.listbox.insert("end", im["title"])
        if self.images:
            self.base = min(self.base, len(self.images) - 1)
            self.listbox.selection_clear(0, "end"); self.listbox.selection_set(self.base)

    def add_image(self):
        paths = filedialog.askopenfilenames(
            title="add image(s)", filetypes=[("images", "*.exr *.png *.hdr *.jpg *.jpeg"), ("all", "*.*")])
        for p in paths:
            default = os.path.basename(os.path.dirname(os.path.dirname(p))) or os.path.basename(p)
            title = simpledialog.askstring("title", f"title for:\n{os.path.basename(p)}",
                                           initialvalue=default, parent=self.root)
            if title is None:
                continue
            try:
                self.images.append({"title": title, "path": p, "lin": load_lin(p)})
            except Exception as e:
                messagebox.showerror("load failed", str(e))
        self.refresh_list(); self.save_project(); self.redraw()

    def remove_image(self):
        sel = self.listbox.curselection()
        if not sel:
            return
        self.images.pop(sel[0]); self.base = 0
        self.refresh_list(); self.save_project(); self.redraw()

    def rename_image(self):
        sel = self.listbox.curselection()
        if not sel:
            return
        i = sel[0]
        t = simpledialog.askstring("rename", "title", initialvalue=self.images[i]["title"], parent=self.root)
        if t:
            self.images[i]["title"] = t
            self.refresh_list(); self.save_project()

    def move(self, d):
        sel = self.listbox.curselection()
        if not sel:
            return
        i = sel[0]; j = i + d
        if 0 <= j < len(self.images):
            self.images[i], self.images[j] = self.images[j], self.images[i]
            self.base = j; self.refresh_list(); self.save_project(); self.redraw()

    def on_pick_base(self, _):
        sel = self.listbox.curselection()
        if sel:
            self.base = sel[0]; self.redraw()

    # ---- loupe list ----
    def refresh_loupes(self):
        self.loupebox.delete(0, "end")
        for i, lp in enumerate(self.loupes):
            self.loupebox.insert("end", f"{i}:  {lp['color']}  ({lp['cx']},{lp['cy']}) r{lp['r']}")
        if self.sel is not None and self.sel < len(self.loupes):
            self.loupebox.selection_set(self.sel)

    def select(self, i):
        self.sel = i
        self.refresh_loupes()
        self.redraw()

    def on_pick_loupe(self, _):
        s = self.loupebox.curselection()
        if s:
            self.sel = s[0]; self.redraw()

    def remove_loupe(self):
        if self.sel is None:
            return
        self.loupes.pop(self.sel); self.sel = None
        self.refresh_loupes(); self.save_project(); self.redraw()

    def recolor_loupe(self):
        if self.sel is None:
            return
        c = simpledialog.askstring("recolor", "hex color (e.g. #ff0000)",
                                   initialvalue=self.loupes[self.sel]["color"], parent=self.root)
        if c:
            self.loupes[self.sel]["color"] = c
            self.refresh_loupes(); self.save_project(); self.redraw()

    # ---- geometry helpers ----
    def _im(self):
        return self.images[self.base]["lin"] if self.images else None

    def clamp(self, f):
        h = self.inset_fr.get() / 2
        return min(1 - h, max(h, f))

    def anchor_xy(self, lp, W, H):
        fx, fy = lp["anchor"]
        return fx * W, (1 - fy) * H

    def hit_anchor(self, x, y, W, H):
        h = self.inset_fr.get() / 2
        for i, lp in enumerate(self.loupes):
            ax_, ay_ = self.anchor_xy(lp, W, H)
            if (ax_ - x) ** 2 + (ay_ - y) ** 2 < (h * W) ** 2:
                return i
        return None

    def hit_marker(self, x, y):
        for i, lp in enumerate(self.loupes):
            if (lp["cx"] - x) ** 2 + (lp["cy"] - y) ** 2 <= lp["r"] ** 2:
                return i
        return None

    def nearest(self, x, y):
        if not self.loupes:
            return None
        return min(range(len(self.loupes)),
                   key=lambda i: (self.loupes[i]["cx"] - x) ** 2 + (self.loupes[i]["cy"] - y) ** 2)

    def next_color(self):
        used = {lp["color"] for lp in self.loupes}
        for c in PALETTE:
            if c not in used:
                return c
        return PALETTE[len(self.loupes) % len(PALETTE)]

    # ---- events ----
    def on_press(self, e):
        im = self._im()
        if im is None or e.inaxes != self.ax or e.xdata is None:
            return
        H, W = im.shape[:2]
        if e.button == 3:
            k = self.hit_anchor(e.xdata, e.ydata, W, H)
            if k is None:
                k = self.hit_marker(e.xdata, e.ydata)
            if k is None:
                k = self.nearest(e.xdata, e.ydata)
            if k is not None:
                self.loupes.pop(k); self.sel = None
                self.refresh_loupes(); self.save_project(); self.redraw()
            return
        if e.button == 1:
            if self.pending is None:
                ka = self.hit_anchor(e.xdata, e.ydata, W, H)
                if ka is not None:
                    self.drag = ("anchor", ka); self.select(ka); return
                km = self.hit_marker(e.xdata, e.ydata)
                if km is not None:
                    self.drag = ("roi", km); self.select(km); return
                self.pending = dict(cx=round(e.xdata), cy=round(e.ydata), r=55, color=self.next_color())
                self.redraw()
            else:
                self.pending["anchor"] = [self.clamp(e.xdata / W), self.clamp(1 - e.ydata / H)]
                self.loupes.append(self.pending); self.pending = None
                self.sel = len(self.loupes) - 1
                self.refresh_loupes(); self.save_project(); self.redraw()

    def on_motion(self, e):
        im = self._im()
        if self.drag is None or im is None or e.inaxes != self.ax or e.xdata is None:
            return
        H, W = im.shape[:2]
        kind, i = self.drag
        lp = self.loupes[i]
        if kind == "anchor":
            lp["anchor"] = [self.clamp(e.xdata / W), self.clamp(1 - e.ydata / H)]
        else:
            lp["cx"] = round(min(W, max(0, e.xdata)))
            lp["cy"] = round(min(H, max(0, e.ydata)))
        self.redraw()

    def on_release(self, e):
        if self.drag is not None:
            self.refresh_loupes(); self.save_project()
        self.drag = None

    def on_scroll(self, e):
        im = self._im()
        if im is None or e.inaxes != self.ax:
            return
        H, W = im.shape[:2]
        if self.pending is not None:
            self.pending["r"] = max(8, round(self.pending["r"] * (1.1 if e.step > 0 else 1 / 1.1)))
        else:
            k = self.hit_marker(e.xdata, e.ydata)
            if k is None:
                k = self.hit_anchor(e.xdata, e.ydata, W, H)
            if k is None:
                k = self.sel if self.sel is not None else self.nearest(e.xdata, e.ydata)
            if k is not None:
                self.loupes[k]["r"] = max(8, round(self.loupes[k]["r"] * (1.1 if e.step > 0 else 1 / 1.1)))
                self.sel = k; self.refresh_loupes(); self.save_project()
        self.redraw()

    # ---- drawing (shared by canvas preview and export) ----
    def draw_loupe(self, ax, im, lp, ev):
        cx, cy, r, col = lp["cx"], lp["cy"], lp["r"], lp["color"]
        fx, fy = lp["anchor"]
        fr = self.inset_fr.get(); half = fr / 2; sw = self.stroke.get()
        if self.shape.get() == "circle":
            ax.add_patch(Circle((cx, cy), r, fill=False, ec=col, lw=sw))
        else:
            ax.add_patch(Rectangle((cx - r, cy - r), 2 * r, 2 * r, fill=False, ec=col, lw=sw))
        ins = ax.inset_axes([fx - half, fy - half, fr, fr])
        ins.imshow(enc(im, ev)); ins.set_xlim(cx - r, cx + r); ins.set_ylim(cy + r, cy - r)
        ins.set_xticks([]); ins.set_yticks([])
        if self.shape.get() == "circle":
            clip = Circle((cx, cy), r, transform=ins.transData)
            for a in ins.get_images():
                a.set_clip_path(clip)
            ins.set_frame_on(False)
            ins.add_patch(Circle((0.5, 0.5), 0.5, transform=ins.transAxes,
                                 fill=False, ec=col, lw=sw, clip_on=False))
        else:
            for sp in ins.spines.values():
                sp.set_color(col); sp.set_linewidth(sw)
        if self.connect.get():
            ax.add_artist(ConnectionPatch((cx, cy), (0.5, 0.5), "data", "axes fraction",
                                          axesA=ax, axesB=ins, color=col, lw=sw * 0.7, alpha=0.8))
        return ins

    def redraw(self):
        for a in self._extra:
            try:
                a.remove()
            except Exception:
                pass
        self._extra.clear()
        self.ax.clear(); self.ax.axis("off")
        im = self._im()
        if im is None:
            self.ax.text(0.5, 0.5, "Add an image ->", ha="center", va="center", transform=self.ax.transAxes)
            self.canvas.draw_idle(); return
        ev = self.ev.get()
        self.ax.imshow(enc(im, ev))
        for lp in self.loupes:
            self._extra.append(self.draw_loupe(self.ax, im, lp, ev))
        if self.sel is not None and self.sel < len(self.loupes):
            lp = self.loupes[self.sel]
            hl = Circle((lp["cx"], lp["cy"]), lp["r"] + 6, fill=False, ec="white",
                        lw=1.4, ls=":"); self.ax.add_patch(hl); self._extra.append(hl)
        if self.pending:
            p = self.pending
            m = Circle((p["cx"], p["cy"]), p["r"], fill=False, ec=p["color"], lw=1.6, ls="--")
            self.ax.add_patch(m); self._extra.append(m)
        self.canvas.draw_idle()

    def export(self):
        if not self.images:
            messagebox.showinfo("nothing", "add images first"); return
        out = filedialog.asksaveasfilename(defaultextension=".png", initialfile="loupe.png",
                                           filetypes=[("PNG", "*.png")])
        if not out:
            return
        stem = os.path.splitext(out)[0]
        ev = self.ev.get()
        n = len(self.images)
        fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.5))
        if n == 1:
            axes = [axes]
        for ax, im in zip(axes, self.images):
            ax.imshow(enc(im["lin"], ev)); ax.set_title(im["title"])
            ax.set_xticks([]); ax.set_yticks([])
            for lp in self.loupes:
                self.draw_loupe(ax, im["lin"], lp, ev)
        fig.tight_layout()
        fig.savefig(stem + ".png", dpi=200, bbox_inches="tight")
        fig.savefig(stem + ".pdf", bbox_inches="tight")
        plt.close(fig)
        messagebox.showinfo("done", f"wrote\n{stem}.png\n{stem}.pdf")


if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1400x760+80+40")
    root.report_callback_exception = lambda *a: __import__("traceback").print_exception(*a)
    LoupeTool(root)
    root.lift(); root.attributes("-topmost", True)
    root.after(400, lambda: root.attributes("-topmost", False))
    root.mainloop()
