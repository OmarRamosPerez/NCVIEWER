#!/usr/bin/env python3
"""
NCViewer — NetCDF file viewer inspired by ncview.
Main application window.
"""

import sys
import os
import numpy as np

try:
    import netCDF4 as nc
    NETCDF4_AVAILABLE = True
except ImportError:
    NETCDF4_AVAILABLE = False

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QTreeWidget, QTreeWidgetItem, QLabel, QComboBox,
    QPushButton, QStatusBar, QFileDialog, QGroupBox,
    QGridLayout, QSizePolicy, QToolBar, QAction,
    QCheckBox, QMessageBox, QTextEdit, QTabWidget
)
from PyQt5.QtCore import Qt, pyqtSignal, QSize
from PyQt5.QtGui import QColor

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

# ── Spatial dimension name heuristics ────────────────────────────────────────
SPATIAL_DIM_KEYWORDS = {
    'lat', 'latitude', 'y', 'rlat', 'south_north', 'nj',
    'lon', 'longitude', 'x', 'rlon', 'west_east', 'ni',
    'row', 'col', 'column', 'nx', 'ny',
}

COLORMAPS = [
    'viridis', 'plasma', 'inferno', 'magma', 'cividis',
    'RdBu_r', 'coolwarm', 'bwr', 'seismic',
    'rainbow', 'jet', 'turbo',
    'Blues', 'Greens', 'Reds', 'Oranges', 'Purples',
    'YlOrRd', 'YlGnBu', 'Spectral',
    'gray', 'bone', 'hot', 'cool',
]

DARK_STYLE = """
QMainWindow, QWidget {
    background-color: #1a1d23;
    color: #dce1e7;
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 13px;
}
QMenuBar {
    background-color: #12141a;
    color: #dce1e7;
    border-bottom: 1px solid #2e3340;
}
QMenuBar::item:selected { background-color: #2a6496; }
QMenu { background-color: #1f2330; border: 1px solid #2e3340; }
QMenu::item:selected { background-color: #2a6496; }
QToolBar {
    background-color: #12141a;
    border-bottom: 1px solid #2e3340;
    spacing: 4px;
    padding: 2px 6px;
}
QToolButton {
    background: transparent;
    border: 1px solid transparent;
    border-radius: 4px;
    padding: 4px 8px;
    color: #dce1e7;
}
QToolButton:hover  { background-color: #2a3044; border-color: #3a4460; }
QToolButton:pressed { background-color: #2a6496; }
QSplitter::handle { background-color: #2e3340; width: 2px; }
QTreeWidget {
    background-color: #1f2330;
    border: 1px solid #2e3340;
    border-radius: 4px;
    alternate-background-color: #1a1d23;
    color: #dce1e7;
}
QTreeWidget::item:selected { background-color: #2a5080; color: #ffffff; }
QTreeWidget::item:hover    { background-color: #2a3044; }
QTreeWidget QHeaderView::section {
    background-color: #12141a;
    color: #9ba8bb;
    border: none;
    border-bottom: 1px solid #2e3340;
    padding: 4px 8px;
    font-weight: bold;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
QGroupBox {
    border: 1px solid #2e3340;
    border-radius: 6px;
    margin-top: 12px;
    padding-top: 8px;
    color: #9ba8bb;
    font-size: 11px;
    font-weight: bold;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
QComboBox {
    background-color: #252936;
    border: 1px solid #3a4460;
    border-radius: 4px;
    padding: 4px 8px;
    color: #dce1e7;
    min-width: 100px;
}
QComboBox:hover { border-color: #4a88c7; }
QComboBox::drop-down { border: none; width: 20px; }
QComboBox QAbstractItemView {
    background-color: #252936;
    border: 1px solid #3a4460;
    selection-background-color: #2a5080;
}
QPushButton {
    background-color: #252936;
    border: 1px solid #3a4460;
    border-radius: 4px;
    padding: 5px 14px;
    color: #dce1e7;
}
QPushButton:hover    { background-color: #2a3044; border-color: #4a88c7; }
QPushButton:pressed  { background-color: #2a6496; }
QPushButton:disabled { color: #555d6e; border-color: #252936; background: #1a1d23; }
QPushButton#nav_btn {
    padding: 2px 8px;
    font-size: 14px;
    min-width: 26px;
    max-width: 26px;
    min-height: 24px;
    max-height: 24px;
}
QLabel { color: #dce1e7; }
QLabel#dim_label  { color: #9ba8bb; font-size: 11px; }
QLabel#idx_label  { color: #4a88c7; font-family: monospace; font-size: 12px; min-width: 80px; }
QLabel#val_label  { color: #4a88c7; font-family: monospace; font-size: 12px; }
QStatusBar {
    background-color: #12141a;
    color: #9ba8bb;
    border-top: 1px solid #2e3340;
    font-size: 12px;
}
QStatusBar::item { border: none; }
QTabWidget::pane { border: 1px solid #2e3340; }
QTabBar::tab {
    background: #1f2330;
    border: 1px solid #2e3340;
    padding: 6px 14px;
    color: #9ba8bb;
    margin-right: 2px;
}
QTabBar::tab:selected { background: #252936; color: #dce1e7; border-bottom: 2px solid #4a88c7; }
QScrollBar:vertical { background: #1a1d23; width: 8px; border-radius: 4px; }
QScrollBar::handle:vertical { background: #3a4460; border-radius: 4px; min-height: 20px; }
QScrollBar::handle:vertical:hover { background: #4a88c7; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QTextEdit {
    background-color: #1f2330;
    border: 1px solid #2e3340;
    border-radius: 4px;
    color: #dce1e7;
    font-family: 'Courier New', monospace;
    font-size: 12px;
}
QCheckBox { color: #dce1e7; spacing: 6px; }
QCheckBox::indicator {
    width: 14px; height: 14px;
    border: 1px solid #3a4460;
    border-radius: 3px;
    background: #252936;
}
QCheckBox::indicator:checked { background: #2a6496; border-color: #4a88c7; }
"""


def _is_spatial(dim_name: str) -> bool:
    """Return True if dim_name looks like a spatial (lat/lon/x/y) dimension."""
    return dim_name.lower() in SPATIAL_DIM_KEYWORDS


# ── Compact arrow navigator (only for non-spatial dims) ───────────────────────

class DimArrows(QWidget):
    """◀  idx / total  ▶  navigator for a single non-spatial dimension."""
    valueChanged = pyqtSignal(int)

    def __init__(self, dim_name: str, size: int, parent=None):
        super().__init__(parent)
        self.dim_name = dim_name
        self.size = size
        self._index = 0

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 1, 0, 1)
        row.setSpacing(4)

        lbl = QLabel(dim_name)
        lbl.setObjectName("dim_label")
        lbl.setFixedWidth(68)
        lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        row.addWidget(lbl)

        self.btn_prev = QPushButton("◀")
        self.btn_prev.setObjectName("nav_btn")
        self.btn_prev.setToolTip("Previous step  (←)")
        self.btn_prev.clicked.connect(self._prev)
        row.addWidget(self.btn_prev)

        self.idx_lbl = QLabel(self._fmt())
        self.idx_lbl.setObjectName("idx_label")
        self.idx_lbl.setAlignment(Qt.AlignCenter)
        row.addWidget(self.idx_lbl)

        self.btn_next = QPushButton("▶")
        self.btn_next.setObjectName("nav_btn")
        self.btn_next.setToolTip("Next step  (→)")
        self.btn_next.clicked.connect(self._next)
        row.addWidget(self.btn_next)

        row.addStretch()
        self._refresh()

    def _fmt(self):
        return f"{self._index} / {self.size - 1}"

    def _prev(self):
        if self._index > 0:
            self._index -= 1
            self._emit()

    def _next(self):
        if self._index < self.size - 1:
            self._index += 1
            self._emit()

    def _emit(self):
        self.idx_lbl.setText(self._fmt())
        self._refresh()
        self.valueChanged.emit(self._index)

    def _refresh(self):
        self.btn_prev.setEnabled(self._index > 0)
        self.btn_next.setEnabled(self._index < self.size - 1)

    def value(self):
        return self._index

    def setValue(self, v: int):
        self._index = max(0, min(v, self.size - 1))
        self.idx_lbl.setText(self._fmt())
        self._refresh()


# ── Matplotlib canvas (with constrained_layout to avoid shrinking) ────────────

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(constrained_layout=True)   # FIX: mantiene tamaño estable
        self.fig.patch.set_facecolor('#1a1d23')
        self.ax = self.fig.add_subplot(111)
        self._style_axes()
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def _style_axes(self):
        self.ax.set_facecolor('#1f2330')
        for spine in self.ax.spines.values():
            spine.set_edgecolor('#3a4460')
        self.ax.tick_params(colors='#9ba8bb', labelsize=9)
        self.ax.xaxis.label.set_color('#9ba8bb')
        self.ax.yaxis.label.set_color('#9ba8bb')
        self.ax.title.set_color('#dce1e7')


# ── Main window ───────────────────────────────────────────────────────────────

class NCViewer(QMainWindow):

    def __init__(self):
        super().__init__()
        self.nc_file = None
        self.current_var = None
        self.dim_controls = {}   # dim_name -> DimArrows  (only non-spatial)
        self.date_label = None   # for time dimension date display
        self._colorbar = None

        self.setWindowTitle("NCViewer – NetCDF Viewer")
        self.resize(1300, 820)
        self.setStyleSheet(DARK_STYLE)

        self._build_menubar()
        self._build_toolbar()
        self._build_ui()
        self._build_statusbar()
        self._show_welcome()

    # ── Menu ──────────────────────────────────────────────────────────────────

    def _build_menubar(self):
        mb = self.menuBar()

        fm = mb.addMenu("&File")
        a = QAction("&Open NetCDF…", self); a.setShortcut("Ctrl+O"); a.triggered.connect(self.open_file); fm.addAction(a)
        fm.addSeparator()
        a = QAction("&Close File", self); a.triggered.connect(self.close_file); fm.addAction(a)
        fm.addSeparator()
        a = QAction("&Quit", self); a.setShortcut("Ctrl+Q"); a.triggered.connect(self.close); fm.addAction(a)

        vm = mb.addMenu("&View")
        self.act_colorbar = QAction("Show Colorbar", self, checkable=True, checked=True)
        self.act_colorbar.triggered.connect(self._replot); vm.addAction(self.act_colorbar)
        self.act_grid = QAction("Show Grid", self, checkable=True, checked=False)
        self.act_grid.triggered.connect(self._replot); vm.addAction(self.act_grid)

        em = mb.addMenu("&Export")
        a = QAction("Save Figure…", self); a.setShortcut("Ctrl+S"); a.triggered.connect(self.save_figure); em.addAction(a)

        hm = mb.addMenu("&Help")
        a = QAction("About", self); a.triggered.connect(self._show_about); hm.addAction(a)

    # ── Toolbar ───────────────────────────────────────────────────────────────

    def _build_toolbar(self):
        tb = QToolBar(); tb.setMovable(False); self.addToolBar(tb)
        tb.setIconSize(QSize(16, 16))

        a = QAction("📂  Open File", self); a.triggered.connect(self.open_file); tb.addAction(a)
        tb.addSeparator()
        tb.addWidget(QLabel("  Colormap: "))
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(COLORMAPS)
        self.cmap_combo.setCurrentText("viridis")
        self.cmap_combo.currentTextChanged.connect(self._replot)
        tb.addWidget(self.cmap_combo)
        tb.addSeparator()
        self.log_check = QCheckBox("Log scale")
        self.log_check.stateChanged.connect(self._replot)
        tb.addWidget(self.log_check)
        tb.addSeparator()
        tb.addWidget(QLabel("  Interp: "))
        self.interp_combo = QComboBox()
        self.interp_combo.addItems(["nearest", "bilinear", "bicubic", "gaussian"])
        self.interp_combo.currentTextChanged.connect(self._replot)
        tb.addWidget(self.interp_combo)
        tb.addSeparator()
        a = QAction("💾  Save Figure", self); a.triggered.connect(self.save_figure); tb.addAction(a)

    # ── Central UI ────────────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        root = QHBoxLayout(central); root.setContentsMargins(0, 0, 0, 0); root.setSpacing(0)

        splitter = QSplitter(Qt.Horizontal); root.addWidget(splitter)

        # ── Left panel ───────────────────────────────────────────────────────
        left = QWidget(); left.setMinimumWidth(260); left.setMaximumWidth(340)
        ll = QVBoxLayout(left); ll.setContentsMargins(8, 8, 4, 8); ll.setSpacing(8)

        fg = QGroupBox("File"); fgl = QVBoxLayout(fg)
        self.file_label = QLabel("No file open")
        self.file_label.setWordWrap(True)
        self.file_label.setStyleSheet("color:#9ba8bb; font-size:11px;")
        fgl.addWidget(self.file_label); ll.addWidget(fg)

        vg = QGroupBox("Variables"); vgl = QVBoxLayout(vg)
        self.var_tree = QTreeWidget()
        self.var_tree.setHeaderLabels(["Name", "Shape", "Type"])
        self.var_tree.setColumnWidth(0, 110); self.var_tree.setColumnWidth(1, 85)
        self.var_tree.itemClicked.connect(self._on_var_selected)
        self.var_tree.setAlternatingRowColors(True)
        vgl.addWidget(self.var_tree); ll.addWidget(vg)

        # Time/level navigation (non-spatial dims only)
        self.nav_group = QGroupBox("Navigation")
        self.nav_layout = QVBoxLayout(self.nav_group)
        self.nav_layout.setSpacing(3)
        _ph = QLabel("Select a variable"); _ph.setStyleSheet("color:#555d6e; font-size:11px;"); _ph.setAlignment(Qt.AlignCenter)
        self.nav_layout.addWidget(_ph)
        ll.addWidget(self.nav_group)

        # Axis selection
        ag = QGroupBox("Plot Axes"); agl = QGridLayout(ag)
        agl.addWidget(QLabel("X axis:"), 0, 0)
        self.x_combo = QComboBox(); self.x_combo.currentTextChanged.connect(self._replot); agl.addWidget(self.x_combo, 0, 1)
        agl.addWidget(QLabel("Y axis:"), 1, 0)
        self.y_combo = QComboBox(); self.y_combo.currentTextChanged.connect(self._replot); agl.addWidget(self.y_combo, 1, 1)
        ll.addWidget(ag)

        ll.addStretch()

        # ── Right panel ──────────────────────────────────────────────────────
        right = QWidget(); rl = QVBoxLayout(right); rl.setContentsMargins(4, 8, 8, 4); rl.setSpacing(0)

        self.tabs = QTabWidget()

        # Plot tab
        pt = QWidget(); ptl = QVBoxLayout(pt); ptl.setContentsMargins(0, 4, 0, 0); ptl.setSpacing(4)
        self.canvas = MplCanvas(self)
        self.nav_toolbar = NavigationToolbar(self.canvas, self)
        self.nav_toolbar.setStyleSheet("background:#12141a; color:#9ba8bb; border:none;")
        ptl.addWidget(self.nav_toolbar)
        ptl.addWidget(self.canvas)

        vrow = QHBoxLayout()
        vrow.addWidget(QLabel("Cursor:"))
        self.cursor_label = QLabel("—"); self.cursor_label.setObjectName("val_label"); vrow.addWidget(self.cursor_label)
        vrow.addStretch()
        for name, attr in [("Min:", "min_label"), ("Max:", "max_label"), ("Mean:", "mean_label")]:
            vrow.addWidget(QLabel(name))
            lbl = QLabel("—"); lbl.setObjectName("val_label"); setattr(self, attr, lbl); vrow.addWidget(lbl)
        ptl.addLayout(vrow)
        self.tabs.addTab(pt, "  Plot  ")

        # Info tab
        it = QWidget(); itl = QVBoxLayout(it)
        self.info_text = QTextEdit(); self.info_text.setReadOnly(True); itl.addWidget(self.info_text)
        self.tabs.addTab(it, "  File Info  ")

        rl.addWidget(self.tabs)
        splitter.addWidget(left); splitter.addWidget(right)
        splitter.setStretchFactor(1, 3)

        self.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)

    def _build_statusbar(self):
        self.status = QStatusBar(); self.setStatusBar(self.status)
        self.status.showMessage("Ready  —  Open a NetCDF file to begin")

    # ── Welcome screen ────────────────────────────────────────────────────────

    def _show_welcome(self):
        ax = self.canvas.ax; ax.clear(); ax.set_facecolor('#1f2330')
        ax.text(0.5, 0.55, "NCViewer", transform=ax.transAxes,
                ha='center', va='center', fontsize=36, fontweight='bold',
                color='#4a88c7', fontfamily='monospace')
        ax.text(0.5, 0.42, "NetCDF File Viewer", transform=ax.transAxes,
                ha='center', va='center', fontsize=14, color='#9ba8bb')
        ax.text(0.5, 0.30, "File → Open  or  Ctrl+O", transform=ax.transAxes,
                ha='center', va='center', fontsize=11, color='#555d6e', style='italic')
        ax.set_axis_off(); self.canvas.draw()

    # ── File operations ───────────────────────────────────────────────────────

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open NetCDF File", "",
            "NetCDF Files (*.nc *.nc4 *.netcdf *.cdf);;All Files (*)")
        if path:
            self._load_file(path)

    def _load_file(self, path: str):
        if not NETCDF4_AVAILABLE:
            QMessageBox.critical(self, "Error", "netCDF4 not installed.\npip install netCDF4"); return
        try:
            if self.nc_file:
                self.nc_file.close()
            self.nc_file = nc.Dataset(path, 'r')
            self.current_var = None
            self.file_label.setText(os.path.basename(path))
            self._populate_tree()
            self._populate_info()
            self.status.showMessage(f"Opened: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Error opening file", str(e))

    def close_file(self):
        if self.nc_file:
            self.nc_file.close(); self.nc_file = None
        self.current_var = None
        self.var_tree.clear(); self.file_label.setText("No file open")
        self.info_text.clear(); self._clear_nav_controls()
        self._show_welcome(); self.status.showMessage("File closed")

    def _populate_tree(self):
        self.var_tree.clear()
        if not self.nc_file: return

        d_root = QTreeWidgetItem(["── DIMENSIONS ──", "", ""])
        d_root.setFlags(Qt.NoItemFlags); d_root.setForeground(0, QColor("#555d6e"))
        self.var_tree.addTopLevelItem(d_root)
        for name, dim in self.nc_file.dimensions.items():
            size = "unlimited" if dim.isunlimited() else str(len(dim))
            item = QTreeWidgetItem([name, f"({size},)", "dim"])
            item.setForeground(2, QColor("#9ba8bb")); d_root.addChild(item)
        d_root.setExpanded(True)

        v_root = QTreeWidgetItem(["── VARIABLES ──", "", ""])
        v_root.setFlags(Qt.NoItemFlags); v_root.setForeground(0, QColor("#555d6e"))
        self.var_tree.addTopLevelItem(v_root)
        ndim_color = {1: "#4a88c7", 2: "#4ac78a", 3: "#c7a04a", 4: "#c74a4a"}
        for name, var in self.nc_file.variables.items():
            item = QTreeWidgetItem([name, str(var.shape), str(var.dtype)])
            item.setData(0, Qt.UserRole, name)
            item.setForeground(0, QColor(ndim_color.get(var.ndim, "#dce1e7")))
            v_root.addChild(item)
        v_root.setExpanded(True)

    def _populate_info(self):
        if not self.nc_file: return
        lines = ["=" * 60, "  FILE INFORMATION", "=" * 60,
                 f"\nFormat:  {self.nc_file.file_format}",
                 f"Path:    {self.nc_file.filepath()}",
                 "\n── Global Attributes ──────────────────────────────"]
        for attr in self.nc_file.ncattrs():
            lines.append(f"  {attr}: {getattr(self.nc_file, attr)}")
        lines.append("\n── Dimensions ─────────────────────────────────────")
        for name, dim in self.nc_file.dimensions.items():
            lines.append(f"  {name}: {len(dim)}" + (" (unlimited)" if dim.isunlimited() else ""))
        lines.append("\n── Variables ──────────────────────────────────────")
        for name, var in self.nc_file.variables.items():
            lines += [f"\n  {name}", f"    dtype:  {var.dtype}",
                      f"    shape:  {var.shape}", f"    dims:   {var.dimensions}"]
            for attr in var.ncattrs():
                lines.append(f"    {attr}: {getattr(var, attr)}")
        self.info_text.setPlainText("\n".join(lines))

    # ── Variable selection ────────────────────────────────────────────────────

    def _on_var_selected(self, item, col):
        var_name = item.data(0, Qt.UserRole)
        if not var_name or not self.nc_file: return
        if var_name not in self.nc_file.variables: return
        self.current_var = var_name
        var = self.nc_file.variables[var_name]
        self.status.showMessage(
            f"Variable: {var_name}  |  Shape: {var.shape}  |  Dims: {var.dimensions}")
        self._build_nav_controls(var)
        self._update_axis_combos(var)
        self._replot()

    # ----- Time string extraction for date display -----
    def _get_time_string(self, dim_name, idx):
        """Return formatted date string for a time dimension, or None if not time-like."""
        if 'time' not in dim_name.lower():
            return None
        if dim_name not in self.nc_file.variables:
            return None
        time_var = self.nc_file.variables[dim_name]
        if not hasattr(time_var, 'units'):
            return None
        try:
            # Try using netCDF4.num2date (works with cftime if available)
            import cftime
            time_val = time_var[idx]
            # Use netCDF4's num2date (handles calendars)
            date = nc.num2date(time_val, units=time_var.units,
                               calendar=getattr(time_var, 'calendar', 'standard'))
            return date.strftime('%Y-%m-%d %H:%M:%S')
        except Exception:
            # Fallback: just show the numeric value
            return f"value: {time_var[idx]:.3f}"

    def _update_date_label(self, dim_name, idx):
        if self.date_label is not None:
            date_str = self._get_time_string(dim_name, idx)
            if date_str:
                self.date_label.setText(f"📅 {date_str}")
            else:
                self.date_label.setText("")

    def _build_nav_controls(self, var):
        """Add ◀▶ controls ONLY for non-spatial (e.g. time, level) dimensions,
        and a date label below if the dimension is time-like."""
        self._clear_nav_controls()

        non_spatial = [d for d in var.dimensions if not _is_spatial(d)]

        if not non_spatial:
            lbl = QLabel("No navigation dims (spatial only)")
            lbl.setStyleSheet("color:#555d6e; font-size:11px;")
            lbl.setAlignment(Qt.AlignCenter)
            self.nav_layout.addWidget(lbl)
            return

        self.date_label = None   # reset

        for dim_name in non_spatial:
            size = len(self.nc_file.dimensions[dim_name])
            ctrl = DimArrows(dim_name, size)
            ctrl.valueChanged.connect(self._replot)
            # Connect to update date label when navigation changes
            ctrl.valueChanged.connect(lambda idx, d=dim_name: self._update_date_label(d, idx))
            self.dim_controls[dim_name] = ctrl
            self.nav_layout.addWidget(ctrl)

            # If dimension looks like time, add a date label widget below it
            if 'time' in dim_name.lower():
                self.date_label = QLabel()
                self.date_label.setObjectName("val_label")
                self.date_label.setAlignment(Qt.AlignCenter)
                self.date_label.setStyleSheet("color:#4a88c7; font-size:11px; margin-top:2px;")
                self.nav_layout.addWidget(self.date_label)
                # Initial update with current index (0)
                self._update_date_label(dim_name, ctrl.value())

    def _clear_nav_controls(self):
        self.dim_controls = {}
        self.date_label = None
        while self.nav_layout.count():
            child = self.nav_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def _update_axis_combos(self, var):
        self.x_combo.blockSignals(True); self.y_combo.blockSignals(True)
        self.x_combo.clear(); self.y_combo.clear()
        dims = list(var.dimensions)
        self.x_combo.addItems(dims); self.y_combo.addItems(dims)
        # Default: last dim → X, second-to-last → Y
        if len(dims) >= 2:
            self.x_combo.setCurrentIndex(len(dims) - 1)
            self.y_combo.setCurrentIndex(len(dims) - 2)
        self.x_combo.blockSignals(False); self.y_combo.blockSignals(False)

    # ----- Helper to get coordinate arrays for geographic axes -----
    def _get_coord_array(self, dim_name):
        """Return 1D coordinate array for a dimension, if it exists as a variable."""
        if not self.nc_file or dim_name not in self.nc_file.variables:
            return None
        var = self.nc_file.variables[dim_name]
        if var.ndim == 1:
            return var[:]
        return None

    # ── Plotting ──────────────────────────────────────────────────────────────

    def _replot(self):
        if not self.nc_file or not self.current_var: return
        var = self.nc_file.variables[self.current_var]
        try:
            data = self._slice_data(var)
        except Exception as e:
            self.status.showMessage(f"Slice error: {e}"); return
        if data is None: return

        ax = self.canvas.ax; fig = self.canvas.fig
        ax.clear()
        if self._colorbar:
            try: self._colorbar.remove()
            except Exception: pass
            self._colorbar = None
        self.canvas._style_axes()

        cmap  = self.cmap_combo.currentText()
        interp = self.interp_combo.currentText()
        units = f" [{var.units}]" if hasattr(var, 'units') else ""
        title = getattr(var, 'long_name', self.current_var)

        if data.ndim == 1:
            ax.plot(data, color='#4a88c7', linewidth=1.5)
            ax.set_xlabel("Index"); ax.set_ylabel(f"{title}{units}")
            ax.set_title(f"{self.current_var}{units}", color='#dce1e7', pad=10)

        elif data.ndim == 2:
            arr = np.array(np.ma.filled(data, np.nan) if np.ma.is_masked(data) else data,
                           dtype=float)
            valid = arr[np.isfinite(arr)]
            if not len(valid):
                ax.text(0.5, 0.5, "No valid data", transform=ax.transAxes,
                        ha='center', va='center', color='#9ba8bb', fontsize=12)
                self.canvas.draw(); return

            # ---- Geographic coordinates handling ----
            x_dim = self.x_combo.currentText()
            y_dim = self.y_combo.currentText()
            x_coords = self._get_coord_array(x_dim)
            y_coords = self._get_coord_array(y_dim)
            extent = None
            if x_coords is not None and y_coords is not None and len(x_coords) == arr.shape[1] and len(y_coords) == arr.shape[0]:
                xmin, xmax = float(x_coords.min()), float(x_coords.max())
                ymin, ymax = float(y_coords.min()), float(y_coords.max())
                # If latitude decreases (N->S), flip array and swap ymin/ymax
                if y_coords[0] > y_coords[-1]:
                    arr = np.flipud(arr)
                    ymin, ymax = ymax, ymin
                # If longitude decreases (e.g. 360->0), flip horizontally
                if x_coords[0] > x_coords[-1]:
                    arr = np.fliplr(arr)
                    xmin, xmax = xmax, xmin
                extent = [xmin, xmax, ymin, ymax]
            # -------------------------------------------------

            if self.log_check.isChecked():
                from matplotlib.colors import LogNorm
                pos = valid[valid > 0]
                norm = LogNorm(vmin=float(pos.min()) if len(pos) else 1e-10,
                               vmax=float(pos.max()) if len(pos) else 1.0)
                im = ax.imshow(arr, origin='lower', cmap=cmap,
                               interpolation=interp, norm=norm, aspect='auto',
                               extent=extent)
            else:
                vmin, vmax = float(np.nanmin(arr)), float(np.nanmax(arr))
                im = ax.imshow(arr, origin='lower', cmap=cmap,
                               interpolation=interp, vmin=vmin, vmax=vmax, aspect='auto',
                               extent=extent)

            if self.act_colorbar.isChecked():
                self._colorbar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
                self._colorbar.ax.tick_params(colors='#9ba8bb', labelsize=8)
                self._colorbar.set_label(units.strip("[] "), color='#9ba8bb')

            ax.set_title(f"{self.current_var}{units}", color='#dce1e7', pad=10)
            ax.set_xlabel(x_dim, color='#9ba8bb')
            ax.set_ylabel(y_dim, color='#9ba8bb')
            self.min_label.setText(f"{np.nanmin(arr):.4g}")
            self.max_label.setText(f"{np.nanmax(arr):.4g}")
            self.mean_label.setText(f"{np.nanmean(arr):.4g}")
        else:
            ax.text(0.5, 0.5,
                    f"Cannot display {data.ndim}D data.\nAdjust navigation controls.",
                    transform=ax.transAxes, ha='center', va='center',
                    color='#9ba8bb', fontsize=12)

        if self.act_grid.isChecked():
            ax.grid(True, color='#2e3340', linewidth=0.5, linestyle='--', alpha=0.7)

        # No tight_layout call - constrained_layout keeps geometry stable
        self.canvas.draw()

    def _slice_data(self, var):
        if var.ndim == 0: return None
        if var.ndim == 1: return var[:]

        x_dim = self.x_combo.currentText()
        y_dim = self.y_combo.currentText()
        dims  = list(var.dimensions)

        idx = []
        for d in dims:
            if d in (x_dim, y_dim):
                idx.append(slice(None))
            elif d in self.dim_controls:
                idx.append(int(self.dim_controls[d].value()))
            else:
                idx.append(0)   # spatial dims not in plot axes → take first index

        data = var[tuple(idx)]

        # Guarantee 2-D with y as axis-0, x as axis-1
        if data.ndim == 2:
            active = [d for d, ix in zip(dims, idx) if isinstance(ix, slice)]
            if len(active) == 2 and active[0] == x_dim:
                data = data.T
        return data

    # ── Mouse ────────────────────────────────────────────────────────────────

    def _on_mouse_move(self, event):
        if event.inaxes and event.xdata is not None:
            self.cursor_label.setText(f"x={event.xdata:.4g}  y={event.ydata:.4g}")
        else:
            self.cursor_label.setText("—")

    # ── Export ────────────────────────────────────────────────────────────────

    def save_figure(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Figure", "",
            "PNG (*.png);;JPEG (*.jpg);;SVG (*.svg);;PDF (*.pdf)")
        if path:
            self.canvas.fig.savefig(path, dpi=150, bbox_inches='tight',
                                     facecolor=self.canvas.fig.get_facecolor())
            self.status.showMessage(f"Saved: {path}")

    def _show_about(self):
        QMessageBox.about(self, "About NCViewer",
            "<h2>NCViewer</h2>"
            "<p>Python NetCDF viewer inspired by ncview.</p>"
            "<p><b>Libraries:</b> PyQt5 · Matplotlib · netCDF4 · NumPy</p>"
            "<p><b>Usage:</b><br>"
            "• <tt>ncviewer</tt> — open viewer<br>"
            "• <tt>ncviewer file.nc</tt> — open with file<br>"
            "• Click a variable → navigate time/levels with ◀ ▶</p>")

    def closeEvent(self, event):
        if self.nc_file: self.nc_file.close()
        event.accept()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("NCViewer")
    app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    win = NCViewer()
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        win._load_file(sys.argv[1])
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()