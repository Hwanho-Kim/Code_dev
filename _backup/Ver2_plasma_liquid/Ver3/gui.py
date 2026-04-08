"""
GUI for NOx Analyzer with Reaction Contribution Analysis.

This module provides the Tkinter-based graphical user interface.
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from config import GUI_CONFIG, DEFAULTS, GAS_PHASE_SPECIES
from preprocessor import GasPhasePreprocessor, PreprocessParams
from chemistry import CompleteAqueousChemistry
from chemistry_utils import apply_henry_law, celsius_to_kelvin
from utils import get_logger, generate_timestamp


class NOxAnalyzerGUI:
    """
    Enhanced NOx Analyzer GUI with Reaction Contribution Analysis.

    Features:
    - CSV data loading
    - Gas phase preprocessing
    - Aqueous phase ODE solver
    - Reaction contribution tracking
    - Results visualization and export
    """

    def __init__(self, root: tk.Tk):
        """
        Initialize the GUI.

        Parameters
        ----------
        root : tk.Tk
            Tkinter root window
        """
        self.root = root
        self.root.title(GUI_CONFIG.title)
        self.root.geometry(f"{GUI_CONFIG.width}x{GUI_CONFIG.height}")

        self.logger = get_logger()

        # Initialize components
        self.preprocessor = GasPhasePreprocessor()
        self.saline_mode = False
        self.chemistry = CompleteAqueousChemistry(saline_mode=self.saline_mode)

        # Data storage
        self.file_path: Optional[str] = None
        self.raw_df: Optional[pd.DataFrame] = None
        self.preprocessed_df: Optional[pd.DataFrame] = None
        self.results_df: Optional[pd.DataFrame] = None
        self.detailed_results_df: Optional[pd.DataFrame] = None
        self.contribution_data: List[Dict] = []
        self.diagnostic_data: List[Dict] = []  # Full diagnostic contributions per timestep
        self.mass_transfer_data: List[Dict] = []  # Mass transfer (gas→liquid) per timestep

        # Setup GUI
        self._setup_gui()
        self._show_welcome_message()

    def _setup_gui(self):
        """Setup all GUI components."""
        # Configure root grid weights FIRST
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky='nsew')

        # Configure main_frame grid weights
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.columnconfigure(2, weight=1)
        main_frame.rowconfigure(3, weight=1)  # Results notebook row

        # Title
        title_label = ttk.Label(
            main_frame,
            text="NOx Analysis with Reaction Contribution Tracking",
            font=('Helvetica', 16, 'bold')
        )
        title_label.grid(row=0, column=0, columnspan=3, pady=10)

        # File selection frame
        self._setup_file_frame(main_frame)

        # Parameters frame
        self._setup_param_frame(main_frame)

        # Results notebook
        self._setup_results_notebook(main_frame)

        # Action buttons
        self._setup_button_frame(main_frame)

        # Force geometry update
        self.root.update_idletasks()

    def _setup_file_frame(self, parent):
        """Setup file selection frame."""
        file_frame = ttk.LabelFrame(parent, text="Data Input", padding="10")
        file_frame.grid(row=1, column=0, columnspan=3, sticky='ew', pady=10)

        self.file_label = ttk.Label(file_frame, text="No file selected")
        self.file_label.grid(row=0, column=0, padx=5)

        ttk.Button(file_frame, text="Select CSV",
                   command=self._select_file).grid(row=0, column=1, padx=5)

        ttk.Button(file_frame, text="Preprocess",
                   command=self._preprocess_data).grid(row=0, column=2, padx=5)

        ttk.Button(file_frame, text="Save Preprocessed",
                   command=self._save_preprocessed).grid(row=0, column=3, padx=5)

        ttk.Button(file_frame, text="Analyze (ODE)",
                   command=self._analyze_data).grid(row=0, column=4, padx=5)

        ttk.Button(file_frame, text="Save ODE Results",
                   command=self._save_ode_results).grid(row=0, column=5, padx=5)

        ttk.Button(file_frame, text="Contribution Analysis",
                   command=self._show_contribution_analysis).grid(row=0, column=6, padx=5)

        ttk.Button(file_frame, text="Export Diagnostics",
                   command=self._export_diagnostics).grid(row=0, column=7, padx=5)

    def _setup_param_frame(self, parent):
        """Setup parameters frame."""
        param_frame = ttk.LabelFrame(parent, text="Parameters", padding="10")
        param_frame.grid(row=2, column=0, columnspan=3, sticky='ew', pady=10)

        # Row 1 - Basic parameters
        ttk.Label(param_frame, text="Temperature (C):").grid(row=0, column=0, sticky='e')
        self.temp_var = tk.StringVar(value="25")
        ttk.Entry(param_frame, textvariable=self.temp_var, width=10).grid(row=0, column=1)

        ttk.Label(param_frame, text="Humidity (%):").grid(row=0, column=2, sticky='e')
        self.humidity_var = tk.StringVar(value="50")
        ttk.Entry(param_frame, textvariable=self.humidity_var, width=10).grid(row=0, column=3)

        ttk.Label(param_frame, text="Initial pH:").grid(row=0, column=4, sticky='e')
        self.pH_var = tk.StringVar(value="7.0")
        ttk.Entry(param_frame, textvariable=self.pH_var, width=10).grid(row=0, column=5)

        # Row 2 - Advanced parameters
        ttk.Label(param_frame, text="O3/H2O2 ratio:").grid(row=1, column=0, sticky='e')
        self.h2o2_ratio_var = tk.StringVar(value="5000")
        ttk.Entry(param_frame, textvariable=self.h2o2_ratio_var, width=10).grid(row=1, column=1)

        ttk.Label(param_frame, text="Smooth window:").grid(row=1, column=2, sticky='e')
        self.smooth_window_var = tk.StringVar(value="11")
        ttk.Entry(param_frame, textvariable=self.smooth_window_var, width=10).grid(row=1, column=3)

        # Row 3 - Solution Type Selection
        ttk.Label(param_frame, text="Solution Type:",
                  font=('Helvetica', 10, 'bold')).grid(row=2, column=0, sticky='e')
        self.solution_type_var = tk.StringVar(value="di_water")
        ttk.Radiobutton(param_frame, text="DI Water", variable=self.solution_type_var,
                        value="di_water", command=self._on_solution_type_change).grid(row=2, column=1)
        ttk.Radiobutton(param_frame, text="Saline Solution", variable=self.solution_type_var,
                        value="saline", command=self._on_solution_type_change).grid(row=2, column=2)

        # NaCl concentration input (for saline mode)
        self.nacl_label = ttk.Label(param_frame, text="NaCl (%):")
        self.nacl_label.grid(row=2, column=3, sticky='e', padx=(20, 0))
        self.nacl_var = tk.StringVar(value="0.9")
        self.nacl_entry = ttk.Entry(param_frame, textvariable=self.nacl_var, width=8)
        self.nacl_entry.grid(row=2, column=4, sticky='w')
        self.nacl_molarity_label = ttk.Label(param_frame, text="(0.154 M Cl⁻)")
        self.nacl_molarity_label.grid(row=2, column=5, sticky='w')

        # Initially disable NaCl input (DI water is default)
        self._set_nacl_input_state(False)

        # Row 4 - Preprocessing options header
        ttk.Label(param_frame, text="Preprocessing Options:",
                  font=('Helvetica', 10, 'bold')).grid(row=3, column=0, columnspan=6, pady=5)

        # Row 5 - Checkboxes
        self.remove_outliers_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(param_frame, text="Remove outliers",
                        variable=self.remove_outliers_var).grid(row=4, column=0, columnspan=2)

        self.smooth_data_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(param_frame, text="Smooth data",
                        variable=self.smooth_data_var).grid(row=4, column=2, columnspan=2)

        self.estimate_n2o4_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(param_frame, text="Estimate N2O4",
                        variable=self.estimate_n2o4_var).grid(row=4, column=4, columnspan=2)

        self.estimate_hono_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(param_frame, text="Estimate HONO/HONO2",
                        variable=self.estimate_hono_var).grid(row=5, column=0, columnspan=2)

        self.estimate_h2o2_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(param_frame, text="Estimate H2O2",
                        variable=self.estimate_h2o2_var).grid(row=5, column=2, columnspan=2)

        # Mass transfer method (fixed to two-film theory)
        self.mass_transfer_var = tk.StringVar(value="two_film")

        # Info label
        ttk.Label(param_frame,
                  text="Gas phase preprocessing -> Two-film mass transfer -> ODE solver",
                  font=('Helvetica', 9, 'italic')).grid(row=6, column=0, columnspan=6, pady=5)

    def _on_solution_type_change(self):
        """Handle solution type change (DI water vs Saline)."""
        solution_type = self.solution_type_var.get()
        self.saline_mode = (solution_type == "saline")

        # Enable/disable NaCl input based on solution type
        self._set_nacl_input_state(self.saline_mode)

        # Reinitialize chemistry with new mode
        self.chemistry = CompleteAqueousChemistry(saline_mode=self.saline_mode)

        mode_str = "Saline Solution" if self.saline_mode else "DI Water"
        self.logger.info(f"Solution type changed to: {mode_str}")
        print(f"[Solution Type] {mode_str} - {len(self.chemistry.reactions)} reactions loaded", flush=True)

    def _set_nacl_input_state(self, enabled: bool):
        """Enable or disable NaCl concentration input."""
        state = 'normal' if enabled else 'disabled'
        self.nacl_entry.configure(state=state)
        if enabled:
            self.nacl_label.configure(foreground='black')
            self.nacl_molarity_label.configure(foreground='black')
            # Bind to update molarity label when value changes
            self.nacl_var.trace_add('write', self._update_nacl_molarity)
            self._update_nacl_molarity()
        else:
            self.nacl_label.configure(foreground='gray')
            self.nacl_molarity_label.configure(foreground='gray')

    def _update_nacl_molarity(self, *args):
        """Update the molarity label based on NaCl percentage input."""
        try:
            nacl_percent = float(self.nacl_var.get())
            # 0.9% NaCl = 9 g/L, NaCl MW = 58.44 g/mol
            # Molarity = (nacl_percent / 100) * 1000 / 58.44 = nacl_percent * 10 / 58.44
            molarity = nacl_percent * 10 / 58.44
            self.nacl_molarity_label.configure(text=f"({molarity:.3f} M Cl⁻)")
        except ValueError:
            self.nacl_molarity_label.configure(text="(invalid)")

    def _get_nacl_molarity(self) -> float:
        """Get NaCl concentration in mol/L (M) from percentage input."""
        try:
            nacl_percent = float(self.nacl_var.get())
            # Convert % to M: (% / 100) * 1000 g/L / 58.44 g/mol
            return nacl_percent * 10 / 58.44
        except ValueError:
            return 0.154  # Default: 0.9% = 0.154 M

    def _setup_results_notebook(self, parent):
        """Setup results notebook with tabs."""
        notebook = ttk.Notebook(parent)
        notebook.grid(row=3, column=0, columnspan=3, sticky='nsew', pady=10)

        # Preprocessing tab
        preprocess_frame = ttk.Frame(notebook)
        notebook.add(preprocess_frame, text="Preprocessing")

        self.preprocess_text = tk.Text(preprocess_frame, height=20, width=100, wrap=tk.WORD)
        self.preprocess_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        preprocess_scroll = ttk.Scrollbar(preprocess_frame, orient="vertical",
                                          command=self.preprocess_text.yview)
        preprocess_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.preprocess_text.configure(yscrollcommand=preprocess_scroll.set)

        # Analysis results tab
        result_frame = ttk.Frame(notebook)
        notebook.add(result_frame, text="Analysis Results")

        self.result_text = tk.Text(result_frame, height=20, width=100, wrap=tk.WORD)
        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        result_scroll = ttk.Scrollbar(result_frame, orient="vertical",
                                      command=self.result_text.yview)
        result_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.result_text.configure(yscrollcommand=result_scroll.set)

        # Contributions tab
        contribution_frame = ttk.Frame(notebook)
        notebook.add(contribution_frame, text="Reaction Contributions")

        self.contribution_text = tk.Text(contribution_frame, height=20, width=100, wrap=tk.WORD)
        self.contribution_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        contrib_scroll = ttk.Scrollbar(contribution_frame, orient="vertical",
                                       command=self.contribution_text.yview)
        contrib_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.contribution_text.configure(yscrollcommand=contrib_scroll.set)

    def _setup_button_frame(self, parent):
        """Setup action buttons frame."""
        button_frame = ttk.Frame(parent)
        button_frame.grid(row=4, column=0, columnspan=3, pady=10)

        ttk.Button(button_frame, text="Plots",
                   command=self._show_plots).grid(row=0, column=0, padx=5)

        ttk.Button(button_frame, text="Save All",
                   command=self._save_results).grid(row=0, column=1, padx=5)

        ttk.Button(button_frame, text="Reset",
                   command=self._reset).grid(row=0, column=2, padx=5)

    def _show_welcome_message(self):
        """Display welcome messages in tabs."""
        self.preprocess_text.insert(tk.END, f"NOx Gas-Liquid Transfer Analyzer v{GUI_CONFIG.version}\n\n")
        self.preprocess_text.insert(tk.END, "Features:\n")
        self.preprocess_text.insert(tk.END, "- Gas phase preprocessing\n")
        self.preprocess_text.insert(tk.END, "- ODE solver for aqueous reactions\n")
        self.preprocess_text.insert(tk.END, "- NO and NO2 radical tracking\n")
        self.preprocess_text.insert(tk.END, "- Reaction contribution analysis\n\n")
        self.preprocess_text.insert(tk.END, "Select a CSV file to begin.\n")

        self.result_text.insert(tk.END, "Aqueous Phase Analysis with Radical Tracking\n\n")
        self.result_text.insert(tk.END, "This version tracks:\n")
        self.result_text.insert(tk.END, "- NO and NO2 radicals in aqueous phase\n")
        self.result_text.insert(tk.END, "- Reaction contributions to NO/NO2 production\n")
        self.result_text.insert(tk.END, "- Percentage breakdown of production pathways\n\n")

        self.contribution_text.insert(tk.END, "Reaction Contribution Analysis\n\n")
        self.contribution_text.insert(tk.END, "Based on J. Phys. D methodology:\n")
        self.contribution_text.insert(tk.END, "- Track individual reaction rates\n")
        self.contribution_text.insert(tk.END, "- Calculate percentage contributions\n")
        self.contribution_text.insert(tk.END, "- Identify dominant production pathways\n\n")
        self.contribution_text.insert(tk.END, "Run analysis to see contributions.\n")

    def _get_params(self) -> PreprocessParams:
        """Get current parameter values."""
        try:
            temp_C = float(self.temp_var.get())
            humidity = float(self.humidity_var.get()) / 100.0
            o3_h2o2_ratio = float(self.h2o2_ratio_var.get())
            smooth_window = int(self.smooth_window_var.get())
        except ValueError:
            temp_C = 25.0
            humidity = 0.5
            o3_h2o2_ratio = 5000.0
            smooth_window = 11

        return PreprocessParams(
            temperature_K=celsius_to_kelvin(temp_C),
            humidity=humidity,
            smooth=self.smooth_data_var.get(),
            smooth_window=smooth_window,
            estimate_missing=self.estimate_n2o4_var.get() or self.estimate_hono_var.get() or self.estimate_h2o2_var.get(),
            o3_h2o2_ratio=o3_h2o2_ratio
        )

    def _select_file(self):
        """Handle file selection."""
        try:
            self.file_path = filedialog.askopenfilename(
                title="Select CSV File",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
        except Exception as e:
            self.logger.error(f"File dialog error: {e}")
            messagebox.showerror("Error", f"Failed to open file dialog:\n{e}")
            return

        if self.file_path:
            self.file_label.config(text=os.path.basename(self.file_path))

            try:
                self.raw_df = pd.read_csv(self.file_path)
                self._display_data_info()
            except Exception as e:
                self.logger.error(f"File load error: {e}")
                messagebox.showerror("Error", f"Failed to load file:\n{e}")

    def _display_data_info(self):
        """Display loaded data information."""
        if self.raw_df is None:
            return

        self.preprocess_text.delete(1.0, tk.END)
        self.preprocess_text.insert(tk.END, f"File: {os.path.basename(self.file_path)}\n")
        self.preprocess_text.insert(tk.END, f"Shape: {self.raw_df.shape[0]} rows x {self.raw_df.shape[1]} columns\n\n")

        self.preprocess_text.insert(tk.END, "Columns:\n")
        for col in self.raw_df.columns:
            self.preprocess_text.insert(tk.END, f"  - {col}\n")

        self.preprocess_text.insert(tk.END, "\nDetected gas species:\n")
        for species in GAS_PHASE_SPECIES:
            if species in self.raw_df.columns:
                self.preprocess_text.insert(tk.END, f"  - {species}\n")

    def _preprocess_data(self):
        """Run preprocessing pipeline."""
        if self.raw_df is None:
            messagebox.showwarning("Warning", "Please select a file first")
            return

        params = self._get_params()

        self.preprocess_text.insert(tk.END, "\n" + "=" * 50 + "\n")
        self.preprocess_text.insert(tk.END, "Starting preprocessing...\n")
        self.root.update_idletasks()

        try:
            self.preprocessed_df = self.preprocessor.preprocess(self.raw_df, params)

            self.preprocess_text.insert(tk.END, f"\nPreprocessing complete!\n")
            self.preprocess_text.insert(tk.END, f"Output shape: {self.preprocessed_df.shape}\n")

            # Show summary statistics
            self.preprocess_text.insert(tk.END, "\nSummary statistics:\n")
            for species in GAS_PHASE_SPECIES:
                if species in self.preprocessed_df.columns:
                    data = self.preprocessed_df[species]
                    self.preprocess_text.insert(
                        tk.END,
                        f"  {species}: min={data.min():.2e}, max={data.max():.2e}, mean={data.mean():.2e}\n"
                    )

        except Exception as e:
            self.logger.error(f"Preprocessing error: {e}")
            messagebox.showerror("Error", f"Preprocessing failed:\n{e}")

    def _analyze_data(self):
        """Run ODE analysis."""
        if self.preprocessed_df is None:
            messagebox.showwarning("Warning", "Please preprocess data first")
            return

        try:
            initial_pH = float(self.pH_var.get())
        except ValueError:
            initial_pH = 7.0

        # Ensure chemistry is initialized with correct solution type
        solution_type = self.solution_type_var.get()
        saline_mode = (solution_type == "saline")
        if saline_mode != self.saline_mode:
            self._on_solution_type_change()

        self.result_text.delete(1.0, tk.END)
        mode_str = "Saline Solution" if self.saline_mode else "DI Water"
        self.result_text.insert(tk.END, f"Starting ODE analysis ({mode_str})...\n")
        self.result_text.insert(tk.END, f"Using {len(self.chemistry.reactions)} reactions\n")

        # Mass transfer method (fixed to two-film theory)
        mass_transfer_method = "two_film"
        self.result_text.insert(tk.END, f"Mass transfer: Two-Film Theory\n")

        # Get Cl- concentration for saline mode
        cl_concentration = None
        if self.saline_mode:
            cl_concentration = self._get_nacl_molarity()
            self.result_text.insert(tk.END, f"Initial Cl- concentration: {cl_concentration:.3f} M ({self.nacl_var.get()}% NaCl)\n")

        self.result_text.insert(tk.END, "\n")
        self.root.update_idletasks()

        results = []
        self.contribution_data = []
        self.diagnostic_data = []  # Reset diagnostic data
        self.mass_transfer_data = []  # Track mass transfer contributions

        total_rows = len(self.preprocessed_df)

        # Accumulated aqueous concentrations (persists across timesteps)
        C_aq_accumulated = {}

        for idx, row in self.preprocessed_df.iterrows():
            if idx % 100 == 0:
                self.result_text.insert(tk.END, f"Processing row {idx}/{total_rows}...\n")
                self.root.update_idletasks()

            try:
                # Convert gas phase to aqueous phase using Henry's law
                # and ADD to accumulated concentrations
                C_aq_initial = C_aq_accumulated.copy()

                # Track mass transfer for this timestep
                mass_transfer_this_step = {}
                time_step = 1  # timestep duration [s] — shared by mass transfer and ODE

                for species in GAS_PHASE_SPECIES:
                    if species in row:
                        gas_conc = row[species]
                        if pd.notna(gas_conc) and gas_conc > 0:
                            current_conc = C_aq_initial.get(species, 0.0)
                            new_conc = apply_henry_law(
                                species, gas_conc,
                                method=mass_transfer_method,
                                delta_t=time_step,
                                current_aq_conc=current_conc
                            )
                            mass_transfer_this_step[species] = new_conc - current_conc
                            C_aq_initial[species] = new_conc

                C_final, contributions = self.chemistry.solve(
                    C_aq_initial, initial_pH, time_step=time_step,
                    cl_concentration=cl_concentration
                )

                # Update accumulated concentrations with results
                # (excluding pH and other non-concentration values)
                for species, conc in C_final.items():
                    if species != 'pH' and isinstance(conc, (int, float)) and conc > 0:
                        C_aq_accumulated[species] = conc

                # [DEV] 20개마다 핵심 파라미터 터미널 출력
                if idx % 20 == 0:
                    pH_val = C_final.get('pH', 0)
                    h2o2_val = C_final.get('H2O2', 0)
                    no2_val = C_final.get('NO2-', 0)
                    no3_val = C_final.get('NO3-', 0)
                    print(f"[{idx:4d}] pH={pH_val:5.2f}  H2O2={h2o2_val:.2e}  NO2-={no2_val:.2e}  NO3-={no3_val:.2e}", flush=True)

                # Store results
                result_row = {'index': idx}
                result_row.update(C_final)
                results.append(result_row)

                self.contribution_data.append({
                    'index': idx,
                    'contributions': contributions
                })

                # Store mass transfer data
                self.mass_transfer_data.append({
                    'index': idx,
                    'mass_transfer': mass_transfer_this_step
                })

                # Store diagnostic contributions (all species, production + consumption)
                # Merge mass transfer into diagnostic contributions
                diag = self.chemistry.get_diagnostic_contributions()

                # Add mass transfer as production source for dissolved species
                # Convert amount (mol/L) to rate (mol/L/s) for consistency with reaction rates
                for species, amount in mass_transfer_this_step.items():
                    if amount > 0:
                        rate = amount / time_step  # Convert to rate [mol/L/s]

                        # Map gas species to aqueous products for contribution tracking
                        # Direct dissolution
                        if species in diag:
                            if 'production' not in diag[species]:
                                diag[species]['production'] = {}
                            diag[species]['production']['Gas dissolution (Henry)'] = rate

                        # N2O5 → 2 NO3- (instant hydrolysis)
                        if species == 'N2O5' and 'NO3-' in diag:
                            if 'production' not in diag['NO3-']:
                                diag['NO3-']['production'] = {}
                            diag['NO3-']['production']['N2O5 dissolution → 2NO3-'] = 2 * rate

                        # N2O4 → NO2- + NO3- (instant hydrolysis)
                        if species == 'N2O4':
                            if 'NO2-' in diag:
                                if 'production' not in diag['NO2-']:
                                    diag['NO2-']['production'] = {}
                                diag['NO2-']['production']['N2O4 dissolution → NO2-'] = rate
                            if 'NO3-' in diag:
                                if 'production' not in diag['NO3-']:
                                    diag['NO3-']['production'] = {}
                                diag['NO3-']['production']['N2O4 dissolution → NO3-'] = rate

                        # HONO → HONO_total (acid-base equilibrium)
                        if species == 'HONO' and 'HONO' in diag:
                            if 'production' not in diag['HONO']:
                                diag['HONO']['production'] = {}
                            diag['HONO']['production']['HONO dissolution (Henry)'] = rate

                        # HONO2 → HONO2_total → NO3- (strong acid, fully dissociated)
                        if species == 'HONO2':
                            if 'HONO2' in diag:
                                if 'production' not in diag['HONO2']:
                                    diag['HONO2']['production'] = {}
                                diag['HONO2']['production']['HONO2 dissolution (Henry)'] = rate
                            if 'NO3-' in diag:
                                if 'production' not in diag['NO3-']:
                                    diag['NO3-']['production'] = {}
                                diag['NO3-']['production']['HONO2 dissolution → NO3-'] = rate

                        # H2O2 dissolution
                        if species == 'H2O2' and 'H2O2' in diag:
                            if 'production' not in diag['H2O2']:
                                diag['H2O2']['production'] = {}
                            diag['H2O2']['production']['H2O2 dissolution (Henry)'] = rate

                self.diagnostic_data.append({
                    'index': idx,
                    'time': row.get('time', idx * 0.1) if 'time' in row else idx * 0.1,
                    'diagnostics': diag
                })

            except Exception as e:
                self.logger.warning(f"Row {idx} analysis error: {e}")
                continue

        self.results_df = pd.DataFrame(results)

        self.result_text.insert(tk.END, f"\nAnalysis complete!\n")
        self.result_text.insert(tk.END, f"Processed {len(results)} rows\n")

        # Show summary
        if 'pH' in self.results_df.columns:
            pH_data = self.results_df['pH']
            self.result_text.insert(
                tk.END,
                f"\npH: min={pH_data.min():.2f}, max={pH_data.max():.2f}, mean={pH_data.mean():.2f}\n"
            )

    def _show_contribution_analysis(self):
        """Display contribution analysis results."""
        if not self.contribution_data:
            messagebox.showwarning("Warning", "Please run analysis first")
            return

        self.contribution_text.delete(1.0, tk.END)
        self.contribution_text.insert(tk.END, "Reaction Contribution Analysis\n")
        self.contribution_text.insert(tk.END, "=" * 60 + "\n\n")

        # Aggregate contributions
        avg_contributions = {'NO': defaultdict(list), 'NO2': defaultdict(list)}

        for data_point in self.contribution_data:
            contributions = data_point['contributions']
            for species in ['NO', 'NO2']:
                if species in contributions:
                    for rxn_label, percentage in contributions[species].items():
                        avg_contributions[species][rxn_label].append(percentage)

        # Display
        for species in ['NO', 'NO2']:
            self.contribution_text.insert(tk.END, f"\n{species} Production Contributors (Average %):\n")
            self.contribution_text.insert(tk.END, "-" * 50 + "\n")

            avg_dict = {}
            for rxn_label, percentages in avg_contributions[species].items():
                if percentages:
                    avg_dict[rxn_label] = np.mean(percentages)

            sorted_reactions = sorted(avg_dict.items(), key=lambda x: x[1], reverse=True)

            for i, (rxn_label, avg_pct) in enumerate(sorted_reactions[:15], 1):
                self.contribution_text.insert(tk.END, f"{i:2d}. {rxn_label:<50s} {avg_pct:6.2f}%\n")

    def _show_plots(self):
        """Show results plots."""
        if self.results_df is None:
            messagebox.showwarning("Warning", "No results to plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # pH plot
        if 'pH' in self.results_df.columns:
            axes[0, 0].plot(self.results_df['pH'])
            axes[0, 0].set_xlabel('Data Point')
            axes[0, 0].set_ylabel('pH')
            axes[0, 0].set_title('pH Evolution')

        # NO/NO2 plot
        if 'NO' in self.results_df.columns:
            axes[0, 1].plot(self.results_df['NO'], label='NO')
        if 'NO2' in self.results_df.columns:
            axes[0, 1].plot(self.results_df['NO2'], label='NO2')
        axes[0, 1].set_xlabel('Data Point')
        axes[0, 1].set_ylabel('Concentration (mol/L)')
        axes[0, 1].set_title('NO and NO2 in Aqueous Phase')
        axes[0, 1].legend()
        axes[0, 1].set_yscale('log')

        # Nitrite/Nitrate plot
        if 'NO2-' in self.results_df.columns:
            axes[1, 0].plot(self.results_df['NO2-'], label='NO2-')
        if 'NO3-' in self.results_df.columns:
            axes[1, 0].plot(self.results_df['NO3-'], label='NO3-')
        axes[1, 0].set_xlabel('Data Point')
        axes[1, 0].set_ylabel('Concentration (mol/L)')
        axes[1, 0].set_title('Nitrite and Nitrate')
        axes[1, 0].legend()
        axes[1, 0].set_yscale('log')

        # H2O2 plot
        if 'H2O2' in self.results_df.columns:
            axes[1, 1].plot(self.results_df['H2O2'])
            axes[1, 1].set_xlabel('Data Point')
            axes[1, 1].set_ylabel('Concentration (mol/L)')
            axes[1, 1].set_title('H2O2 Concentration')
            axes[1, 1].set_yscale('log')

        plt.tight_layout()
        plt.show()

    def _save_preprocessed(self):
        """Save preprocessed data."""
        if self.preprocessed_df is None:
            messagebox.showwarning("Warning", "No preprocessed data to save")
            return

        save_path = filedialog.asksaveasfilename(
            title="Save Preprocessed Data",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )

        if save_path:
            try:
                self.preprocessed_df.to_csv(save_path, index=False)
                messagebox.showinfo("Success", f"Saved to:\n{save_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Save failed:\n{e}")

    def _save_ode_results(self):
        """Save ODE analysis results."""
        if self.results_df is None:
            messagebox.showwarning("Warning", "No results to save")
            return

        save_path = filedialog.asksaveasfilename(
            title="Save ODE Results",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )

        if save_path:
            try:
                self.results_df.to_csv(save_path, index=False)
                messagebox.showinfo("Success", f"Saved to:\n{save_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Save failed:\n{e}")

    def _save_results(self):
        """Save all results to a directory."""
        if self.results_df is None:
            messagebox.showwarning("Warning", "No results to save")
            return

        save_dir = filedialog.askdirectory(title="Select directory to save results")

        if save_dir:
            timestamp = generate_timestamp()
            base_name = f"nox_analysis_{timestamp}"

            try:
                # Save preprocessed data
                if self.preprocessed_df is not None:
                    preprocess_path = os.path.join(save_dir, f"{base_name}_preprocessed.csv")
                    self.preprocessed_df.to_csv(preprocess_path, index=False)

                # Save ODE results
                summary_path = os.path.join(save_dir, f"{base_name}_summary.csv")
                self.results_df.to_csv(summary_path, index=False)

                # Save contribution analysis
                if self.contribution_data:
                    contrib_path = os.path.join(save_dir, f"{base_name}_contributions.txt")
                    self._save_contribution_file(contrib_path)

                messagebox.showinfo("Success", f"Results saved to:\n{save_dir}")

            except Exception as e:
                messagebox.showerror("Error", f"Save failed:\n{e}")

    def _save_contribution_file(self, path: str):
        """Save contribution analysis to text file."""
        with open(path, 'w', encoding='utf-8') as f:
            f.write("NO and NO2 Production Contribution Analysis\n")
            f.write("=" * 60 + "\n\n")

            # Aggregate
            avg_contributions = {'NO': defaultdict(list), 'NO2': defaultdict(list)}
            for data_point in self.contribution_data:
                contributions = data_point['contributions']
                for species in ['NO', 'NO2']:
                    if species in contributions:
                        for rxn_label, percentage in contributions[species].items():
                            avg_contributions[species][rxn_label].append(percentage)

            # Write
            for species in ['NO', 'NO2']:
                f.write(f"\n{species} Production Contributors (Average %):\n")
                f.write("-" * 50 + "\n")

                avg_dict = {}
                for rxn_label, percentages in avg_contributions[species].items():
                    if percentages:
                        avg_dict[rxn_label] = np.mean(percentages)

                sorted_reactions = sorted(avg_dict.items(), key=lambda x: x[1], reverse=True)

                for i, (rxn_label, avg_pct) in enumerate(sorted_reactions[:20], 1):
                    f.write(f"{i:2d}. {rxn_label:<60s} {avg_pct:6.2f}%\n")

    def _export_diagnostics(self):
        """Export diagnostic contribution analysis to folder with per-species CSV files."""
        if not self.diagnostic_data:
            messagebox.showwarning("Warning", "Please run analysis first")
            return

        # Ask user for folder name
        from tkinter import simpledialog
        folder_name = simpledialog.askstring(
            "Folder Name",
            "Enter folder name for diagnostic results:",
            initialvalue=f"diagnostics_{generate_timestamp()}"
        )
        if not folder_name:
            return

        # Select parent directory
        parent_dir = filedialog.askdirectory(title="Select parent directory")
        if not parent_dir:
            return

        # Create the folder
        save_dir = os.path.join(parent_dir, folder_name)
        os.makedirs(save_dir, exist_ok=True)

        try:
            # Export per-species CSV files (Origin format)
            self._export_diagnostics_per_species(save_dir)

            # Export summary TXT (with time info)
            txt_path = os.path.join(save_dir, "summary.txt")
            self._export_diagnostics_txt(txt_path)

            # Count files
            csv_count = len([f for f in os.listdir(save_dir) if f.endswith('.csv')])

            messagebox.showinfo("Success",
                f"Diagnostic files saved to:\n{save_dir}\n\n"
                f"- {csv_count} CSV files (per species, Origin format)\n"
                f"- summary.txt (time-averaged + final timestep)")

        except Exception as e:
            self.logger.error(f"Diagnostic export error: {e}")
            messagebox.showerror("Error", f"Export failed:\n{e}")

    def _export_diagnostics_per_species(self, save_dir: str):
        """
        Export diagnostic contributions as per-species CSV files (Origin format).

        Each species gets two files:
        - {species}_production.csv
        - {species}_consumption.csv

        Format (Origin-friendly, wide format):
        time, R1: reaction_label, R2: reaction_label, R3: ...
        0.0, 1.23e-5, 4.56e-6, 7.89e-7, ...
        0.1, 1.34e-5, 4.67e-6, 8.90e-7, ...
        """
        # Collect all unique reactions per species per type
        species_reactions = {}
        for species in self.chemistry.diagnostic_species:
            species_reactions[species] = {
                'production': set(),
                'consumption': set()
            }

        # First pass: collect all reaction labels
        for data_point in self.diagnostic_data:
            diagnostics = data_point['diagnostics']
            for species, type_data in diagnostics.items():
                for rxn_label in type_data['production'].keys():
                    species_reactions[species]['production'].add(rxn_label)
                for rxn_label in type_data['consumption'].keys():
                    species_reactions[species]['consumption'].add(rxn_label)

        # Export each species
        for species in self.chemistry.diagnostic_species:
            # Sanitize species name for filename (replace special chars)
            safe_name = species.replace('+', 'plus').replace('-', 'minus').replace('/', '_')

            for contrib_type in ['production', 'consumption']:
                reactions = sorted(species_reactions[species][contrib_type])
                if not reactions:
                    continue  # Skip if no reactions

                # Build data rows
                rows = []
                for data_point in self.diagnostic_data:
                    row = {'time': data_point['time']}
                    type_data = data_point['diagnostics'].get(species, {}).get(contrib_type, {})

                    for rxn_label in reactions:
                        rate = type_data.get(rxn_label, 0.0)
                        row[rxn_label] = rate

                    rows.append(row)

                # Create DataFrame and save
                df = pd.DataFrame(rows)

                # Reorder columns: time first, then reactions
                cols = ['time'] + reactions
                df = df[cols]

                # Save to CSV
                filename = f"{safe_name}_{contrib_type}.csv"
                filepath = os.path.join(save_dir, filename)
                df.to_csv(filepath, index=False)

        self.logger.info(f"Exported per-species diagnostics to {save_dir}")

    def _export_diagnostics_txt(self, path: str, top_n: int = 10):
        """
        Export summary diagnostic contributions to TXT.

        Includes:
        1. Time-averaged summary (across all timesteps)
        2. Final timestep snapshot
        3. Time range information
        """
        # Get time range
        times = [d['time'] for d in self.diagnostic_data]
        t_start, t_end = min(times), max(times)
        n_timesteps = len(self.diagnostic_data)

        # Aggregate contributions across all timesteps (for averaging)
        aggregated = {}
        for species in self.chemistry.diagnostic_species:
            aggregated[species] = {
                'production': defaultdict(list),
                'consumption': defaultdict(list)
            }

        for data_point in self.diagnostic_data:
            diagnostics = data_point['diagnostics']
            for species, type_data in diagnostics.items():
                for rxn_label, rate in type_data['production'].items():
                    aggregated[species]['production'][rxn_label].append(rate)
                for rxn_label, rate in type_data['consumption'].items():
                    aggregated[species]['consumption'][rxn_label].append(rate)

        # Calculate average rates
        averaged = {}
        for species in self.chemistry.diagnostic_species:
            averaged[species] = {'production': {}, 'consumption': {}}
            for rxn_label, rates in aggregated[species]['production'].items():
                averaged[species]['production'][rxn_label] = np.mean(rates)
            for rxn_label, rates in aggregated[species]['consumption'].items():
                averaged[species]['consumption'][rxn_label] = np.mean(rates)

        # Get final timestep data
        final_data = self.diagnostic_data[-1]['diagnostics']
        final_time = self.diagnostic_data[-1]['time']

        # Write summary file
        with open(path, 'w', encoding='utf-8') as f:
            mode_str = "Saline Solution" if self.saline_mode else "DI Water"
            f.write(f"{'=' * 80}\n")
            f.write(f"Reaction Contribution Analysis ({mode_str})\n")
            f.write(f"{'=' * 80}\n\n")
            f.write(f"Based on: Liu 2015 (DI Water) / Liu 2016 (Saline) methodology\n")
            f.write(f"Generated: {generate_timestamp()}\n\n")
            f.write(f"Time range: {t_start:.3f} ~ {t_end:.3f} s\n")
            f.write(f"Total timesteps: {n_timesteps}\n")
            f.write(f"Final timestep: t = {final_time:.3f} s\n\n")
            f.write(f"Note: Values below are TIME-AVERAGED across all timesteps.\n")
            f.write(f"      For time-resolved data, see individual CSV files.\n")
            f.write("=" * 80 + "\n")

            for species in self.chemistry.diagnostic_species:
                f.write(f"\n{'=' * 25} {species} {'=' * 25}\n\n")

                # === TIME-AVERAGED ===
                f.write(f"[TIME-AVERAGED over {n_timesteps} timesteps]\n\n")

                # Production (averaged)
                prod_data = averaged[species]['production']
                total_prod = sum(prod_data.values())

                f.write(f"  Production (Total: {total_prod:.2e} mol/L/s)\n")
                if total_prod > 0:
                    sorted_prod = sorted(prod_data.items(), key=lambda x: x[1], reverse=True)[:top_n]
                    for i, (rxn_label, rate) in enumerate(sorted_prod, 1):
                        pct = (rate / total_prod * 100) if total_prod > 0 else 0
                        f.write(f"    {i:2d}. {pct:5.1f}%  {rate:.2e}  {rxn_label}\n")
                else:
                    f.write("    (No production reactions)\n")

                # Consumption (averaged)
                cons_data = averaged[species]['consumption']
                total_cons = sum(cons_data.values())

                f.write(f"\n  Consumption (Total: {total_cons:.2e} mol/L/s)\n")
                if total_cons > 0:
                    sorted_cons = sorted(cons_data.items(), key=lambda x: x[1], reverse=True)[:top_n]
                    for i, (rxn_label, rate) in enumerate(sorted_cons, 1):
                        pct = (rate / total_cons * 100) if total_cons > 0 else 0
                        f.write(f"    {i:2d}. {pct:5.1f}%  {rate:.2e}  {rxn_label}\n")
                else:
                    f.write("    (No consumption reactions)\n")

                # Net rate (averaged)
                net_rate = total_prod - total_cons
                f.write(f"\n  Net rate: {net_rate:+.2e} mol/L/s\n")

                # === FINAL TIMESTEP ===
                f.write(f"\n[FINAL TIMESTEP: t = {final_time:.3f} s]\n\n")

                final_species = final_data.get(species, {'production': {}, 'consumption': {}})

                # Production (final)
                final_prod = final_species.get('production', {})
                total_prod_final = sum(final_prod.values())

                f.write(f"  Production (Total: {total_prod_final:.2e} mol/L/s)\n")
                if total_prod_final > 0:
                    sorted_prod_final = sorted(final_prod.items(), key=lambda x: x[1], reverse=True)[:5]
                    for i, (rxn_label, rate) in enumerate(sorted_prod_final, 1):
                        pct = (rate / total_prod_final * 100) if total_prod_final > 0 else 0
                        f.write(f"    {i:2d}. {pct:5.1f}%  {rate:.2e}  {rxn_label}\n")
                else:
                    f.write("    (No production reactions)\n")

                # Consumption (final)
                final_cons = final_species.get('consumption', {})
                total_cons_final = sum(final_cons.values())

                f.write(f"\n  Consumption (Total: {total_cons_final:.2e} mol/L/s)\n")
                if total_cons_final > 0:
                    sorted_cons_final = sorted(final_cons.items(), key=lambda x: x[1], reverse=True)[:5]
                    for i, (rxn_label, rate) in enumerate(sorted_cons_final, 1):
                        pct = (rate / total_cons_final * 100) if total_cons_final > 0 else 0
                        f.write(f"    {i:2d}. {pct:5.1f}%  {rate:.2e}  {rxn_label}\n")
                else:
                    f.write("    (No consumption reactions)\n")

                net_final = total_prod_final - total_cons_final
                f.write(f"\n  Net rate: {net_final:+.2e} mol/L/s\n")

        self.logger.info(f"Exported summary diagnostics to {path}")

    def _reset(self):
        """Reset application state."""
        self.file_path = None
        self.raw_df = None
        self.preprocessed_df = None
        self.results_df = None
        self.detailed_results_df = None
        self.contribution_data = []
        self.diagnostic_data = []
        self.mass_transfer_data = []

        self.file_label.config(text="No file selected")
        self.preprocess_text.delete(1.0, tk.END)
        self.result_text.delete(1.0, tk.END)
        self.contribution_text.delete(1.0, tk.END)

        self._show_welcome_message()
