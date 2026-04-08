#!/usr/bin/env python3
"""
NOx Analyzer - Plasma-Liquid Interaction Simulation

Main entry point for the application.

Usage:
    python main.py         # Launch GUI
    python main.py --help  # Show help
"""

import sys
import traceback
from pathlib import Path

# Add package directory to path
sys.path.insert(0, str(Path(__file__).parent))


def main():
    """Main entry point."""
    import tkinter as tk

    from utils import setup_global_exception_handler, get_logger
    from gui import NOxAnalyzerGUI
    from config import GUI_CONFIG

    # Setup logging and exception handling
    log_dir = Path(__file__).parent
    setup_global_exception_handler(log_dir / "error_log.txt")
    logger = get_logger()

    root = None

    try:
        logger.info(f"Starting NOx Analyzer v{GUI_CONFIG.version}")

        # Create root window
        root = tk.Tk()

        # Prevent window from closing unexpectedly
        root.protocol("WM_DELETE_WINDOW", lambda: root.quit())

        # Create application
        app = NOxAnalyzerGUI(root)

        # Update GUI
        root.update_idletasks()

        logger.info("GUI initialized successfully")

        # Start main loop
        root.mainloop()

    except KeyboardInterrupt:
        logger.info("Application interrupted by user")

    except Exception as e:
        error_msg = f"Fatal error:\n{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        logger.error(error_msg)
        print(error_msg)

        try:
            if root and root.winfo_exists():
                from tkinter import messagebox
                messagebox.showerror("Fatal Error", f"Application crashed:\n{str(e)}")
        except Exception:
            pass

    finally:
        # Cleanup
        try:
            if root and root.winfo_exists():
                root.destroy()
        except Exception:
            pass

        logger.info("Application terminated")


def run_cli():
    """Run in CLI mode (for testing/batch processing)."""
    import argparse

    parser = argparse.ArgumentParser(
        description="NOx Analyzer - Plasma-Liquid Interaction Simulation"
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        help='Input CSV file path'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output directory'
    )
    parser.add_argument(
        '--temperature', '-t',
        type=float,
        default=25.0,
        help='Temperature in Celsius (default: 25)'
    )
    parser.add_argument(
        '--humidity',
        type=float,
        default=50.0,
        help='Humidity in percent (default: 50)'
    )
    parser.add_argument(
        '--pH',
        type=float,
        default=7.0,
        help='Initial pH (default: 7.0)'
    )
    parser.add_argument(
        '--gui',
        action='store_true',
        help='Launch GUI mode'
    )

    args = parser.parse_args()

    if args.gui or (not args.input):
        main()
    else:
        # Batch processing mode
        batch_process(args)


def batch_process(args):
    """Run batch processing without GUI."""
    import pandas as pd

    from utils import get_logger
    from preprocessor import GasPhasePreprocessor, PreprocessParams
    from chemistry import CompleteAqueousChemistry
    from chemistry_utils import apply_henry_law, celsius_to_kelvin
    from config import GAS_PHASE_SPECIES

    logger = get_logger()

    logger.info(f"Batch processing: {args.input}")

    # Load data
    df = pd.read_csv(args.input)
    logger.info(f"Loaded {len(df)} rows")

    # Preprocess
    params = PreprocessParams(
        temperature_K=celsius_to_kelvin(args.temperature),
        humidity=args.humidity / 100.0
    )

    preprocessor = GasPhasePreprocessor()
    df_processed = preprocessor.preprocess(df, params)
    logger.info("Preprocessing complete")

    # Analyze
    chemistry = CompleteAqueousChemistry()
    results = []

    for idx, row in df_processed.iterrows():
        if idx % 100 == 0:
            logger.info(f"Processing row {idx}/{len(df_processed)}")

        try:
            C_aq_initial = {}
            for species in GAS_PHASE_SPECIES:
                if species in row:
                    gas_conc = row[species]
                    if pd.notna(gas_conc) and gas_conc > 0:
                        C_aq_initial[species] = apply_henry_law(species, gas_conc)

            C_final, _ = chemistry.solve(C_aq_initial, args.pH)

            result_row = {'index': idx}
            result_row.update(C_final)
            results.append(result_row)

        except Exception as e:
            logger.warning(f"Row {idx} error: {e}")

    # Save results
    results_df = pd.DataFrame(results)

    if args.output:
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)

        preprocessed_path = output_path / "preprocessed.csv"
        df_processed.to_csv(preprocessed_path, index=False)
        logger.info(f"Saved preprocessed data to {preprocessed_path}")

        results_path = output_path / "results.csv"
        results_df.to_csv(results_path, index=False)
        logger.info(f"Saved results to {results_path}")
    else:
        print(results_df.to_string())

    logger.info("Batch processing complete")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] not in ['--gui', '-h', '--help']:
        run_cli()
    else:
        main()
