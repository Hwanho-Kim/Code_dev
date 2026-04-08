"""
Gas Phase Data Preprocessor for Plasma-Liquid Interaction simulation.

This module handles preprocessing of gas phase chemical species data,
including outlier removal, smoothing, and estimation of missing species.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from config import (
    GAS_PHASE_SPECIES, DEFAULTS, PHYSICAL
)
from chemistry_utils import (
    estimate_n2o4_from_no2,
    estimate_h2o2_from_o3,
    calculate_hono_hono2,
    calculate_n2o4_equilibrium_constant
)
from utils import get_logger, log_execution


@dataclass
class PreprocessParams:
    """Parameters for data preprocessing"""
    temperature_K: float = 298.15
    humidity: float = 0.5
    smooth: bool = True
    smooth_window: int = 11
    estimate_missing: bool = True
    o3_h2o2_ratio: float = 5000.0
    outlier_iqr_factor: float = 3.0


class GasPhasePreprocessor:
    """
    Gas phase chemical species data preprocessor.

    Handles:
    - Outlier removal using IQR method
    - Data smoothing using rolling mean
    - Estimation of missing species (N2O4, HONO, HONO2, H2O2)
    - Unit conversions
    """

    def __init__(self, T: float = 298.15):
        """
        Initialize preprocessor.

        Parameters
        ----------
        T : float
            Temperature in Kelvin
        """
        self.T = T
        self.logger = get_logger()

    def preprocess(
        self,
        df: pd.DataFrame,
        params: Optional[PreprocessParams] = None
    ) -> pd.DataFrame:
        """
        Main preprocessing pipeline.

        Parameters
        ----------
        df : pd.DataFrame
            Raw gas phase data with species columns in molecules/cm³
        params : PreprocessParams, optional
            Preprocessing parameters

        Returns
        -------
        pd.DataFrame
            Preprocessed data
        """
        if params is None:
            params = PreprocessParams()

        self.T = params.temperature_K
        self.logger.info(f"Starting preprocessing: shape={df.shape}, T={self.T}K")

        df_processed = df.copy()

        # Step 1: Remove outliers
        df_processed = self._remove_outliers(
            df_processed,
            GAS_PHASE_SPECIES,
            params.outlier_iqr_factor
        )

        # Step 2: Smooth data
        if params.smooth:
            df_processed = self._apply_smoothing(
                df_processed,
                GAS_PHASE_SPECIES,
                params.smooth_window
            )

        # Step 3: Estimate missing species
        if params.estimate_missing:
            df_processed = self._estimate_n2o4(df_processed, params.temperature_K)
            df_processed = self._estimate_hono_hono2(
                df_processed,
                params.humidity,
                params.temperature_K
            )
            df_processed = self._estimate_h2o2(df_processed, params.o3_h2o2_ratio)

        # Step 4: Identify plasma periods
        df_processed = self._identify_plasma_periods(df_processed)

        # Step 5: Convert units
        df_processed = self._convert_units(df_processed, GAS_PHASE_SPECIES)

        self.logger.info(f"Preprocessing complete: final shape={df_processed.shape}")
        return df_processed

    def _remove_outliers(
        self,
        df: pd.DataFrame,
        species_list: List[str],
        iqr_factor: float = 3.0
    ) -> pd.DataFrame:
        """
        Remove outliers using IQR method.

        Parameters
        ----------
        df : pd.DataFrame
            Input data
        species_list : List[str]
            List of species columns to process
        iqr_factor : float
            IQR multiplier for outlier detection

        Returns
        -------
        pd.DataFrame
            Data with outliers removed
        """
        self.logger.debug("Removing outliers...")

        for species in species_list:
            if species not in df.columns:
                continue

            data = df[species].values

            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            IQR = Q3 - Q1

            lower_bound = Q1 - iqr_factor * IQR
            upper_bound = Q3 + iqr_factor * IQR

            # Clip negative values to 0, outliers to upper bound
            df[species] = np.clip(data, 0, upper_bound)

            self.logger.debug(f"  {species}: IQR={IQR:.2e}, bounds=[{lower_bound:.2e}, {upper_bound:.2e}]")

        return df

    def _apply_smoothing(
        self,
        df: pd.DataFrame,
        species_list: List[str],
        window_size: int = 11
    ) -> pd.DataFrame:
        """
        Apply rolling mean smoothing to data.

        Parameters
        ----------
        df : pd.DataFrame
            Input data
        species_list : List[str]
            List of species columns to smooth
        window_size : int
            Rolling window size

        Returns
        -------
        pd.DataFrame
            Smoothed data
        """
        self.logger.debug(f"Applying smoothing with window={window_size}...")

        for species in species_list:
            if species not in df.columns:
                continue

            data = df[species].values

            # Skip if too few data points
            if len(data) < window_size:
                self.logger.debug(f"  {species}: Skipped (insufficient data)")
                continue

            try:
                # Handle NaN/inf
                data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

                # Use pandas rolling mean (stable)
                series = pd.Series(data)
                smoothed = series.rolling(
                    window=window_size,
                    center=True,
                    min_periods=1
                ).mean().values

                # Ensure non-negative
                smoothed = np.maximum(smoothed, 0)

                df[species] = smoothed
                self.logger.debug(f"  {species}: Smoothing complete")

            except Exception as e:
                self.logger.warning(f"  {species}: Smoothing failed - {e}")

        return df

    def _estimate_n2o4(
        self,
        df: pd.DataFrame,
        T: float
    ) -> pd.DataFrame:
        """
        Estimate N2O4 from NO2 using equilibrium.

        Parameters
        ----------
        df : pd.DataFrame
            Input data
        T : float
            Temperature in Kelvin

        Returns
        -------
        pd.DataFrame
            Data with N2O4 estimated
        """
        if 'NO2' not in df.columns:
            return df

        self.logger.debug("Estimating N2O4 from NO2 equilibrium...")

        # Check if N2O4 column exists
        if 'N2O4' in df.columns:
            # Only estimate where measured value is 0
            mask_zero = df['N2O4'] == 0

            df.loc[mask_zero, 'N2O4'] = df.loc[mask_zero, 'NO2'].apply(
                lambda x: estimate_n2o4_from_no2(x, T) if pd.notna(x) and x > 0 else 0
            )

            df['N2O4_source'] = 'measured'
            df.loc[mask_zero, 'N2O4_source'] = 'equilibrium'

            n_measured = (~mask_zero).sum()
            n_estimated = mask_zero.sum()
            self.logger.info(f"N2O4: {n_measured} measured, {n_estimated} estimated")
        else:
            # Create N2O4 column entirely from equilibrium
            df['N2O4'] = df['NO2'].apply(
                lambda x: estimate_n2o4_from_no2(x, T) if pd.notna(x) and x > 0 else 0
            )
            df['N2O4_source'] = 'equilibrium'
            self.logger.info(f"N2O4: All {len(df)} points estimated from equilibrium")

        # Store equilibrium values for comparison
        df['N2O4_equilibrium'] = df['NO2'].apply(
            lambda x: estimate_n2o4_from_no2(x, T) if pd.notna(x) and x > 0 else 0
        )

        # Log equilibrium check
        if len(df) > 0:
            sample_idx = df.index[0]
            NO2 = df.loc[sample_idx, 'NO2']
            N2O4 = df.loc[sample_idx, 'N2O4']
            Kp = calculate_n2o4_equilibrium_constant(T)
            self.logger.debug(f"  Equilibrium check at T={T:.1f}K: Kp={Kp:.2f}")
            self.logger.debug(f"  NO2={NO2:.2e}, N2O4={N2O4:.2e}")

        return df

    def _estimate_hono_hono2(
        self,
        df: pd.DataFrame,
        humidity: float,
        T: float
    ) -> pd.DataFrame:
        """
        Estimate HONO and HONO2 from hydrolysis reactions.

        Parameters
        ----------
        df : pd.DataFrame
            Input data
        humidity : float
            Relative humidity (0-1)
        T : float
            Temperature in Kelvin

        Returns
        -------
        pd.DataFrame
            Data with HONO/HONO2 estimated
        """
        self.logger.debug("Estimating HONO/HONO2...")

        # Get concentration series
        N2O4_values = df.get('N2O4', pd.Series(0, index=df.index))
        N2O5_values = df.get('N2O5', pd.Series(0, index=df.index))
        NO2_values = df.get('NO2', pd.Series(0, index=df.index))

        # Initialize columns if not present
        if 'HONO' not in df.columns:
            df['HONO'] = 0.0
        if 'HONO2' not in df.columns:
            df['HONO2'] = 0.0

        # Process in batches for efficiency
        batch_size = 1000
        total_rows = len(df)

        for batch_start in range(0, total_rows, batch_size):
            batch_end = min(batch_start + batch_size, total_rows)
            batch_indices = df.index[batch_start:batch_end]

            for idx in batch_indices:
                try:
                    N2O4 = N2O4_values.loc[idx] if pd.notna(N2O4_values.loc[idx]) else 0
                    N2O5 = N2O5_values.loc[idx] if pd.notna(N2O5_values.loc[idx]) else 0
                    NO2 = NO2_values.loc[idx] if pd.notna(NO2_values.loc[idx]) else 0

                    HONO_calc, HONO2_calc = calculate_hono_hono2(
                        N2O4, N2O5, NO2, humidity, T
                    )

                    # Use estimated value if current is 0 or NaN
                    current_hono = df.loc[idx, 'HONO']
                    if pd.isna(current_hono) or current_hono == 0:
                        df.loc[idx, 'HONO'] = HONO_calc
                    else:
                        df.loc[idx, 'HONO'] = max(current_hono, HONO_calc)

                    current_hono2 = df.loc[idx, 'HONO2']
                    if pd.isna(current_hono2) or current_hono2 == 0:
                        df.loc[idx, 'HONO2'] = HONO2_calc
                    else:
                        df.loc[idx, 'HONO2'] = max(current_hono2, HONO2_calc)

                except Exception as e:
                    self.logger.warning(f"  Row {idx} processing error: {e}")
                    continue

        self.logger.debug("HONO/HONO2 estimation complete")
        return df

    def _estimate_h2o2(
        self,
        df: pd.DataFrame,
        o3_h2o2_ratio: float
    ) -> pd.DataFrame:
        """
        Estimate H2O2 from O3 concentration.

        Parameters
        ----------
        df : pd.DataFrame
            Input data
        o3_h2o2_ratio : float
            O3/H2O2 ratio

        Returns
        -------
        pd.DataFrame
            Data with H2O2 estimated
        """
        if 'O3' not in df.columns:
            return df

        self.logger.debug(f"Estimating H2O2 from O3 (ratio={o3_h2o2_ratio})...")

        df['H2O2'] = df['O3'].apply(
            lambda x: estimate_h2o2_from_o3(x, o3_h2o2_ratio) if pd.notna(x) and x > 0 else 0
        )

        return df

    def _identify_plasma_periods(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify plasma ON/OFF periods based on O3 concentration.

        Parameters
        ----------
        df : pd.DataFrame
            Input data

        Returns
        -------
        pd.DataFrame
            Data with plasma_on column
        """
        if 'O3' not in df.columns:
            df['plasma_on'] = False
            return df

        try:
            threshold = df['O3'].quantile(0.1)
            df['plasma_on'] = df['O3'] > threshold
            self.logger.debug(f"Plasma period threshold: O3 > {threshold:.2e}")
        except Exception as e:
            self.logger.warning(f"Plasma period identification failed: {e}")
            df['plasma_on'] = False

        return df

    def _convert_units(
        self,
        df: pd.DataFrame,
        species_list: List[str]
    ) -> pd.DataFrame:
        """
        Add mol/L columns for all species.

        molecules/cm³ → mol/L conversion factor: 1.66e-21

        Parameters
        ----------
        df : pd.DataFrame
            Input data
        species_list : List[str]
            List of species columns

        Returns
        -------
        pd.DataFrame
            Data with additional _mol_L columns
        """
        self.logger.debug("Converting units to mol/L...")

        conversion_factor = PHYSICAL.MOLECULES_TO_MOL_L

        for species in species_list:
            if species in df.columns:
                df[f'{species}_mol_L'] = df[species] * conversion_factor

        return df


# Legacy function for backward compatibility
def preprocess_batch_data(
    df: pd.DataFrame,
    T: float = 298.15,
    humidity: float = 0.5,
    smooth: bool = True,
    estimate_missing: bool = True,
    o3_h2o2_ratio: float = 5000.0
) -> pd.DataFrame:
    """
    Legacy wrapper for backward compatibility.

    Parameters
    ----------
    df : pd.DataFrame
        Input data
    T : float
        Temperature in Kelvin
    humidity : float
        Relative humidity
    smooth : bool
        Whether to smooth data
    estimate_missing : bool
        Whether to estimate missing species
    o3_h2o2_ratio : float
        O3/H2O2 ratio

    Returns
    -------
    pd.DataFrame
        Preprocessed data
    """
    params = PreprocessParams(
        temperature_K=T,
        humidity=humidity,
        smooth=smooth,
        estimate_missing=estimate_missing,
        o3_h2o2_ratio=o3_h2o2_ratio
    )

    preprocessor = GasPhasePreprocessor(T)
    return preprocessor.preprocess(df, params)
