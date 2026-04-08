#!/usr/bin/env python3
"""
Quick Test - 핵심 파라미터 빠른 검증 스크립트

Usage:
    python quick_test.py              # 기본 설정으로 실행
    python quick_test.py --pH 5.0     # 초기 pH 지정
    python quick_test.py -v           # 상세 출력
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from chemistry import CompleteAqueousChemistry
from chemistry_utils import apply_henry_law


# 예상 범위 (필요시 수정)
EXPECTED_RANGES = {
    'pH': (2.0, 7.0),
    'H2O2': (1e-10, 1e-2),
    'NO2-': (1e-12, 1e-2),
    'NO3-': (1e-12, 1e-2),
}


def check_range(name: str, value: float) -> str:
    """값이 예상 범위 내인지 확인"""
    if name not in EXPECTED_RANGES:
        return ""
    low, high = EXPECTED_RANGES[name]
    if low <= value <= high:
        return "\033[92m✓\033[0m"  # 녹색 체크
    else:
        return "\033[91m✗\033[0m"  # 빨간 X


def format_conc(value: float) -> str:
    """농도를 읽기 쉽게 포맷"""
    if value < 1e-6:
        return f"{value:.2e} M"
    elif value < 1e-3:
        return f"{value*1e6:.2f} μM"
    else:
        return f"{value*1e3:.2f} mM"


def run_quick_test(initial_pH: float = 7.0, verbose: bool = False):
    """Quick test 실행"""

    print("\n" + "=" * 40)
    print("   NOx Analyzer - Quick Validation")
    print("=" * 40)

    # 기본 입력값 (플라즈마 처리 대표값)
    # 필요시 이 값들을 수정하세요
    gas_concentrations = {
        'NO': 1e14,      # molecules/cm³
        'NO2': 5e13,
        'O3': 1e13,
        'OH': 1e10,
        'HNO3': 1e12,
    }

    print(f"\n[입력 조건]")
    print(f"  초기 pH: {initial_pH}")
    if verbose:
        print(f"  Gas concentrations (molecules/cm³):")
        for species, conc in gas_concentrations.items():
            print(f"    {species}: {conc:.1e}")

    # 가스상 → 수용액상 변환 (Henry's law)
    C_aq_initial = {}
    for species, gas_conc in gas_concentrations.items():
        try:
            aq_conc = apply_henry_law(species, gas_conc)
            if aq_conc > 0:
                C_aq_initial[species] = aq_conc
        except:
            pass

    if verbose:
        print(f"\n  Aqueous initial (mol/L):")
        for species, conc in C_aq_initial.items():
            print(f"    {species}: {conc:.2e}")

    # Chemistry solver 실행
    print(f"\n[시뮬레이션 실행 중...]")
    chemistry = CompleteAqueousChemistry()
    C_final, contributions = chemistry.solve(C_aq_initial, initial_pH)

    # 핵심 파라미터 출력
    print(f"\n[결과 - 핵심 파라미터]")
    print("-" * 40)

    key_params = ['pH', 'H2O2', 'NO2-', 'NO3-']

    for param in key_params:
        if param in C_final:
            value = C_final[param]
            if param == 'pH':
                check = check_range('pH', value)
                print(f"  {param:8s}: {value:6.2f}       {check}")
            else:
                check = check_range(param, value)
                print(f"  {param:8s}: {format_conc(value):12s} {check}")
        else:
            print(f"  {param:8s}: N/A")

    print("-" * 40)

    # 추가 정보 (verbose 모드)
    if verbose:
        print(f"\n[추가 종 농도]")
        other_species = ['HONO', 'HNO3', 'ONOO-', 'OH', 'O3']
        for species in other_species:
            if species in C_final:
                print(f"  {species:8s}: {format_conc(C_final[species])}")

        # 반응 기여도
        if contributions:
            print(f"\n[주요 반응 기여도]")
            for species, rxns in contributions.items():
                if rxns:
                    print(f"\n  {species} 생성:")
                    for rxn, pct in list(rxns.items())[:3]:
                        print(f"    {pct:5.1f}% - {rxn}")

    print("\n" + "=" * 40)
    print("   테스트 완료")
    print("=" * 40 + "\n")

    return C_final


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quick validation for NOx chemistry")
    parser.add_argument('--pH', type=float, default=7.0, help='Initial pH (default: 7.0)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')

    args = parser.parse_args()
    run_quick_test(initial_pH=args.pH, verbose=args.verbose)
