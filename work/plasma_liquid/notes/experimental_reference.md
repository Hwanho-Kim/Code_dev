# Experimental Reference Data (OAS, Dry condition)

## Source
- Gas: `OAS data/Dry/(P-L) 가스활성종 농도.xlsx`
- Liquid: `OAS data/Dry/(P-L) 액체활성종 농도, pH, conductivity.xlsx`
- Treatment time: **10 min (600s)**
- Gas data interval: 2s, 301 points per voltage

## DIW Liquid Results

| Voltage (kVpp) | pH | NO₂⁻ (µM) | NO₃⁻ (µM) | H₂O₂ (µM) |
|----------------|-----|----------|----------|-----------|
| 2.6 | 5.09 | 0 | 32.63 | 4.76 |
| **3.2** | **3.61** | **3.58** | **62.74** | **11.21** |
| 3.6 | 3.25 | 20.74 | 70.42 | 16.25 |

## Saline Liquid Results

| Voltage (kVpp) | pH | NO₂⁻ (µM) | NO₃⁻ (µM) | H₂O₂ (µM) |
|----------------|-----|----------|----------|-----------|
| 2.6 | 5.15 | 0 | 4.70 | 2.00 |
| **3.2** | **3.60** | **0** | **10.45** | **5.14** |
| 3.6 | 3.43 | 0 | 16.92 | 7.73 |

## pH

| Voltage | DIW | Saline |
|---------|-----|--------|
| control | 6.47 | 6.67 |
| 2.6 | 5.09 | 5.15 |
| 3.2 | 3.61 | 3.60 |
| 3.6 | 3.25 | 3.43 |

## Conductivity

| Voltage | DIW (µS) | Saline (mS) |
|---------|----------|-------------|
| control | 9.23 | 14.27 |
| 2.6 | 12.23 | 14.30 |
| 3.2 | 123.87 | 14.23 |
| 3.6 | 226.67 | 14.47 |

## Gas-phase Data (cm⁻³)

### 3.2 kVpp (reference)
| Species | Max concentration |
|---------|------------------|
| O₃ | ~1.05×10¹⁷ |
| NO₂ | ~1.88×10¹⁵ |
| NO₃ | ~1.64×10¹⁴ |
| N₂O₅ | ~2.53×10¹⁵ |

**NOTE**: 이전 CSV (1kHz3.2kVpp.csv)는 0~720s(12min, 361점).
새 OAS data는 0~600s(10min, 301점). **새 데이터가 정확한 reference.**

## 이전 근사값과의 차이

| | 이전 (CLAUDE.md) | 새 OAS data | 차이 |
|---|---|---|---|
| DIW NO₂⁻ | 3 µM | 3.58 µM | +19% |
| DIW NO₃⁻ | 63 µM | 62.74 µM | -0.4% |
| DIW H₂O₂ | 11 µM | 11.21 µM | +2% |
| DIW pH | 3.61 | 3.61 | 동일 |
| Saline NO₃⁻ | 102 µM | 10.45 µM | **대폭 차이!** |
| Saline H₂O₂ | 5 µM | 5.14 µM | +3% |
| Treatment time | 12 min | 10 min | -2 min |

**중요**: Saline NO₃⁻가 이전 102µM → 10.45µM으로 대폭 감소. 이전 값 출처 재확인 필요.

---
<!-- Last updated: 2026-04-14 -->
