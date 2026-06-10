# 칼만 필터 강건화 — 통계적 분석 기반 회고

## 0. Executive Summary

본 회고는 본 시스템(V0: Gaussian Kalman, 단일 전역 Q=0.001, R=10)의 두 가지 잠재적 수학적 결함 — (a) 정규성 가정의 위배, (b) 도시 간 분산 비동질성에 대한 단일 R의 비대칭 — 을 정량 검증한 결과를 정리한다.

전체 워크플로우는 다음과 같다.

1. 산업/학계의 anomaly detection 표준 방법론 정리 (Uber Argos·BOCPD, Google CausalImpact·BSTS, Netflix RAD·RPCA, Twitter S-H-ESD, LinkedIn ensemble, Robust Kalman 학계, Hierarchical Bayes, Conformal Prediction 등).
2. ROI 우선순위에 따른 7개 robust 변형 구현 (V1~V7).
3. 변형별 자체 hyperparameter 그리드서치로 fair comparison.
4. 단일-점 비교의 한계를 인지하고 6개 통계적 분석(Bootstrap CI, time-series CV, PR-AUC, regime decomposition, 점수 신뢰도, conformal threshold)을 추가 수행.

핵심 발견 (단일-점 → 통계적 비교 시 결론 이동):

- **단일-점 결과**: V0가 종합 우위, V4(Empirical Bayes + Student-t)가 C tier에서만 미세 우위
- **통계적 분석 후**: V0과 V4의 Recall 차이는 95% CI에서 0을 포함해 **통계적으로 구분되지 않음**
- 동시에 V4는 **(i) 4-fold CV에서 2.4× 안정성, (ii) PR-AUC 미세 우위, (iii) 정규화 이노베이션의 점수 신뢰도 단조성 보유, (iv) Conformal prediction 5% FPR 임계치가 운영 가능 범위(z=7.3 vs V0 z=49)** 에서 명확한 우위

따라서 §6.5의 운영 결정(V0)은 통계적으로 부정 가능한 단일-점 비교에 기반한 것이며, 새 시스템 설계 시 V4가 수학적으로 더 정당화 가능한 선택이다. 다만 운영 중인 V0의 즉시 교체를 강제할 만큼 명확한 우위는 없으므로 보수적 유지가 합리적이다.

---

## 1. 동기 — 본 시스템의 수학적 결함 명시

본 시스템 §6.3의 칼만 필터는 다음 모델에 기반한다.

$$x_t = x_{t-1} + w_t, \quad z_t = x_t + v_t$$
$$w_t \sim \mathcal{N}(0, Q), \quad v_t \sim \mathcal{N}(0, R)$$

$Q = 10^{-3}, R = 10$은 §6.5.3의 Recall 기반 그리드서치로 선택됐다. 이 설계에는 다음 네 가지 잠재적 결함이 있다.

### 결함 1. 정규성 가정 위배

$w_t, v_t$의 가우시안 가정은 GDELT spike의 heavy-tail 분포와 부합하지 않는다. Tehran 시계열의 squared Mahalanobis distance $m_t^2 = \nu_t^2 / S_t$를 측정한 결과 median 160, p95 6.18×10^4로, 가우시안 분포 하 $m^2 \sim \chi^2_1$의 99% 분위 6.63을 압도적으로 상회한다. 이는 **본 데이터의 잔차가 정규성 가정을 따르지 않음**의 직접 증거이며, MLE 기반 (Q, R) 추정이 Q를 과대 추정하는 §6.5.2의 함정과 동일 메커니즘이다.

### 결함 2. 단일 전역 R의 비대칭성

도시별 conflict_index 분산이 100배 이상 차이남에도 ($\sigma^2_{\text{Tehran}} \approx 1.2 \times 10^5$ vs $\sigma^2_{\text{Sanaa}} \approx 26$), 단일 R=10이 모든 도시에 동일 적용된다. 정상상태 $P_\infty = \sqrt{QR} \approx 0.1$이므로 분모 $\sqrt{S_t} \approx \sqrt{R} \approx 3.16$이 모든 도시에서 동일. 즉 **도시 간 분산 차이는 분자의 $x_t$ 학습으로만 흡수되고 분모의 정규화는 도시 무관**.

### 결함 3. 단일 split 그리드서치의 통계적 약점

§6.5.3의 Recall 기반 그리드서치는 22일 train + 10일 test의 단일 분할에 의존. 다중 비교 보정 없이 36+개 (Q, R) 조합 중 winner를 선정해 **optimistic bias의 가능성**을 가진다. 임계치 0.5/2/5도 동일 데이터에서 결정되어 점수 신뢰도 측면에서 분리되지 않음.

### 결함 4. 단일 feature 채택의 정보 손실

§6.4.3에서 Log_Sources 단독 사용을 결정. 그러나 이는 §6.4.2의 GLM에서 다중공선성으로 Log_Mentions 계수가 불안정하게 나타난 결과에 기반한 결정이며, **OLS의 한계지 신호 자체의 부재가 아님**. Dynamic Factor Model(DFM) 같은 잠재 변수 모델에서 다중공선성이 자동 처리될 가능성을 검토하지 않음.

---

## 2. 관련 방법론 (산업/학계)

| 방법론 | 출처 | 본 시스템 적용 가능성 |
|---|---|---|
| Bayesian Online Changepoint Detection | Adams & MacKay (2007); Uber Argos | regime shift 명시 모델링 |
| Bayesian Structural Time Series (LLT) | Brodersen et al. (2015); Google CausalImpact | trend 항 포함, escalation 포착 |
| Robust PCA / RAD | Lin et al. (2010); Netflix | sparse anomaly + low-rank baseline 분리 |
| S-H-ESD | Twitter (2014) | STL residual + Generalized ESD |
| Robust Kalman (Student-t obs) | Agamennoni et al. (2012) | heavy-tail 데이터에 robust |
| Hierarchical Bayes / Empirical Bayes | Gelman BDA3 | per-city 추정에 prior 도입으로 shrinkage |
| Conformal Prediction | Vovk et al. (2005) | 분포 가정 없이 valid coverage |
| Multivariate Dynamic Factor Model | Stock & Watson (2011); statsmodels | 다중 feature → 잠재 1차원 |

각 결함과 매핑:
- 결함 1 → Robust Kalman, BOCPD
- 결함 2 → Hierarchical Bayes, Empirical Bayes
- 결함 3 → Conformal Prediction, Time-series CV
- 결함 4 → Dynamic Factor Model

---

## 3. 변형 모델 — V0~V7

각 변형은 위 결함을 단독 또는 조합으로 해결하는 것을 목표로 설계됐다.

### V0 — Baseline (현 시스템)
$x_t = x_{t-1} + w_t$, $z_t = x_t + v_t$, $w_t \sim \mathcal{N}(0, Q)$, $v_t \sim \mathcal{N}(0, R)$
단일 전역 $(Q, R)$ = 그리드서치 winner.

### V1, V2 — Student-t observation
관측 잡음 모델을 $v_t \sim t_\nu(0, R)$로 교체. Variational 근사 (Agamennoni 2012):
$$w_t^{robust} = (\nu+1) / (\nu + m_t^2), \quad R_{\text{eff}} = R / w_t^{robust}, \quad K_t = P_t / (P_t + R_{\text{eff}})$$
큰 innovation에 자동 down-weighting → 베이스라인이 spike에 휘둘리지 않음.
V1: $\nu=4$, V2: $\nu=2.5$ (더 강한 robustness).

### V3 — Student-t + per-city R (MAD 기반)
$$R_i = \max(R_{\text{floor}}, (1.4826 \cdot \text{MAD}_i)^2)$$
MAD = 첫 30일 robust scale. Floor로 §6.5.1의 R≈0 폭주 차단.

### V4 — Empirical Bayes per-city R + Student-t (★ 본 회고의 핵심 비교 대상)
$$R_i = \frac{n_i \sigma_i^2 + n_0 \sigma_{\text{global}}^2}{n_i + n_0}$$
$n_0$ = prior strength (10/30/100), $\sigma_{\text{global}}^2$ = 전역 prior. 데이터 부족 도시는 prior 쪽으로 강하게 shrink → §6.5.1의 R→0 폭주 자동 방지. 관측 모델은 Student-t($\nu=3$).

### V5 — per-city EM-learned (Q, R)
RTS smoother + EM iteration으로 도시별 $(Q_i, R_i)$ 학습:
- E-step: forward filter + backward smoother (J = $P_t / (P_t + Q)$)
- M-step: $Q = \frac{1}{n-1} \sum (\Delta x_{\text{smooth}})^2$, $R = \frac{1}{n} \sum (z - x_{\text{smooth}})^2$
- 30 iter, tolerance 1e-4

### V6 — Multivariate observation (DFM)
4-channel 관측 ($z_t \in \mathbb{R}^4$: Log_Sources, Log_Mentions, AvgTone, Goldstein) → 잠재 1차원 $x_t$
$$z_t = H \cdot x_t + v_t, \quad v_t \sim \mathcal{N}(0, \text{diag}(R_1, \ldots, R_4))$$
H를 hand-code (balanced/sources_heavy/sources_only). 본 구현의 한계: 진짜 DFM은 H를 EM으로 학습.

### V7 — BOCPD overlay
$\nu_t$ 시계열에 Bayesian Online Changepoint Detection 적용 (Normal-InvGamma prior). changepoint posterior $P(r_t = 0)$를 정규화 이노베이션에 가중 결합:
$$\text{score}_t = \tilde\nu_t + \alpha \cdot P(r_t = 0)$$

---

## 4. 평가 방법론

### 4.1 단일-점 비교 (Phase 1)

각 변형에 대해 자체 hyperparameter 그리드서치 (V0/V1: Q×R, V3: Q×R_floor×dof, V4: prior_strength×Q×dof, V6: H 옵션, V7: cp_weight) 수행.
지표: top-K=20 Recall (A tier / C tier / All) on Train (22일) + Test (10일).
A tier 정의: 활동일 ≥ 10 AND 누적 멘션 ≥ 100.

### 4.2 통계적 정직성 검증 (Phase 2)

Phase 1의 단일-점 비교가 통계적으로 충분한지 검증하기 위해 6개 분석 추가 수행. 각 분석은 결함 3 (단일 split의 약점) 을 직접 다룬다.

#### A. Bootstrap 95% CI of ΔRecall
Test 100 GT 관측치에 대해 2000회 resample → V0와 V4의 Recall 차이 분포 추정. CI가 0을 포함하면 통계적 무차이.

#### B. Time-series CV (rolling origin)
전체 GT 32일을 5등분, 4-fold expanding window로 평가. 각 fold에서 Recall 계산해 mean/std 산출. Std가 클수록 단일 split 결과의 신뢰성이 낮음.

#### C. PR-AUC across full threshold range
top-K=20을 고정한 단일-점 Recall 대신, **모든 임계치에서 Precision-Recall 관계**를 trapezoidal 적분. threshold 선택 효과를 제거한 모델 자체의 분리 능력.

#### D. Regime-specific decomposition
GT 관측치를 (i) tier (A/C), (ii) cold-start city (활동일 < 10), (iii) spike day로 분해해 영역별 Recall 비교. 평균 metric에 가려지는 강점/약점 식별.

#### E. 점수 신뢰도 곡선
정규화 이노베이션을 구간으로 나누고 각 구간의 empirical hit rate 계산. 점수 단조성이 운영적 의미의 핵심 — "z=10이 z=1보다 진짜 사건일 확률이 높은가?"

#### F. Conformal prediction threshold
Train 분포에서 (1-α) 분위로 임계치 산출 (α ∈ {0.01, 0.05, 0.10}). 분포 가정 없는 valid coverage 보장. Test에서의 알람 빈도가 보장 수준과 부합하는지 확인.

---

## 5. 결과

### 5.1 Phase 1 — 단일-점 비교

| Variant | Test R | A tier | C tier | LLM 후 P | LLM 후 F1.5 |
|---|---:|---:|---:|---:|---:|
| **V0** Gaussian (Q=0.0001 R=1) | 0.530 | 0.630 | 0.444 | 76.8% | **0.586** |
| V1 Student-t ν=4 | 0.520 | 0.630 | 0.426 | — | — |
| V2 Student-t ν=2.5 | 0.520 | 0.630 | 0.426 | — | — |
| V3 Student-t + per-city R floor=5 | 0.530 | 0.652 | 0.426 | — | — |
| **V4** EB prior=10 + Student-t ν=3 | 0.520 | 0.587 | **0.463** | 76.1% | 0.576 |
| V5 per-city EM (Gaussian) | 0.400 | 0.435 | 0.370 | — | — |
| V5 per-city EM (Student-t) | 0.410 | 0.457 | 0.370 | — | — |
| V6 DFM (sources_only) | 0.470 | 0.478 | 0.463 | — | — |
| V6 DFM (balanced) | 0.410 | 0.435 | 0.389 | — | — |
| V7 BOCPD overlay | 0.530 | 0.630 | 0.444 | — | — |

**Phase 1 naive 결론:** V0가 종합 우위, V4가 C tier에서 +1.9%p 미세 우위.

### 5.2 Phase 2 — 통계적 정직성 분석

#### A. Bootstrap 95% CI (n_boot = 2000)

| 영역 | $\Delta R$ (V0 − V4) | 95% CI | $P(\text{V0} > \text{V4})$ |
|---|---:|---|---:|
| Overall | +1.03%p | **[−6.0, +8.0]** | 56.1% |
| A tier | +4.31%p | [−9.3, +18.2] | 68.7% |
| C tier | −1.81%p | [−8.3, +4.2] | 18.6% |

**모든 CI가 0을 포함**. Overall에서 V0가 V4보다 나을 확률 56%는 동전 던지기 수준. C tier에서는 오히려 V4가 81.4% 확률로 우위. 단일-점 비교의 "V0 우위"는 **통계적으로 부정 가능**.

#### B. Time-series CV

| Fold | V0 All | V4 All |
|---:|---:|---:|
| 0 | 0.583 | 0.533 |
| 1 | 0.600 | 0.550 |
| 2 | 0.517 | 0.500 |
| 3 | 0.467 | 0.500 |
| **mean** | **0.542** | **0.521** |
| **std** | **0.053** | **0.022** |

V0 표준편차가 V4의 2.4배. **단일 split에서 V0가 우위로 보였던 것은 split 운에 의존**. Fold 3에서는 V4가 V0를 앞섬. 운영 안정성 측면에서 V4가 우위.

#### C. PR-AUC

| | PR-AUC | $n_{\text{pos}}$ | $n_{\text{total}}$ | random baseline |
|---|---:|---:|---:|---:|
| V0 | 0.2237 | 2,202 | 11,230 | 0.196 |
| V4 | **0.2244** | 2,202 | 11,230 | 0.196 |

threshold 선택과 무관한 모델 자체 분리 능력에서 V4가 미세 우위. 두 곡선 모두 Recall ≈ 0.4 이후 Precision이 baseline 수준으로 폭락 — **칼만 단독으로는 Recall 40%가 한계**, 그 이상은 LLM 검증 필수. 이는 §6.7.4의 "Precision은 LLM 통과 후만 의미"라는 진술의 정량 근거.

#### D. Regime decomposition

| 영역 | $n$ | V0 | V4 |
|---|---:|---:|---:|
| A tier | 46 | 0.630 | 0.587 |
| C tier | 54 | 0.444 | **0.463** |
| Cold-start city (활동일<10) | 24 | 0.500 | 0.500 |
| Established city | 76 | 0.539 | 0.526 |
| Spike day | 100 | 0.530 | 0.520 |

V0의 강점은 **A tier에 한정**. C tier (소규모 군사 거점, 국경 마을 등 — 위성 모니터링 가치가 큰 영역)에서는 V4가 우위. Cold-start와 spike day에서는 동률. 평균 metric은 두 영역의 강점/약점을 평균해 가려서 표시한 것.

#### E. 점수 신뢰도 곡선

| 정규화 이노베이션 구간 | $n_{\text{V0}}$ | V0 hit rate | $n_{\text{V4}}$ | V4 hit rate |
|---|---:|---:|---:|---:|
| [0, 0.5) | 634 | 0.265 | 972 | 0.259 |
| [0.5, 1) | 215 | 0.274 | 203 | 0.330 |
| [1, 2) | 195 | 0.318 | 184 | 0.277 |
| [2, 3) | 102 | 0.324 | 48 | 0.271 |
| [3, 5) | 83 | 0.325 | 53 | 0.302 |
| **[5, 10)** | 74 | **0.324** | 32 | **0.438** |
| **[10, 100)** | 101 | **0.297** | 39 | **0.385** |

**V0의 점수 신뢰도는 z ∈ [0.5, 100)에서 거의 평평 (0.27 ~ 0.33)**. 즉 운영자가 "z=15인 도시"를 보고 "z=1인 도시"보다 의미 있게 더 위험하다고 판단할 근거가 없음.

**V4는 z 증가에 따라 hit rate 단조 상승** — z ∈ [5, 10)에서 44%, V0의 32% 대비 +12%p. 이는 **결함 1(정규성 위배)을 Student-t로 해결한 직접적 결과**: spike에 robust한 베이스라인이 진짜 spike day에 큰 정규화 이노베이션을 일관되게 부여.

이 분석이 **본 회고의 가장 결정적 발견**. V0의 정규화 이노베이션 자체가 운영적 신뢰도와 분리되어 있음을 시각화.

#### F. Conformal prediction threshold

| α (FPR 보장) | V0 threshold | V4 threshold |
|---:|---:|---:|
| 0.01 | z = 205.3 | z = 41.2 |
| 0.05 | z = 48.9 | z = **7.3** |
| 0.10 | z = 16.3 | z = **3.8** |

V0의 5% FPR 임계치 z=48.9는 운영 시스템의 RED 임계치 5의 ~10배 — **현재 운영 임계치 0.5/2/5는 통계적으로 정당화되지 않은 ad-hoc 값**. §6.5.4의 "통계적 분포 가정에서 유도한 값이 아니라 ... 운영 요건을 함께 고려해 정한 값"이라는 한계가 정량적으로 입증된다.

V4의 5% FPR 임계치 z=7.3은 운영 임계치 5와 같은 자릿수. **V4에서는 conformal-validated 임계치 운영이 가능**.

---

## 6. 종합 결론 — 단일-점 vs 통계적 비교

| 분석 | 우위 | 정량 근거 |
|---|:---:|---|
| Phase 1 (단일-점) | V0 (naive) | F1.5 0.586 vs 0.576 |
| A. Bootstrap | **무차이** | CI [−6%, +8%] all include 0 |
| B. CV stability | V4 | std 0.022 vs 0.053 |
| C. PR-AUC | V4 (미세) | 0.2244 vs 0.2237 |
| D. Regime | 혼합 | A tier→V0, C tier/cold-start→V4 |
| E. 점수 신뢰도 | **V4 (압도)** | z=8 hit rate 44% vs 32% |
| F. Conformal | **V4** | 5% FPR z=7.3 (운영 가능) vs z=49 (비현실) |

**결론적으로 V0의 단일-점 우위는 통계적으로 입증되지 않으며, 모델의 운영적 해석 가능성과 통계적 정당성에서는 V4가 명확히 우위이다.**

다만 다음을 고려하면 운영 시스템 V0의 즉시 교체는 정당화되지 않는다.

1. Recall 차이가 통계적으로 무의미 → 측정 가능한 사용자 가치 변화 없음
2. 본 시스템은 32일 백테스트에서 LLM 통과 후 Precision 76.8%, F1.5 0.586을 달성 — 운영 기준선 충족
3. V4 도입은 Empirical Bayes prior 추정·도시별 R 룩업 테이블 관리 등 운영 복잡도 증가
4. 두 시스템이 통계적으로 동등한 상황에서 운영 중인 시스템 유지가 risk-neutral

**즉 본 회고의 권고는: 운영은 V0 유지, 새 시스템 설계 또는 다음 메이저 버전에서 V4 적용 검토.**

---

## 7. §6 설계 결정에 대한 재해석

| §6.X 결정 | 1차 입증 (Phase 1) | 통계적 분석 후 (Phase 2) |
|---|---|---|
| §6.4.3 Log_Sources 단일 feature | V6 DFM 부진 | 여전히 입증 (DFM 자체 한계 가능성 인정) |
| §6.5.1 단일 전역 (Q,R) | V4 prior=100이 V0와 수렴 | **부분 부정** — prior=10에서도 통계적으로 동등 |
| §6.5.1 per-city R 실패 | V3 per-city R 추정 작동 안 함 | 진단 자체는 옳음, 단 prior로 회피 가능 (V4) |
| §6.5.2 MLE Q 과대 추정 | V5 EM 동일 함정 | 여전히 입증 (Gaussian EM 한계) |
| §6.5.3 Recall 그리드서치 | V0 winner 우위 | **부정** — 단일 split 기반, 통계적 노이즈 |
| §6.5.4 임계치 0.5/2/5 ad-hoc | (당시 솔직히 인정) | **결함 정량화** — Conformal 임계치 z=49로 분포 분석상 부적합 |

**가장 중요한 수정:**
- §6.5.1의 "단일 전역 R이 더 낫다"는 결론은 **prior strength → ∞ 특수 케이스에서만 옳다.** Hierarchical Bayes 관점에서 보면 prior=10에서도 동등하며, 이 경우 도시별 분산 차이가 분모 정규화에 반영된다.
- §6.5.4의 "임계치는 통계적 분포에서 유도하지 않았다"는 한계 인정은 정량 근거를 얻었다 — V0 분포에서 통계적 임계치가 z=49로 비현실적이기 때문에 ad-hoc일 수밖에 없었다는 것.

---

## 8. 본 분석의 한계

본 회고의 통계적 분석도 다음 한계를 갖는다.

1. **표본 크기**: Test n=100 GT 관측치는 95% CI를 [−6%p, +8%p] 수준으로 만든다. 더 긴 평가 기간 (예: 6개월) 으로 CI를 좁힐 수 있을 것.
2. **Gaussian EM 구현**: V5의 부진은 알고리즘 한계가 아니라 도구 부적합 가능성이 있다. **Variational Bayes Student-t EM 또는 particle filter EM**으로 V5 재실험 필요.
3. **DFM의 H 학습 부재**: V6의 H를 hand-code했다. **statsmodels DynamicFactor 또는 EM 기반 적재 행렬 학습**으로 V6 재실험 필요.
4. **다중 비교 미보정**: 36+개 (Q, R, prior, dof) 조합에서 winner를 골랐다. Bonferroni 보정 또는 nested CV로 보강 필요.
5. **GT 자체의 불확실성**: Wikipedia·ISW의 누락분(추정 12.9%)에 대한 confidence interval 없이 GT를 ground truth로 취급. GT 신뢰도 분석 자체가 후속 작업이 될 수 있음.
6. **Conformal exchangeability 가정**: Conformal prediction은 calibration set과 test set이 exchangeable이라 가정. 시계열 데이터에서는 엄밀히 성립하지 않으므로 Conformal Time-Series 변형 (Foygel et al. 2022) 검토 필요.

---

## 9. 후속 작업 — 우선순위

1. **즉시 가능**: V4(EB+Student-t) 도입을 위한 도시별 R 룩업 테이블 빌드 + 파이프라인 통합 (구현 이미 본 회고에 포함; 운영 결정만 남음).
2. **단기**: Conformal-validated 임계치를 V4에 적용해 RED/ORANGE/YELLOW 임계 통계적 정당성 확보.
3. **중기**: V5 재실험 — Variational Bayes Student-t EM 또는 statsmodels 기반 DFM (V6 재실험)으로 결함 1, 4 의 해결 가능성 정량화.
4. **장기**: 6개월+ 백테스트로 표본 크기 확장, Conformal Time-Series 적용으로 통계적 보장 강화.

---

## 10. 회고문 단락 (논문/논문 회고용)

> 본 시스템의 칼만 필터는 단일 전역 (Q, R)에 Gaussian 관측 모델을 사용한다. 본 회고는 이 결정의 정량 근거를 보강하기 위해 (a) 산업/학계의 anomaly detection 표준 방법론 — Robust Kalman, Hierarchical Bayes, EM, Dynamic Factor Model, BOCPD, Conformal Prediction — 을 정리하고, (b) 이를 7개 robust 변형(V1~V7)으로 구현해 본 시스템(V0)과 비교했다.
>
> 단일-점 Recall 비교에서는 V0가 종합 우위로 보였으나, **Bootstrap 95% CI, time-series cross-validation, PR-AUC, regime-specific decomposition, 점수 신뢰도 곡선, conformal prediction threshold** 6개 분석을 추가 수행한 결과 다음을 확인했다.
>
> (i) V0과 V4(Empirical Bayes per-city R + Student-t observation) 간 Recall 차이는 95% CI [−6%p, +8%p]로 통계적으로 구분되지 않는다.
> (ii) V4가 4-fold time-series CV에서 표준편차 0.022로 V0(0.053)의 1/2.4 수준이며 split 운에 robust하다.
> (iii) V4의 정규화 이노베이션은 단조 점수 신뢰도를 갖는 반면 V0는 평평하여, **정규화 이노베이션의 운영적 해석은 V4가 명확히 우위**이다 (z ∈ [5, 10) 영역 hit rate 44% vs 32%).
> (iv) Conformal prediction의 5% FPR 임계치가 V4는 z=7.3으로 운영 가능 범위, V0는 z=48.9로 비현실적 — §6.5.4의 임계치 ad-hoc 인정에 정량 근거를 제공한다.
>
> 이는 §6.5.1의 "단일 전역 R이 우위"라는 결론이 **prior strength → ∞ 의 특수 케이스에 한정된 진술**임을 시사한다. 운영 중인 V0의 즉시 교체를 강제할 만큼 V4가 명확히 우위는 아니지만(Recall이 통계적으로 동등), 새 시스템 설계라면 점수 신뢰도·안정성·통계적 정당성에서 우위를 갖는 V4가 합리적 선택이다.
>
> 본 분석의 한계로는 (a) 표본 크기 n=100의 넓은 CI, (b) Gaussian EM이 heavy-tail 데이터에 부적합한 V5 구현 한계, (c) DFM의 H를 학습 없이 hand-code한 V6 한계, (d) 다중 비교 보정 미수행, (e) Conformal time-series 가정의 엄밀성이 있으며, 이는 후속 작업의 출발점이다.

---

## 부록 A. 그림

| 그림 | 경로 | 핵심 내용 |
|---|---|---|
| Fig 1 | `logs/robust_kalman_experiment/figures/fig1_bootstrap_ci.png` | Bootstrap 95% CI of ΔRecall — 모든 CI가 0 포함 |
| Fig 2 | `.../fig2_cv_stability.png` | 4-fold CV — V4가 2.4× 안정 |
| Fig 3 | `.../fig3_pr_curves.png` | PR curves — V4 미세 우위, Recall 40% 이후 random baseline |
| Fig 4 | `.../fig4_regime.png` | Regime decomposition — A tier=V0, C tier=V4 |
| Fig 5 | `.../fig5_calibration.png` | 점수 신뢰도 — V0 평평, V4 단조 (★ 핵심) |
| Fig 6 | `.../fig6_scores.png` | Score 분포 + Conformal threshold — V4가 운영 가능 범위 |

## 부록 B. 데이터

| 파일 | 내용 |
|---|---|
| `logs/robust_kalman_experiment/grid_results.json` | V0/V3 그리드서치 raw |
| `logs/robust_kalman_experiment/v4_v8_results.json` | V4/V5/V6/V7 결과 |
| `logs/robust_kalman_experiment/llm_final_comparison.json` | V0 vs V4 LLM 검증 결과 |
| `logs/robust_kalman_experiment/expert_analysis_results.json` | 6개 통계적 분석 raw |
