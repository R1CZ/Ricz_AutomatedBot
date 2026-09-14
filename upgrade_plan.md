# RCZ v5 FULLY ADAPTIVE - CODEBASE AUDIT & UPGRADE PLAN

## PHASE 1: ARCHITECTURE DEPENDENCY MAP

### Classes Identified:
1. AdaptiveSpreadModel (line 328) - Spread tracking and adaptive thresholds
2. MicrostructureEngine (line 386) - Tick-level microstructure analysis
3. NewsAdaptiveFilter (line 445) - Market abnormal state detection
4. LearningMemory (line 499) - Historical trade records + logistic model
5. AdaptiveWeightEngine (line 853) - Component weight calculation
6. AdaptiveThresholdEngine (line 896) - Entry threshold adaptation
7. AdaptiveProbabilityModel (line 942) - Probability calibration
8. MarketStateEngine (line 983) - Market regime computation
9. AdaptiveComponentEngine (line 1126) - All signal components
10. AdaptiveSignalAggregator (line 1787) - Signal aggregation logic
11. AdaptiveExitModel (line 1860) - SL/TP and management params
12. AdaptiveSignalEngine (line 1926) - Full signal analysis pipeline
13. TradingBot (line 2156) - Execution, management, invalidation

### Data Flow Analysis:

```
TICK DATA → spread_model.update() + micro.update()
     ↓
MARKET DATA (M1/M5/M15) → get_market_data()
     ↓
MarketStateEngine.compute() → state dict
     ↓
NewsAdaptiveFilter.update() → news_state
     ↓
AdaptiveComponentEngine.compute_all() → components dict
     ↓
AdaptiveWeightEngine.get_weights() → weights, reliabilities, reliability_avg
     ↓
AdaptiveSignalAggregator.aggregate() → agg dict (buy_score, sell_score, agreement, conflict)
     ↓
AdaptiveThresholdEngine.get() → threshold
     ↓
AdaptiveProbabilityModel.calibrate() → calibrated probability
     ↓
_entry_quality() → entry_quality score
     ↓
Final entry_score = 0.58*calibrated + 0.42*entry_quality
     ↓
Decision: entry_score >= threshold AND calibrated >= threshold-0.07 AND entry_quality >= threshold-0.18
```

### Critical Issues Found:

#### ISSUE 1: reliability_avg NOT persisted to signal/learning record correctly
- Line 2024: `reliability_avg` is calculated
- Line 2054: Used in features vector as placeholder
- Line 3121: Saved as `signal.get("report", {}).get("reliability_avg", 0.5)` - WRONG PATH!
  - Should be from top-level signal dict, not report

#### ISSUE 2: news_state NOT persisted correctly
- Line 2148: news_state is NOT in the returned signal dict
- Line 3136: Tries to save `context.get("news_state", "NORMAL")` - but context never has it!
  - news_state is computed at line 3451 but not passed into signal

#### ISSUE 3: Dashboard startup before main loop
- Lines 3602-3606: Dashboard thread starts BEFORE main loop
- This is actually CORRECT - dashboard runs independently

#### ISSUE 4: Duplicate/competing logic
- manage_positions() calculates mgmt params per position
- _run_profit_protection() uses those params
- But _profit_protect_thresholds() recalculates similar values

#### ISSUE 5: Winner-only learning bias
- Line 748: get_base_threshold() filters ONLY winning trades
- Line 765: get_score_scale() filters ONLY winning trades
- This creates upward bias in thresholds

#### ISSUE 6: Component reliability feedback loop
- Line 718-728: get_component_reliability() uses win_rate from past trades
- But those trades were selected using weights based on previous reliability
- Creates circular dependency

#### ISSUE 7: Invalidation uses fixed MIN_HOLD_SECONDS
- Line 3325: `hold_sec < MIN_HOLD_SECONDS and adverse_atr < 1.0`
- MIN_HOLD_SECONDS = 12 (line 41) - static value
- Should be adaptive based on market state

#### ISSUE 8: Flip logic combines invalidation + reversal
- Lines 3018-3023: can_flip() checks if opposite signal is stronger
- Then immediately flips without separate invalidation check

#### ISSUE 9: Entry quality uses stale data
- Line 1954: `df = None` explicitly avoids fresh data
- Uses component scores from analyze() call which may be 0.5s old

#### ISSUE 10: Reliability avg fallback to 0.5
- Line 889: If COMPONENTS empty, returns 0.5
- Line 2054: Features use 0.5 default
- Should propagate actual calculated value

### Variables Calculated But Not Used Downstream:
- breakout_prob (state) - calculated but never used in decisions
- reversal_prob (state) - calculated but never used
- momentum_state (state) - calculated but limited usage
- market_speed - used minimally

### State Synchronization Issues:
- signal dict has its own context copy
- meta has separate context copy  
- learning record has another copy
- No single authoritative state object

---

## PHASE 2-26: IMPLEMENTATION PRIORITY LIST

### HIGH PRIORITY FIXES:

1. **Fix reliability_avg persistence** (Issue #1)
   - Move reliability_avg to top-level signal dict
   - Use correct key in meta save

2. **Fix news_state persistence** (Issue #2)
   - Add news_state to signal return dict
   - Pass through to meta and learning

3. **Fix winner-only learning bias** (Issue #5)
   - Change get_base_threshold() to use ALL trades with positive expectancy
   - Optimize for EV boundary, not winner scores

4. **Create single signal state object**
   - Consolidate all signal-related data into one authoritative dict
   - Use same object for execution, metadata, learning, dashboard

5. **Build true predictive entry layer**
   - Enhance feature vector with structure/momentum/pressure features
   - Improve logistic model with interaction terms
   - Add expected favorable/adverse excursion estimation

6. **Independent component reliability**
   - Measure each component's contribution vs future movement
   - Separate from weight feedback loop

7. **Predictive thesis invalidation**
   - Build failure probability model
   - Use adverse excursion + opposite signals + time decay

8. **Adaptive grace period**
   - Replace MIN_HOLD_SECONDS with state-aware grace
   - Strong momentum = shorter grace for opposite signals
   - Normal conditions = standard grace

9. **Separate invalidation from flipping**
   - First evaluate: is current trade invalid?
   - Then independently: is new entry valid?

10. **Execution retry revalidation**
    - Before each retry after requote/price_change
    - Lightweight validation of signal freshness

### MEDIUM PRIORITY:

11. Structure intelligence upgrade (recency weighting)
12. Momentum continuation vs exhaustion detection
13. Spread model improvement (spread/ATR ratio)
14. Pressure/order-flow enhancement
15. Adaptive cooldown (context-dependent)
16. Better risk-state awareness
17. Learning-data separation (entry vs exit quality)

### LOW PRIORITY:

18. Code cleanup
19. Logging improvements
20. Dashboard polish
21. Performance optimization

---

## IMPLEMENTATION STRATEGY:

1. Make minimal surgical changes first (fixes)
2. Build predictive layer on top of existing components
3. Preserve all working strategy logic
4. Maintain trading frequency
5. Test each change incrementally
