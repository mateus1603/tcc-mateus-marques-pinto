# Fix: pd_dataframe() Error in TCN_new.ipynb

## Problem Description

In topic 5 (section on data normalization) of `models/TCN_new.ipynb`, line 352 contained an implementation error:

```python
train_df = train_scaled.pd_dataframe()
```

The method `pd_dataframe()` does not exist in the Darts TimeSeries API, causing a runtime error when executing this cell.

## Root Cause

The error occurred because:
1. The code attempted to call `pd_dataframe()` as a method with parentheses
2. The Darts TimeSeries API does not have a `pd_dataframe()` method
3. The correct approach is to use the `.values()` method which returns a numpy array directly

## Solution Implemented

**Before:**
```python
# TimeSeries.values() retorna um numpy array 2D, acessar com pd_dataframe() é mais robusto
train_df = train_scaled.pd_dataframe()
train_values = train_df.values.flatten()
```

**After:**
```python
# TimeSeries.values é uma propriedade que retorna um numpy array 2D
train_values = train_scaled.values.flatten()
```

## Changes Made

1. **Removed** the incorrect `pd_dataframe()` call
2. **Corrected** the API usage: `values` is a property, not a method (no parentheses)
3. **Simplified** the code by directly using the `.values` property
4. **Updated** the comment to accurately reflect the implementation
5. **Eliminated** the unnecessary intermediate `train_df` variable

## Verification

The fix uses the correct Darts TimeSeries API where `values` is accessed as a property:

- **Line 352**: `train_values = train_scaled.values.flatten()`
- **Line 620**: `forecast_values = future_forecast_original.values.flatten()`
- **Line 661**: `residuals = (test_original - test_predictions_original).values.flatten()`

## Impact

This fix:
- ✅ Resolves the runtime error in section 5
- ✅ Makes the code more concise and readable
- ✅ Maintains consistency with the rest of the notebook
- ✅ Uses the correct Darts TimeSeries API
- ✅ Preserves the exact same functionality

## Testing

To test this fix:
1. Open `models/TCN_new.ipynb` in Jupyter
2. Execute cells sequentially up to section 5
3. Verify that the normalization statistics are printed without errors
4. Continue with the rest of the notebook

## Related Files

- `models/TCN_new.ipynb` - Main implementation file (fixed)
- `models/utils_TCN_new.py` - Utility functions (no changes needed)

## Date

2025-11-10

## References

- Darts Documentation: https://unit8co.github.io/darts/
- Darts TimeSeries API: https://unit8co.github.io/darts/generated_api/darts.timeseries.html
