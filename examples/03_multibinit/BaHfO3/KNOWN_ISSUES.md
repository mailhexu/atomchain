# Known Issues with MULTIBINIT Integration

## Calculator State Loss During Phonon Calculations

**Status**: Under investigation  
**Severity**: High  
**Affects**: Phonon calculations with finite displacement method

### Problem Description

When running phonon calculations that require multiple force evaluations on displaced structures, the MULTIBINIT calculator loses its initialized state after the first calculation. This manifests as:

```
RuntimeError: Potential not initialized. Use from_abi() or from_params().
```

### Root Cause

The issue appears to be in pymultibinit's `MultibinitCalculator.__del__()` method, which calls `self.potential.free()`. When ASE or atomchain internal code copies the calculator (e.g., during phonon supercell generation), Python's garbage collector may call `__del__` on the copy, which frees the underlying C library resources that are still needed by the original calculator.

### Temporary Workaround

Until this is fixed in pymultibinit, you can comment out the `__del__` method:

```python
# In pymultibinit/src/pymultibinit/calculator.py, line 183-186:
# def __del__(self):
#     """Destructor - ensure cleanup."""
#     if hasattr(self, 'potential'):
#         self.potential.free()
```

This prevents premature cleanup of the C library resources. The resources will still be freed when Python exits.

### Additional Observations

- First force evaluation succeeds
- Forces computed are unreasonably large (1e6 to 1e9 eV/Å), suggesting numerical issues
- This may indicate the structure being evaluated doesn't match the potential's expectations

### Status

This issue has been reported to the pymultibinit development team. Track progress at:
- pymultibinit GitHub: https://github.com/abinit/pymultibinit

### Workaround for Users

For now, MULTIBINIT integration works best for:
1. Single-point energy/force calculations
2. Structure relaxations (these work because they don't copy the calculator)
3. Molecular dynamics with careful calculator management

Phonon calculations require the fix mentioned above.
