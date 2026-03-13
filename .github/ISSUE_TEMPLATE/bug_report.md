```
---
name: Bug Report
about: Report a bug in Synaphe
title: "[Bug] "
labels: bug
---

**Describe the bug**
A clear description of what the bug is.

**To reproduce**
```
// Synaphe code that triggers the bug
let x = ...
```

**Expected behavior**
What you expected to happen.

**Actual behavior**
What actually happened (include error messages).

**Environment**
- Python version:
- OS:
- Synaphe version:
- Backend (PyTorch/PennyLane/Qiskit version if relevant):
```

6. Click **"Commit changes"** at the top right
7. In the popup, leave "Commit directly to main" selected and click **"Commit changes"**

Now repeat for the feature request:

8. Go back to the repo main page
9. Click **"+"** → **"Create new file"**
10. Type filename: `.github/ISSUE_TEMPLATE/feature_request.md`
11. Paste this:
```
---
name: Feature Request
about: Suggest a new feature for Synaphe
title: "[Feature] "
labels: enhancement
---

**What problem does this solve?**
Describe the pain point or use case.

**Proposed syntax**
```
// How would this look in Synaphe?
```

**Which pillar does this relate to?**
- [ ] Tensor shape safety
- [ ] Linear quantum types
- [ ] Autodiff bridge
- [ ] Hardware constraints
- [ ] Data pipelines
- [ ] Developer experience
- [ ] Other

**Alternatives considered**
Any other approaches you've thought about.
