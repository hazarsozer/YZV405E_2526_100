# Report

## Compiling

### Option A — Overleaf (recommended)

1. Go to [overleaf.com](https://overleaf.com) → New Project → Blank Project.
2. Upload `report.tex` and `references.bib`.
3. In Overleaf, go to Menu → Compiler → set to **pdfLaTeX**.
4. Overleaf includes the ACL 2023 style files — no manual download needed.
   If the template is missing, add a new file from the template gallery:
   search "ACL" and import the ACL 2023 template, then replace the body with our content.
5. Click Compile. The PDF will be generated in the right pane.

### Option B — Local (requires TeXLive)

```bash
# Install TeX (Ubuntu/PopOS)
sudo apt install texlive-full

# Download ACL 2023 style files
curl -L https://github.com/acl-org/acl-style-files/archive/master.zip -o acl-style.zip
unzip acl-style.zip
cp acl-style-files-master/acl.sty .
cp acl-style-files-master/acl_natbib.bst .

# Compile
cd report/
pdflatex report
bibtex report
pdflatex report
pdflatex report   # second pass to resolve cross-references
```

## Notes on References

Some bibliography entries in `references.bib` are best-guess approximations
based on known publication dates. Verify these before final submission:

- `admire2026` — fill in the actual proceedings details once published
- `semeval2025admire1` — update with the final SemEval-2025 task paper citation
- `siglip2_2025` — verify authors and title of the SigLIP2 technical report

All other entries (BGE-M3, NLLB-200, Phi-3, CLIP, BLIP-2, Yuksekgonul 2022,
SemEval-2022 Task 2) are stable and can be used as-is.
