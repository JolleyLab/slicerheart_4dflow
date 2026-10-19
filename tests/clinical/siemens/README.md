# Siemens

Planned source: Children's Hospital of Philadelphia, MAGNETOM Avanto Fit
1.5 T, compressed-sensing 4D flow (Ma et al., Magn Reson Med 2019, PMID
30803006), retrospective ECG gating, cross-beam respiratory navigator,
ferumoxytol or gadobutrol contrast, pediatric congenital heart disease.
Two cases are intended: one with a jet exceeding venc throughout the cycle
(coarctation or RVOT obstruction) and one without.

Vendor notes to confirm from the exported headers when the data arrive and
record in `manifest.json`:

- how the three velocity-encoded series are identified (series description,
  `ImageType`), and whether they are phase-difference images or phase
  images relative to a separate reference series
- the scale between stored pixel value and velocity (rescale slope and
  intercept, or venc-based scaling), per series
- the velocity-encoding directions relative to the image axes, and their
  sign convention, so that `encoding_to_world` in the Import module is
  checked against a known-direction flow (ascending aorta is cranial)
- whether concomitant-gradient (Maxwell) correction was applied on the
  scanner (Bernstein et al., Magn Reson Med 1998, PMID 9469714)

Until a case exists this folder holds only this file; `test_clinical.py`
skips the Siemens case.
