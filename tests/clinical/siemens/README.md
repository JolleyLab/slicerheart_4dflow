# Siemens

Planned source: Children's Hospital of Philadelphia, MAGNETOM Avanto Fit
1.5 T, compressed-sensing 4D flow (Ma LE, Markl M, Chow K, et al. Aortic
4D flow MRI in 2 minutes using compressed sensing, respiratory controlled
adaptive k-space reordering, and inline reconstruction. Magn Reson Med
2019;81(6):3675-3690. PMID 30803006. doi:10.1002/mrm.27684), retrospective ECG gating, cross-beam respiratory navigator,
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
  scanner (Bernstein MA, Zhou XJ, Polzin JA, et al. Concomitant gradient
  terms in phase contrast MR: analysis and correction. Magn Reson Med
  1998;39(2):300-308. PMID 9469714. doi:10.1002/mrm.1910390218)

Until a case exists this folder holds only this file; `test_clinical.py`
skips the Siemens case.
