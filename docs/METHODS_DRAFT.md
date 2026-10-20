# Example Methods text for the new pre-processing, filtering, QA and visualization features

Drafted to slot into the JCMR framework manuscript (Lasso, Iacovella et al.) in
place of, or alongside, the current "Pre-processing", "Visualization" and
"Quantification" subsections. Plain prose, manuscript voice, references in the
numbered style used there. Identifiers: DOI is the verified identifier for every
entry; PMIDs are given where known and should be checked on import.

---

## Pre-processing

4D flow data were imported into SlicerHeart with the Import 4D Flow module and
assembled into a time-resolved series in which each voxel holds a three-component
velocity vector at every cardiac phase, as described above. Pre-processing was
then applied in a fixed order: background phase correction, velocity anti-aliasing,
noise masking, and, optionally, divergence-free filtering. Background phase
correction is applied first because any masking or unwrapping step that precedes
it alters the static-tissue velocities on which the correction depends [1].

### Background phase (eddy current) offset correction

Residual background phase offsets arising from eddy currents and concomitant
gradient terms were corrected by fitting a low-order polynomial to the time-averaged
velocity of static tissue and subtracting the fitted surface from every cardiac
phase [2, 3]. Candidate static-tissue voxels were selected automatically from the
time-averaged magnitude image, the temporal standard deviation of velocity, a
velocity-weighted angiogram used to exclude vessels, and the dilated lumen
segmentation; the body contour was eroded to exclude the rim most affected by
spatial wrap-around. Four fitting strategies are implemented and selectable: ordinary
least squares [2]; M-estimate sample consensus (MSAC), in which many minimal-subset
fits are scored by a truncated residual cost and the best model is refitted on its
inliers, making the fit insensitive to vessels or lung that leak into the candidate
set [4]; weighted regularized least squares with automatic rejection of temporally
invariant outliers, such as steady venous flow and spatial wrap-around, by a
Gaussian mixture model of the residuals, after Pruitt et al. [5] (implemented here
with inverse-temporal-variance weights, a ridge penalty and a central field-of-view
first fit); and a combined approach, used by
default, in which MSAC removes gross outliers before wrap-around rejection and a
final MSAC refit. The residual velocity in static tissue after correction is reported
as a quality metric against the 0.6 cm/s threshold proposed for regurgitant and
shunt quantification [6].

### Velocity anti-aliasing

Velocities exceeding the velocity-encoding (venc) value produce phase wraps.
Wraps were corrected in two passes. First, a single-step Laplacian method [7]
estimated the Laplacian of the true phase from the wrapped phase [8] and solved the
resulting Poisson equation in one step using discrete cosine transforms with
Neumann boundaries, applied to each cardiac phase; this resolves aliasing that
persists throughout the cardiac cycle, as in stenotic jets and surgical baffles.
Second, the temporal method [9] identified any remaining single-phase jumps along
each voxel's time course. Voxels outside the signal region (see noise masking) were
excluded from the spatial solve. The resulting wrap-count image is stored as a label
map and can be reviewed and edited with the Segment Editor before the corrected
velocities are regenerated.

### Noise masking and phase-contrast angiography

Voxels with low time-averaged magnitude or high temporal velocity standard
deviation were identified as noise and excluded from visualization and seeding [10].
A velocity-weighted phase-contrast MR angiogram [11] was computed as the root of the
time-averaged product of squared magnitude and squared speed and stored as a
scalar volume for thresholding or automated segmentation.

### Divergence-free filtering

Because blood is incompressible, the measured velocity field should be
divergence-free within the lumen, and the divergent component of the measurement
is attributable to noise. Four filters are implemented. The finite-difference
projection method of Song et al. [12], as evaluated by Loecher et al. [13], applies
the minimum-norm correction such that the discrete divergence, computed with a
masked operator using central differences within the lumen and one-sided differences
at the wall, vanishes at every lumen voxel; the constraint is enforced exactly by
conjugate-gradient solution of the associated normal equations. A relaxed variant
[14] penalises rather than constrains divergence. A divergence-free wavelet filter
in the spirit of Ong et al. [15] and Bostan et al. [16] separates the field into
divergence-free and curl-free components with the Leray projector, applies an
undecimated isotropic wavelet transform [17] to each with normalized convolution so
that the lumen wall does not contribute [18], soft-thresholds the divergence-free
detail coefficients lightly and the curl-free coefficients more strongly, and
recombines the two; this softly suppresses divergence while preserving the large,
structured divergence that discretisation produces at vessel edges and, unlike
projection, also attenuates the solenoidal component of the noise. Projection
removes only the curl-free component of isotropic noise, equal to one third of its
power, which sets a floor on the achievable noise reduction. Filtering is optional
and is recommended for visualization and for derived quantities such as vorticity
and wall shear stress; flow quantification is performed on unfiltered or
projection-filtered data to avoid amplitude bias.

## Visualization

In addition to the velocity glyph, streamline and Doppler-like renderings described
above, time-resolved pathlines (particle traces) [19] were generated by emitting
particles from a measurement plane and integrating their trajectories through the
time-varying velocity field with a fourth-order Runge-Kutta scheme and trilinear
spatial and linear periodic temporal interpolation, terminating particles that leave
the lumen segmentation. Pathlines are computed with the VTK particle-path filter
[20], the integrator used by ParaView, with the velocity frames linearly interpolated
in time by a factor of four before integration so that the integration step is a
fraction of a voxel at peak velocity, and are stored as a model node with per-point
time and particle index so they can be animated with the Sequences module and
coloured by time or speed.

## Quantification

Measurement planes were positioned either interactively, as described above, or
automatically at fixed arc-length intervals perpendicular to a vessel centerline
extracted with the Vascular Modeling Toolkit [21], yielding flow as a function of
position along the vessel. For every study, internal consistency was assessed by
comparing inflow and outflow at user-defined plane sets (ascending aorta against
arch branches and descending aorta; main pulmonary artery against ascending aorta
for Qp:Qs; superior and inferior caval veins against branch pulmonary arteries in
cavopulmonary circulations), reported as percent conservation-of-mass error at each
pre-processing stage. This supports the quality assurance addressed by the 2023
consensus statement [22]; conservation of mass was used as an internal control in a
comparison of commercial software [23], and agreement of peak velocity between
packages has been reported to be poor [24].

## Validation of the pre-processing steps

Each step was validated on synthetic data with known ground truth. Background
phase correction was evaluated on a chest phantom containing lungs, a pulsatile
artery, a slow steady vein and a phase-encode wrap-around band with a second-order
offset field of up to 3.6 cm/s; the combined MSAC and wrap-around rejection method
recovered the offset with a maximum error below 0.1 cm/s in all scenarios, whereas
ordinary least squares erred by up to 1.0 cm/s when wrap and steady flow were both
present. Anti-aliasing, divergence-free filtering, pathlines and plane placement were
evaluated on a publicly available aortic computational fluid
dynamics field [25] resampled to 1.19 mm, with Gaussian noise at 5, 10 and 20% of
venc and with aliasing induced by setting venc below the peak velocity. At 10% noise,
finite-difference projection reduced the root-mean-square velocity error from 15.0
to 12.3 cm/s and the wavelet filter to 3.1 cm/s, with mean angular error reduced from
25 to 4 degrees and through-plane flow preserved within 2 mL/s. With a jet exceeding
venc throughout the cycle, Laplacian followed by temporal unwrapping left one fifth
as many residual wrapped voxels as temporal unwrapping alone.

---

## References (new entries only)

1. Keller EJ, Collins JD, Rigsby CK, Carr JC, Markl M, Schnell S. Optimized 4D flow MRI processing for evaluation of abdominal blood flow. Proc Intl Soc Mag Reson Med 2016;24:1939.
2. Walker PG, Cranney GB, Scheidegger MB, Waseleski G, Pohost GM, Yoganathan AP. Semiautomated method for noise reduction and background phase error correction in MR phase velocity data. J Magn Reson Imaging 1993;3(3):521-530. PMID 8324312. doi:10.1002/jmri.1880030315
3. Lankhaar JW, Hofman MB, Marcus JT, Zwanenburg JJ, Faes TJ, Vonk-Noordegraaf A. Correction of phase offset errors in main pulmonary artery flow quantification. J Magn Reson Imaging 2005;22(1):73-79. PMID 15971181. doi:10.1002/jmri.20361
4. Fischer C, Wetzl J, Schaeffter T, Giese D. Fully automated background phase correction using M-estimate SAmple consensus (MSAC): application to 2D and 4D flow. Magn Reson Med 2022;88(6):2709-2717. doi:10.1002/mrm.29363
5. Pruitt AA, Jin N, Liu Y, Simonetti OP, Ahmad R. A method to correct background phase offset for phase-contrast MRI in the presence of steady flow and spatial wrap-around artifact. Magn Reson Med 2019;81(4):2424-2438. PMID 30431176. doi:10.1002/mrm.27572
6. Gatehouse PD, Rolf MP, Graves MJ, et al. Flow measurement by cardiovascular magnetic resonance: a multi-centre multi-vendor study of background phase offset errors that can compromise the accuracy of derived regurgitant or shunt flow measurements. J Cardiovasc Magn Reson 2010;12:5. PMID 20074359. doi:10.1186/1532-429X-12-5
7. Loecher M, Schrauben E, Johnson KM, Wieben O. Phase unwrapping in 4D MR flow with a 4D single-step Laplacian algorithm. J Magn Reson Imaging 2016;43(4):833-842. PMID 26417641. doi:10.1002/jmri.25045
8. Schofield MA, Zhu Y. Fast phase unwrapping algorithm for interferometric applications. Opt Lett 2003;28(14):1194-1196. PMID 12885018. doi:10.1364/OL.28.001194
9. Xiang QS. Temporal phase unwrapping for cine velocity imaging. J Magn Reson Imaging 1995;5(5):529-534. PMID 8574036. doi:10.1002/jmri.1880050509
10. Bock J, Kreher BW, Hennig J, Markl M. Optimized pre-processing of time-resolved 2D and 3D phase contrast MRI data. Proc Intl Soc Mag Reson Med 2007;15:3138.
11. Bock J, Frydrychowicz A, Stalder AF, et al. 4D phase contrast MRI at 3 T: effect of standard and blood-pool contrast agents on SNR, PC-MRA, and blood flow visualization. Magn Reson Med 2010;63(2):330-338. PMID 20024953. doi:10.1002/mrm.22199
12. Song SM, Napel S, Glover GH, Pelc NJ. Noise reduction in three-dimensional phase-contrast MR velocity measurements. J Magn Reson Imaging 1993;3(4):587-596. PMID 8347951. doi:10.1002/jmri.1880030407
13. Loecher M, Kecskemeti S, Turski P, Wieben O. Comparison of divergence-free algorithms for 3D MRI with three-directional velocity encoding. J Cardiovasc Magn Reson 2012;14(Suppl 1):W64. doi:10.1186/1532-429X-14-S1-W64
14. Mura J, Pino AM, Sotelo J, et al. Enhancing the velocity data from 4D flow MR images by reducing its divergence. IEEE Trans Med Imaging 2016;35(10):2353-2364. doi:10.1109/TMI.2016.2570010
15. Ong F, Uecker M, Tariq U, et al. Robust 4D flow denoising using divergence-free wavelet transform. Magn Reson Med 2015;73(2):828-842. PMID 24549830. doi:10.1002/mrm.25176
16. Bostan E, Unser M, Ward JP. Divergence-free wavelet frames. IEEE Signal Process Lett 2015;22(8):1142-1146. doi:10.1109/LSP.2015.2388794
17. Starck JL, Fadili J, Murtagh F. The undecimated wavelet decomposition and its reconstruction. IEEE Trans Image Process 2007;16(2):297-309. PMID 17269625. doi:10.1109/TIP.2006.887733
18. Knutsson H, Westin CF. Normalized and differential convolution. Proc IEEE CVPR 1993:515-523. doi:10.1109/CVPR.1993.341081
19. Wigstrom L, Ebbers T, Fyrenius A, et al. Particle trace visualization of intracardiac flow using time-resolved 3D phase contrast MRI. Magn Reson Med 1999;41(4):793-799. PMID 10332856. doi:10.1002/(SICI)1522-2594(199904)41:4<793::AID-MRM19>3.0.CO;2-2
20. Schroeder W, Martin K, Lorensen B. The Visualization Toolkit: An Object-Oriented Approach to 3D Graphics. 4th ed. Kitware; 2006. ISBN 978-1930934191.
21. Antiga L, Piccinelli M, Botti L, Ene-Iordache B, Remuzzi A, Steinman DA. An image-based modeling framework for patient-specific computational hemodynamics. Med Biol Eng Comput 2008;46(11):1097-1112. PMID 19002516. doi:10.1007/s11517-008-0420-1
22. Bissell MM, Raimondi F, Ait Ali L, et al. 4D Flow cardiovascular magnetic resonance consensus statement: 2023 update. J Cardiovasc Magn Reson 2023;25(1):40. PMID 37474977. doi:10.1186/s12968-023-00942-z
23. Burkhardt BEU, Kellenberger CJ, Callaghan FM, Valsangiacomo Buechel ER, Geiger J. Flow evaluation software for four-dimensional flow MRI: a reliability and validation study. Radiol Med 2023;128(10):1225-1235. PMID 37620674. doi:10.1007/s11547-023-01697-4
24. Oechtering TH, Nowak A, Sieren MM, et al. Repeatability and reproducibility of various 4D Flow MRI postprocessing software programs in a multi-software and multi-vendor cross-over comparison study. J Cardiovasc Magn Reson 2023;25(1):22. PMID 36978131. doi:10.1186/s12968-023-00921-4
25. Ferdian E, Suinesiaputra A, Dubowitz D, et al. 4DFlowNet: super-resolution 4D flow MRI using deep learning and computational fluid dynamics. Front Phys 2020;8:138. doi:10.3389/fphy.2020.00138

Related but not cited in the text: Sereno M, Koehler B, Preim B. Comparison of divergence-free filters for cardiac 4D PC-MRI data. In: Bildverarbeitung fuer die Medizin 2018. Informatik aktuell. Springer; 2018:139-144. doi:10.1007/978-3-662-56537-7_44. Pirrello A. Comparing the performances of different divergence-free approaches to increase the accuracy of 4D flow MRI velocity data. MSc thesis, Politecnico di Torino, 2024. You S, Masutani E, Alley MT, et al. Deep learning automated background phase error correction for abdominopelvic 4D flow MRI. Radiology 2022;302(3):584-592. PMID 34846200. doi:10.1148/radiol.2021211270.
