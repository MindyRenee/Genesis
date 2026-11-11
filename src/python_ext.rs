//! `genesis_body` — the PyO3 bridge.
//!
//! The mind holds the body's organs in-process through this module:
//! `genesis_body.VisionSystem` is the same `self_::vision::VisionSystem`
//! the daemon would step — same code, same dynamics — so the Python
//! side drives the eye with real frames and reads real observables
//! without a wire round-trip. The module only exists when the crate
//! is built `--features python`; the daemon path is unaffected.

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::self_::vision::{RadianceField, VisionSystem};

/// The assembled eye: optics → retina → V1 → gaze, stepped by Python.
#[pyclass(name = "VisionSystem")]
pub struct PyVision {
    inner: VisionSystem,
    width: usize,
    height: usize,
}

#[pymethods]
impl PyVision {
    /// A retinotopic field of `width × height` loci.
    #[new]
    fn new(width: usize, height: usize) -> Self {
        Self {
            inner: VisionSystem::new(width, height),
            width,
            height,
        }
    }

    /// One afferent step over a packed RGB24 frame (`width*height*3`
    /// bytes). Returns the tick's observables as a dict: ocular state,
    /// ganglion/cortical planes, mean luminance, fixation point.
    fn step<'py>(&mut self, py: Python<'py>, rgb24: &[u8], dt: f64) -> PyResult<Py<PyDict>> {
        if rgb24.len() != self.width * self.height * 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "expected {} bytes of RGB24, got {}",
                self.width * self.height * 3,
                rgb24.len()
            )));
        }
        let field = RadianceField::from_rgb24(self.width, self.height, rgb24);
        Ok(output_dict(py, self.inner.step(&field, dt))?)
    }

    /// Step with no incident light — the dark-adapted eye. This is
    /// the "no camera" case: the organ still runs, it just sees
    /// nothing (a real biological state, not a synthetic frame).
    fn step_dark<'py>(&mut self, py: Python<'py>, dt: f64) -> PyResult<Py<PyDict>> {
        let field = RadianceField::dark(self.width, self.height);
        Ok(output_dict(py, self.inner.step(&field, dt))?)
    }

    /// Voluntary lid set-point ∈ [0,1] — sleep drops it toward ~0.15.
    #[setter]
    fn set_lid_set_point(&mut self, v: f64) {
        self.inner.lid.set_point = v.clamp(0.0, 1.0);
    }

    /// Sympathetic (norepinephrine) drive ∈ [0,1] for pupillary dilation.
    #[setter]
    fn set_ne_drive(&mut self, v: f64) {
        self.inner.ne_drive = v.clamp(0.0, 1.0);
    }

    /// Ciliary accommodation — defocus σ in pixels; 0 is sharp.
    #[setter]
    fn set_defocus_sigma(&mut self, v: f64) {
        self.inner.defocus_sigma = v.max(0.0);
    }

    #[getter]
    fn gaze(&self) -> (f64, f64) {
        (self.inner.gaze.x, self.inner.gaze.y)
    }
}

fn output_dict(py: Python<'_>, out: crate::self_::vision::VisionOutput) -> PyResult<Py<PyDict>> {
    let d = PyDict::new(py);
    d.set_item("mean_luminance", out.mean_luminance)?;
    d.set_item("lid", out.ocular.lid)?;
    d.set_item("pupil_mm", out.ocular.pupil_mm)?;
    d.set_item("irradiance_gain", out.ocular.irradiance_gain)?;
    d.set_item("defocus_sigma", out.ocular.defocus_sigma)?;
    d.set_item("gaze", out.gaze)?;
    d.set_item("dog", out.dog)?;
    d.set_item("ganglion_on", out.ganglion_on)?;
    d.set_item("ganglion_off", out.ganglion_off)?;
    d.set_item("cortical", out.cortical)?;
    Ok(d.unbind())
}

#[pymodule]
fn genesis_body(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyVision>()?;
    Ok(())
}
