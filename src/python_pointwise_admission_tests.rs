use super::*;
use pyo3::exceptions::PyMemoryError;

#[test]
fn admission_shared_reentrancy_and_errors_release_every_borrow() {
    Python::initialize();
    Python::attach(|py| {
        let a = Py::new(py, PyTensor::new(CoreTensor::ones([2]).unwrap())).unwrap();
        let b = Py::new(py, PyTensor::new(CoreTensor::ones([2]).unwrap())).unwrap();
        let inputs = PyTuple::new(py, [a.bind(py), b.bind(py)]).unwrap();
        let held = b.borrow_mut(py);
        assert!(admit_inputs(&inputs).is_err());
        assert!(a.try_borrow_mut(py).is_ok());
        drop(held);
        // Model a callback during allocation while the actual shared owner is
        // active. Nested admission may borrow, but mutation must be rejected.
        let error = with_inputs(&inputs, |_| -> PyResult<()> {
            assert!(a.try_borrow_mut(py).is_err());
            assert!(b.try_borrow_mut(py).is_err());
            assert!(
                admit_inputs(&inputs)
                    .unwrap_err()
                    .is_instance_of::<PyNotImplementedError>(py)
            );
            Err(PyMemoryError::new_err("controlled allocation failure"))
        })
        .unwrap_err();
        assert!(error.is_instance_of::<PyMemoryError>(py));
        assert!(a.try_borrow_mut(py).is_ok());
        assert!(b.try_borrow_mut(py).is_ok());
        assert!(admit_inputs(&inputs).is_err());
        assert!(a.try_borrow_mut(py).is_ok());
    });
}

#[test]
fn admission_native_layout_and_cpu_precedence() {
    if crate::cuda::device_count() == 0 {
        eprintln!("skipping native admission layout controls: CUDA unavailable");
        return;
    }
    Python::initialize();
    Python::attach(|py| {
        let cpu = Py::new(py, PyTensor::new(CoreTensor::ones([2]).unwrap())).unwrap();
        let cuda = Py::new(
            py,
            PyTensor::new(
                CoreTensor::ones([2, 2])
                    .unwrap()
                    .try_copy_cpu_to_cuda(crate::Device::Cuda(0))
                    .unwrap(),
            ),
        )
        .unwrap();
        let repeated = PyTuple::new(py, [cuda.bind(py), cuda.bind(py)]).unwrap();
        with_inputs(&repeated, |_| {
            assert_eq!(admit_inputs(&repeated)?.len(), 2);
            assert!(cuda.try_borrow_mut(py).is_err());
            Ok(())
        })
        .unwrap();
        assert_eq!(validate_inputs(&repeated).unwrap(), 0);
        let transposed = cuda.borrow(py).inner.transpose(0, 1).unwrap();
        cuda.borrow_mut(py).inner = transposed;
        assert!(
            admit_inputs(&repeated)
                .unwrap_err()
                .to_string()
                .contains("contiguous")
        );
        for inputs in [
            PyTuple::new(py, [cuda.bind(py), cpu.bind(py)]).unwrap(),
            PyTuple::new(py, [cpu.bind(py), cuda.bind(py)]).unwrap(),
        ] {
            assert!(
                admit_inputs(&inputs)
                    .unwrap_err()
                    .to_string()
                    .contains("default backend does not compile CPU tensors")
            );
            assert!(cuda.try_borrow_mut(py).is_ok());
        }
        let restored = cuda.borrow(py).inner.transpose(0, 1).unwrap();
        cuda.borrow_mut(py).inner = restored;
        assert_eq!(admit_inputs(&repeated).unwrap().len(), 2);
    });
}
