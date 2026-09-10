import os, unittest
assert os.environ['CUDA_VISIBLE_DEVICES'] == '0,1'
names = [
 'tests.test_cuda_sum_rows.CudaSumRowsDeviceTests',
 'tests.test_cuda_add.CudaAddDeviceTests',
 'tests.test_cuda_add_trailing_vector.CudaAddTrailingVectorDeviceTests',
 'tests.test_cuda_neg.CudaNegDeviceTests',
 'tests.test_cuda_mul_scalar.CudaMulScalarDeviceTests',
 'tests.test_cuda_host_transfer.CudaHostTransferDeviceGuardTests',
 'tests.test_cuda_same_device_copy.CopyDeviceGuardTests',
 'tests.test_cuda_zero_roundtrip.CudaCurrentDeviceGuardTests',
 'tests.test_compile_cuda_boundary.CompileCudaDeviceTests',
 'tests.test_compile_cuda_mul_scalar.CompileCudaMulScalarDeviceTests',
 'tests.test_compile_cuda_neg.CompileCudaNegDeviceTests',
 'tests.test_compile_cuda_trailing_vector.CompileCudaTrailingVectorDeviceTests',
]
# The caller puts the appropriate checkout on sys.path, preserving each .venv.
suite = unittest.defaultTestLoader.loadTestsFromNames(names)
result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(0 if result.wasSuccessful() and not result.skipped else 1)
