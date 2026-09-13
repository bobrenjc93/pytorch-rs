import json, os, pathlib, subprocess, sys, time
root=pathlib.Path.cwd()
base=root/'target/tensor-madd'
script='docs/diagnostics/compile-pointwise-tensor-madd/capture.py'
python=str(root/'.venv/bin/python')
receipts=[]
def run(label,args,devices='0'):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=devices)
    def inventory():
        return subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,driver_version,utilization.gpu,memory.used','--format=csv'],text=True)
    before=inventory()
    started=time.time()
    with (base/'logs'/f'{label}.log').open('w') as log:
        p=subprocess.run(args,env=env,stdout=log,stderr=subprocess.STDOUT)
    receipt={'label':label,'command':args,'cudaVisibleDevices':devices,'started':started,'finished':time.time(),'returncode':p.returncode,'gpuBefore':before,'gpuAfter':inventory()}
    receipts.append(receipt)
    (base/'final-commands.json').write_text(json.dumps(receipts,indent=2)+'\n')
    print(label,p.returncode,flush=True)
    return p.returncode==0
if not run('verified-build',[python,script,'--build',str(base/'verified-build')]):
    raise SystemExit(1)
manifest=str(base/'verified-build/build.json')
run('verified-provenance',[python,'.github/scripts/verify_native_extension.py'])
run('tensor-madd-final-two-devices',[python,'-m','unittest','discover','-s','tests','-p','test_compile_pointwise_tensor_madd.py','-v'],'0,1')
checks=['test_compile_pointwise_jit.Hardware.test_two_device_restoration_and_module_ownership',
        'test_compile_pointwise_broadcast.Broadcast.test_device_restoration_on_compile_run_failure_and_module_release',
        'test_compile_pointwise_runtime_scalars.RuntimeScalarHardware.test_runtime_parameters_restore_current_device',
        'test_compile_pointwise_runtime_scalars.PositionalScalarHardware.test_positional_runtime_parameters_restore_device_on_success_and_failure']
run('existing-two-device-guards',[python,'-c','import sys, unittest; sys.path.insert(0,"tests"); unittest.main(module=None,argv=["unittest",*sys.argv[1:]],verbosity=2)',*checks],'0,1')
run('portable-metadata',[python,'-m','unittest','discover','-s','tests','-p','test_compile_pointwise_tensor_madd.py','-v'],'')
for i,implementation in enumerate(('native','reference','reference','native')):
    run(f'paired-{i}-{implementation}',[python,script,implementation,str(base/f'paired-{i}-{implementation}'),manifest])
for a,b in (('0-native','1-reference'),('2-reference','3-native')):
    run(f'compare-{a}-{b}',[python,script,'--compare',str(base/f'paired-{a}/report.json.gz'),str(base/f'paired-{b}/report.json.gz'),str(base/f'comparison-{a}-{b}.json')])
run('default-clippy',['cargo','clippy','--locked','--offline','--all-targets','--','-D','warnings'])
run('default-rust',['cargo','test','--locked','--offline','--lib','--','--test-threads=1'])
run('final-format',['cargo','fmt','--all','--','--check'])
raise SystemExit(0 if all(r['returncode']==0 for r in receipts) else 1)
