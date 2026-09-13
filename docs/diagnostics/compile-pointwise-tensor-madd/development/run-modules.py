"""Run the existing compiler selection in separate processes, retaining each log."""
import hashlib,json,os,pathlib,re,subprocess,sys,time
root=pathlib.Path.cwd()
base=root/'target/tensor-madd'
logs=base/'module-logs'
logs.mkdir()
files=sorted((root/'tests').glob('test_compile*.py'))
record={'selection':'tests/test_compile*.py','files':{},'runs':[],'passed':False}
for path in files:
    record['files'][str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
for path in files:
    command=[sys.executable,'-m','unittest','discover','-s','tests','-p',path.name,'-v']
    start=time.time()
    with (logs/(path.stem+'.log')).open('w') as log:
        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    text=(logs/(path.stem+'.log')).read_text()
    count=re.findall(r'Ran (\d+) tests?',text)
    row={'file':str(path.relative_to(root)),'command':command,'start':start,'end':time.time(),
         'returncode':result.returncode,'tests':int(count[-1]) if count else None}
    record['runs'].append(row)
    (base/'module-results.json').write_text(json.dumps(record,indent=2)+'\n')
    print(path.name,result.returncode,row['tests'],flush=True)
record['sourceUnchanged']=all(hashlib.sha256((root/p).read_bytes()).hexdigest()==h for p,h in record['files'].items())
record['passed']=record['sourceUnchanged'] and len(record['runs'])==len(files) and all(r['returncode']==0 and r['tests'] is not None for r in record['runs'])
(base/'module-results.json').write_text(json.dumps(record,indent=2)+'\n')
raise SystemExit(0 if record['passed'] else 1)
