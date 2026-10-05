"""Execute archived MuonH geometry helper with NumPy, flat tree and identity sharding.
No Muon direction computation, JAX lowering, GPU collective or AdamH execution.
"""
import ast,hashlib,json,pathlib,types
import numpy as np
R=pathlib.Path(__file__).resolve().parents[1];P=R/'sources/optimizer_2026_10_05/optimizer.py';tree=ast.parse(P.read_text());checks=[];pins=[]
def tm(f,*trees,**kw):return {k:f(*(t[k] for t in trees)) for k in trees[0]}
def pin(x,p):pins.append(p.shape);return x
env={'jnp':np,'jax':types.SimpleNamespace(tree=types.SimpleNamespace(map=tm)),'_match_named_sharding_to_params':lambda u,p:u,'_pin_sharding':pin}
f=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='_scale_invariant_hyperball_updates');exec(compile(ast.fix_missing_locations(ast.Module(body=[f],type_ignores=[])),str(P),'exec'),env)
def step(p,u,lr):return p+env[f.name]({'p':p},{'p':u},lr)['p']
def ck(name,value):assert value,name;checks.append(name)
p=np.array([[1.,0.],[0.,0.]],dtype=np.float32);u=np.array([[0.,1.],[0.,0.]],dtype=np.float32);lr=.2;q=step(p,u,lr)
ck('Orthogonal direction matches analytic cos(theta)=1/sqrt(1+lr^2)',np.allclose(q[0,0],1/np.sqrt(1+lr**2)))
ck('Orthogonal direction matches analytic sin(theta)=-lr/sqrt(1+lr^2)',np.allclose(q[0,1],-lr/np.sqrt(1+lr**2)))
ck('Nonzero angular step preserves 2D Frobenius norm',np.allclose(np.linalg.norm(q),np.linalg.norm(p)) and not np.allclose(q,p))
ck('Positive direction rescaling leaves geometry unchanged',np.allclose(step(p,u*17,lr),q))
ck('Negative direction scaling reverses angular orientation',np.allclose(step(p,-u,lr),q*np.array([[1,-1],[1,1]])))
ck('Positive parameter scaling yields correspondingly scaled result',np.allclose(step(p*7,u,lr),q*7))
ck('Zero learning rate leaves parameter unchanged',np.allclose(step(p,u,0),p))
ck('Zero direction leaves nonzero parameter unchanged',np.allclose(step(p,np.zeros_like(u),lr),p))
ck('Zero parameter cannot grow under norm multiplier',np.allclose(step(p*0,u,lr),p*0))
ck('Exactly collinear lr=1 synthetic edge collapses at epsilon guard',np.allclose(step(p,p,1),0))
p3=np.stack([p,p*3]);u3=np.stack([u,u*5]);q3=step(p3,u3,lr)
ck('3D stacked update conserves each leading slice norm',np.allclose(np.linalg.norm(q3,axis=(1,2)),[1,3]))
p4=np.array([[[[1.,0.],[0.,0.]],[[2.,0.],[0.,0.]]]],dtype=np.float32);u4=np.zeros_like(p4);u4[0,0,0,1]=1;q4=step(p4,u4,lr)
ck('4D conserves leading slice joint norm',np.allclose(np.linalg.norm(q4),np.linalg.norm(p4)))
a=np.linalg.norm(p4,axis=(2,3));b=np.linalg.norm(q4,axis=(2,3))
ck('4D does not generally conserve each inner matrix norm',not np.allclose(a,b))
ck('Original function retains masked None update',env[f.name]({'p':p},{'p':None},lr)['p'] is None)
ck('Identity pin stub called on intermediate before projection',len(pins)==10)
cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='GrugMoeMuonHConfig');build=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='build');mu=next(x for x in ast.walk(build) if isinstance(x,ast.FunctionDef) and x.name=='muonh_transform');calls=[x for x in ast.walk(mu) if isinstance(x,ast.Call)]
ck('MuonH branch contains no outer optax.scale call',not any(isinstance(x.func,ast.Attribute) and x.func.attr=='scale' for x in calls))
ck('Geometry source pins computed intermediate in both dimensional branches',sum(isinstance(x,ast.Call) and isinstance(x.func,ast.Name) and x.func.id=='_pin_sharding' for x in ast.walk(f))==2)
tangent_u=np.array([[lr,np.sqrt(1-lr**2)],[0,0]],dtype=np.float32);tangent_q=step(p,tangent_u,lr);bound=np.sqrt(2-2*np.sqrt(1-lr**2))
ck('Postprojection tangent step exceeds preprojection eta bound',np.linalg.norm(tangent_q-p)>lr)
ck('Tangent direction attains independently derived postprojection bound',np.allclose(np.linalg.norm(tangent_q-p),bound))
ck('Moment norm ratio is not norm of elementwise normalized direction',not np.isclose(np.linalg.norm(np.array([1.,1.]))/np.sqrt(np.linalg.norm(np.array([1.,100.]))),np.linalg.norm(np.array([1.,1.])/np.sqrt(np.array([1.,100.])))))
out={'checks_passed' :len(checks),'checks':checks,'scope':__doc__,'source_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'artificial_lr':lr,'postprojection_tight_bound_eta_0p2':float(bound),'tangent_update_norm_eta_0p2':float(np.linalg.norm(tangent_q-p)),'orthogonal_angle_degrees':float(np.degrees(np.arctan(lr))),'four_dimensional_inner_norm_before':a.tolist(),'four_dimensional_inner_norm_after':b.tolist(),'four_dimensional_joint_norm_before':float(np.linalg.norm(p4)),'four_dimensional_joint_norm_after':float(np.linalg.norm(q4)),'actual_Muon_direction':None,'actual_JAX_SPMD':None,'actual_GPU_result':None,'historical_execution_sha':None};(R/'analysis/muon_geometry_probe.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print('Muon geometry checks:',len(checks))
