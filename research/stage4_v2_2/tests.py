"""Synthetic implementation gate. Never run a scientific job or open label arrays."""
from __future__ import annotations
from collections import Counter
from dataclasses import asdict
import copy
import hashlib
import inspect
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
from unittest.mock import patch
import numpy as np
import torch
from research.stage4_v2_2 import core as c,a40,method,selection,controller
from research.stage4_v2_2.verify_selection import run_checks
from research.stage4_v2_2.sampler import EqualCitySampler,SOURCE,EXPECTED
from research.stage1_v2_2 import core as frozen
from research.v2_2.snapshots import FitRequest
from research.development_data_v2_2.registry import DevelopmentRegistry


def rejects(fn,types=(Exception,)):
    try:fn()
    except types:return True
    return False


def run():
    original_root=c.ROOT;root=original_root/"tmp/stage4_v2_2_synthetic_tests"/uuid.uuid4().hex
    root.mkdir(parents=True)
    (root/"PURPOSE.json").write_text(json.dumps({"purpose":"NON_SCIENTIFIC_IMPLEMENTATION_FIXTURES","real_optimizer_updates":0,"real_evaluations":0}))
    checks=[];integration=[]
    def check(name,condition):
        ok=bool(condition);checks.append({"test":name,"status":"PASS" if ok else "FAIL"})
        if not ok:raise AssertionError(name)
    a40.configure_runtime()
    # Read only frozen metadata/hashes. No development/final label arrays are opened.
    for p,h in {**c.UPSTREAM,**c.PINNED_DATA}.items():check("upstream_hash:"+p,c.sha(p)==h)
    independent=run_checks();check("independent_ranking_eight_cases",independent["synthetic_tests"]==8 and independent["status"]=="PASS")
    check("stage3_contract_binding",c.UPSTREAM[c.STAGE3]=="f5cb2653dd097eb1288f419884778ae01e9034e9e1d88006b33beac4cff069b5")
    jobs=c.registered_jobs();c.validate_jobs(jobs);sources=[j for j in jobs if j["phase"]=="source"];adapts=[j for j in jobs if j["phase"]=="adaptation"]
    check("source_fit_cardinality_72",len(sources)==72)
    check("adaptation_fit_cardinality_72",len(adapts)==72)
    check("total_144_unique_fits",len(jobs)==len({j['job_id'] for j in jobs})==144)
    check("pseudo_target_exclusion_all_folds",all(j["pseudo_target"] not in j["source_cities"] for j in sources))
    check("seven_source_domains_exactly",all(set(j["source_cities"])==set(c.DEVELOPMENT)-{j["pseudo_target"]} and len(j["source_cities"])==7 for j in sources))
    check("source_domain_labels",all(j["domain_classes"]=={str(i):k for k,i in enumerate(sorted(j["source_cities"]))} for j in sources))
    check("source_updates_12000",all(j["updates"]==12000 for j in sources))
    check("adaptation_updates_300",all(j["updates"]==300 for j in adapts))
    check("adaptation_one_per_source_dependency",{j["depends_on"] for j in adapts}=={j["job_id"] for j in sources})
    check("frozen_sampler_bytes",hashlib.sha256(SOURCE.read_bytes()).hexdigest()==EXPECTED)
    for target in c.FOLDS:
        j=next(j for j in sources if j["pseudo_target"]==target);seq=EqualCitySampler(tuple(j["source_cities"]),j["source_schedule_seed"]).sequence(12000)
        expected=[j["source_cities"][i%7] for i in range(12000)];__import__('random').Random(j['source_schedule_seed']).shuffle(expected)
        check(f"equal_city_sampler_{target}",seq==expected and max(Counter(seq).values())-min(Counter(seq).values())==1 and target not in seq)
    q=c.Contract();reg=DevelopmentRegistry(c.ROOT,c.DATA,c.DATA_SHA)
    check("v22_only_cache",c.read(c.CACHE)["protocol_version"]=="2.2" and not c.read(c.CACHE)["retrospective_labels_in_cache"])
    for city in (195,199,237,617):
        check(f"final_predictor_firewall_{city}",rejects(lambda:reg.predictors(city),(PermissionError,)))
        check(f"final_label_firewall_{city}",rejects(lambda:reg.retrospective_labels(city),(PermissionError,)))
    check("v21_predictor_firewall",rejects(lambda:reg.path('processed/protocol_v2_1/development/development_panel.parquet'),(PermissionError,)))
    for target in c.FOLDS:
        req=FitRequest.registered(q,phase="development",kind="adaptation",budget="7",target_city=target)
        meta=c.read(next(j for j in adapts if j["pseudo_target"]==target)["snapshot"]["path"])
        check(f"elapsed_7day_no_extension_{target}",req.start==q.boundaries["HD"]-168*c.HOUR and req.cutoff==q.boundaries["HD"] and c.metadata_equal(asdict(req),meta["request"]))
        bad=copy.copy(asdict(req));bad["start"]-=c.HOUR
        check(f"backfill_request_rejected_{target}",rejects(lambda:FitRequest(**bad).validate(q)))
    for lam in c.LAMBDAS:
        x=torch.tensor([1.,-2.,3.],requires_grad=True);g=torch.tensor([2.,-3.,.5]);y=method.grl(x,lam)
        check(f"grl_forward_identity_{lam}",torch.equal(x,y));y.backward(g)
        check(f"grl_exact_negative_magnitude_{lam}",torch.equal(x.grad,-lam*g))
        check(f"constant_schedule_{lam}",all(method.lambda_at(lam,'constant',j)==lam for j in (0,1,2999,3000,11999)))
        check(f"linear_endpoints_{lam}",[method.lambda_at(lam,'linear',j) for j in (0,1,2999,3000,11999)]==[0.,lam*(1/2999),lam,lam,lam])
    zero=torch.ones(2,requires_grad=True);method.grl(zero,0.).sum().backward()
    check("grl_lambda_zero_encoder_gradient",torch.equal(zero.grad,torch.zeros_like(zero)))
    check("schedule_out_of_budget_rejected",rejects(lambda:method.lambda_at(.1,'linear',12000),(ValueError,)))
    h=torch.tensor([[[1.,2.],[100.,200.],[3.,4.]]],requires_grad=True);mask=torch.tensor([[True,False,True]])
    pooled=method.pool_current_mask(h,mask);check("current_mask_mean_only",torch.equal(pooled,torch.tensor([[2.,3.]])))
    pooled.sum().backward();check("pool_gradient_excludes_unobserved_nodes",torch.equal(h.grad,torch.tensor([[[.5,.5],[0.,0.],[.5,.5]]])))
    check("historical_mask_pooling_rejected",rejects(lambda:method.pool_current_mask(h,torch.ones(1,24,3,dtype=torch.bool)),(ValueError,)))
    check("demand_weight_pooling_rejected",rejects(lambda:method.pool_current_mask(h,torch.tensor([[1.,0.,3.]])),(ValueError,)))
    check("empty_current_mask_rejected",rejects(lambda:method.pool_current_mask(h,torch.zeros(1,3,dtype=torch.bool)),(ValueError,)))
    check("pooling_interface_no_history_or_demand",list(inspect.signature(method.pool_current_mask).parameters)==['hidden','current_target_mask'])
    torch.manual_seed(17);model=method.SourceInvariantModel();model.eval();inputs=a40.toy();targets=torch.ones(2,3);masks=torch.ones(2,3,dtype=torch.bool)
    disc=model.discriminator
    check("discriminator_architecture",[(disc[0].in_features,disc[0].out_features),type(disc[1]).__name__,disc[2].p,(disc[3].in_features,disc[3].out_features)]==[(32,64),'ReLU',.1,(64,7)])
    check("frozen_backbone_3403_total_5970",sum(p.numel() for p in model.forecast.parameters())==3403 and sum(p.numel() for p in model.parameters())==5970 and model.forecast.config==c.model_config())
    z=torch.ones(16,32);disc.train();torch.manual_seed(17);d1=disc(z);d2=disc(z)
    check("discriminator_dropout_active_in_train",not torch.equal(d1,d2));disc.eval()
    check("discriminator_deterministic_in_eval",torch.equal(disc(z),disc(z)))
    torch.manual_seed(17);one=method.SourceInvariantModel();torch.manual_seed(17);two=method.SourceInvariantModel()
    check("deterministic_registered_initialization",all(torch.equal(v,two.state_dict()[k]) for k,v in one.state_dict().items()))
    _,loss,_=model.objectives(inputs,targets,masks,0,.1);loss.backward()
    check("no_direct_domain_gradient_forecast_head",all(p.grad is None for p in model.forecast.head.parameters()))
    check("domain_gradient_reaches_encoder",any(p.grad is not None and p.grad.abs().sum()>0 for p in model.forecast.cell.parameters()))
    dg={k:p.grad.clone() for k,p in disc.named_parameters()};eg={k:p.grad.clone() for k,p in model.forecast.cell.named_parameters()}
    model.zero_grad(set_to_none=True);_,loss2,_=model.objectives(inputs,targets,masks,0,.5);loss2.backward()
    check("unit_ce_discriminator_not_alpha_scaled",all(torch.equal(dg[k],p.grad) for k,p in disc.named_parameters()))
    check("encoder_gradient_lambda_ratio",all(torch.allclose(5*eg[k],p.grad,rtol=1e-5,atol=1e-7) for k,p in model.forecast.cell.named_parameters()))
    opt=method.source_optimizer(model);check("joint_fresh_adamw_owns_all_once",not opt.state and len(opt.param_groups)==1 and len(opt.param_groups[0]['params'])==len(list(model.parameters())))
    log=method.source_step(model,opt,{'model_inputs':inputs,'target':targets,'mask':masks},0,.1)
    check("synthetic_joint_step_all_params_clip1",all(int(s['step'])==1 for s in opt.state.values()) and torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in model.parameters()]))<=1.000001 and log['valid_targets']==6)
    trainer=method.adaptation_trainer(model.forecast,17)
    check("fresh_adaptation_optimizer_no_discriminator",not trainer.optimizer.state and trainer.step==0 and {id(p) for g in trainer.optimizer.param_groups for p in g['params']}=={id(p) for p in model.forecast.parameters()})
    check("adaptation_optimizer_frozen",trainer.optimizer_config==c.optimizer_config('adaptation') and trainer.config.gradient_clip_global_norm==1 and trainer.config.checkpoint_mode=='final')
    check("adaptation_rejects_discriminator_wrapper",rejects(lambda:method.adaptation_trainer(model,17),(TypeError,)))
    trainer.train_step({'model_inputs':inputs,'target':targets,'mask':masks})
    check("synthetic_adaptation_step",trainer.step==1 and all(int(s['step'])==1 for s in trainer.optimizer.state.values()))
    left={'x':(np.int64(3),{'y':np.float64(.9),'p':Path('example')}),'b':True}
    right={'x':[3,{'y':.9,'p':'example'}],'b':True}
    check("recursive_representation_canonicalization",c.metadata_equal(left,right))
    check("genuine_value_difference_rejected",not c.metadata_equal(right,{**right,'x':[3,{'y':.9000000000000001,'p':'example'}]}))
    check("missing_key_rejected",not c.metadata_equal(right,{'x':right['x']}))
    check("no_default_or_rounding",not c.metadata_equal({}, {'x':None}) and not c.metadata_equal([1.00001],[1.00002]))
    check("ordered_list_preserved",not c.metadata_equal([1,2],[2,1]))
    check("nonfinite_metadata_rejected",rejects(lambda:c.canonicalize(float('nan')),(ValueError,)))
    # Fully synthetic origin cache exercises the production City wrapper and inherited loader.
    fixture_root=root/'cache_fixture';fixture_root.mkdir()
    arrays={'x_hist':np.zeros((5,24,3,2),np.float32),'m_hist':np.zeros((5,24,3),bool),'x_week':np.zeros((5,3,2),np.float32),
            'x_static':np.zeros((3,2),np.float32),'x_calendar':np.arange(30,dtype=np.float32).reshape(5,6),'adjacency':np.eye(3,dtype=np.float32),
            'origin_us':np.arange(5,dtype=np.int64)*c.HOUR,'station_ids':np.arange(3),'origin_feature_sha256':np.array([str(i) for i in range(5)],dtype='S64'),
            'fit_count':np.arange(15,dtype=np.float32).reshape(5,3),'fit_mask':np.ones((5,3),bool),'adaptation_7d_mask':np.array([[False]*3]*3+[[True,False,True]]*2),
            'source_anchors':np.arange(5),'adaptation_anchors':np.array([3,4])}
    entries={}
    for k,v in arrays.items():
        p=fixture_root/(k+'.npy');np.save(p,v);entries[k]={'path':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    graph={'city_id':129,'k':4,**entries['adjacency']}
    with patch.object(c,'ROOT',fixture_root),patch.object(frozen,'ROOT',fixture_root),patch.object(c,'read',return_value={'graphs':[graph]}):
        city=a40.City({'city_id':129,'arrays':entries},'cpu');ix=np.array([4,1]);batch=city.batch(ix,'source')
        check("causal_predictor_origin_preserved",np.array_equal(city.origin_us[ix],arrays['origin_us'][ix]) and torch.equal(batch['model_inputs']['x_calendar'],torch.from_numpy(arrays['x_calendar'][ix])) and torch.equal(batch['model_inputs']['x_hist'],torch.from_numpy(arrays['x_hist'][ix])))
        check("current_fit_mask_not_history_mask",batch['mask'].all() and not batch['model_inputs']['m_hist'].any())
        check("adaptation_mask_no_backfill",torch.equal(city.batch(np.array([3,4]),'adaptation')['mask'],torch.from_numpy(arrays['adaptation_7d_mask'][[3,4]])))
        check("no_pseudo_target_source_rows",set(j['source_cities'])==set(seq) and j['pseudo_target'] not in seq)
    # Synthetic completed checkpoint schema, not a trained candidate. Artificial step counters
    # exercise validators only; no production package accepts these fixture jobs or manifest.
    valroot=root/'validator_fixture';valroot.mkdir();fake_manifest='0'*64
    sj=copy.deepcopy(sources[0]);sj['job_id']='synthetic_validator_source'
    with patch.object(c,'ROOT',valroot),patch.object(c,'OUT','o/'):
        for rel in (c.CONTRACT,c.JOBMAP):
            p=c.path(rel);p.parent.mkdir(parents=True,exist_ok=True);p.write_text('{"purpose":"SYNTHETIC_ONLY"}')
        rt={'settings':a40.SETTINGS,'device':'NVIDIA A40','execution_environment':c.ENVIRONMENT,'purpose':'SYNTHETIC_ONLY_NOT_RUNTIME_ATTESTATION'}
        pf=c.write(c.OUT+'preflight/synthetic.json',{'status':'PASS','manifest_sha256':fake_manifest,'workers':4,'scientific_optimizer_updates':0,'scientific_evaluations':0,'runtime':rt,'scientific_contract_sha256':c.sha(c.CONTRACT),'job_map_sha256':c.sha(c.JOBMAP)})
        for state in opt.state.values():state['step']=torch.tensor(12000.)
        seq=EqualCitySampler(tuple(sj['source_cities']),sj['source_schedule_seed']).sequence(12000);counts=dict(Counter(seq))
        cp=a40.save_checkpoint(c.OUT+'attempts/'+sj['job_id']+'/synthetic/final.pt',model,opt,a40.metadata(sj,fake_manifest,rt,None),np.random.default_rng(17),counts)
        sl=[{'step':step,'lambda_t':method.lambda_at(sj['lambda_max'],sj['schedule'],step-1),'forecast_loss':0.,'domain_ce':0.,'domain_accuracy':0.,'city_updates':{str(k):v for k,v in counts.items()}} for step in (1,12000)]
        rec={'status':'COMPLETED_VALIDATED_FIT','completed_fit':True,'evaluation_completed':False,'job':sj,'manifest_sha256':fake_manifest,'runtime':rt,'preflight':pf,'checkpoint':cp,'metric':None,'prediction_commitment':None,'optimizer_initial_state_empty':True,'checkpoint_roundtrip_max_abs':0.,'logs':sl}
        a40.commit_completion(sj,rec);check("synthetic_completed_checkpoint_validation",a40.validate_completed(sj,fake_manifest,{})==rec)
        with patch.object(a40,'verify_package',return_value=([sj],{})),patch.object(a40,'runtime',return_value=rt),patch.object(method,'source_optimizer',side_effect=AssertionError('optimizer forbidden on skip')):
            a40.run_job(sj['job_id'],fake_manifest,pf)
        check("resume_valid_completed_skips_optimizer",True)
        raw=c.path(cp['path']).read_bytes();c.path(cp['path']).write_bytes(raw+b'corrupt')
        check("checkpoint_hash_corruption_rejected",rejects(lambda:a40.validate_completed(sj,fake_manifest,{})))
        c.path(cp['path']).write_bytes(raw)
        prediction=c.write_bytes(c.OUT+'synthetic_prediction.npz',b'SYNTHETIC_PREDICTION_BYTES');c.path(prediction['path']).write_bytes(b'changed')
        check("prediction_hash_corruption_rejected",rejects(lambda:c.bound(prediction)))
        comp=c.path(a40.completed_path(sj['job_id']));rawcomp=comp.read_bytes();comp.write_bytes(rawcomp+b' ')
        check("completion_record_hash_corruption_rejected",rejects(lambda:a40.validate_completed(sj,fake_manifest,{})));comp.write_bytes(rawcomp)
        commitfile=c.path(a40.commit_path(sj['job_id']));rawcommit=commitfile.read_bytes()
        payload=torch.load(io.BytesIO(raw),weights_only=True)
        for variant in ('metadata_value','metadata_key','partial_budget','parameter_shape','completed_fit_false'):
            altered=copy.deepcopy(payload);changed=copy.deepcopy(rec)
            if variant=='metadata_value':altered['metadata']['optimizer']['learning_rate']+=1e-15
            elif variant=='metadata_key':del altered['metadata']['partial_resumed']
            elif variant=='partial_budget':next(iter(altered['optimizer_state']['state'].values()))['step']=torch.tensor(11999.)
            elif variant=='parameter_shape':altered['model_state'][next(iter(altered['model_state']))]=torch.zeros(1)
            else:changed['completed_fit']=False
            buf=io.BytesIO();torch.save(altered,buf);c.path(cp['path']).write_bytes(buf.getvalue());changed['checkpoint']=c.entry(cp['path'])
            comp.write_text(json.dumps(changed));commitfile.write_text(json.dumps({'status':'IMMUTABLE_COMPLETION_COMMITTED','job_id':sj['job_id'],'completion':c.entry(a40.completed_path(sj['job_id']))}))
            check('rehash_does_not_hide_'+variant,rejects(lambda:a40.validate_completed(sj,fake_manifest,{})))
            c.path(cp['path']).write_bytes(raw);comp.write_bytes(rawcomp);commitfile.write_bytes(rawcommit)
        half={**sj,'job_id':'synthetic_half_commit'};c.write(a40.completed_path(half['job_id']),{'purpose':'SYNTHETIC_ONLY'})
        check("half_committed_job_stops_resume",rejects(lambda:a40.validate_existing(fake_manifest,verified=([sj,half],{}))))
    with patch.object(c,'ROOT',root/'partial_only'),patch.object(c,'OUT','o/'):
        c.write(c.OUT+'attempts/synthetic_partial/first/started.json',{'purpose':'SYNTHETIC_ONLY','optimizer_updates':100})
        vr=a40.validate_existing(fake_manifest,verified=([sj],{}))
        check("partial_never_completed_or_resumed",vr['completed']==0 and vr['partial_attempts']==1 and 'fresh UUID' in vr['partial_policy'])
    check("strict_a40_rejects_local_cpu_python",rejects(a40.runtime))
    # Exercise the actual PowerShell Start-Process interface, real module imports and shared
    # four-worker dispatcher with stdlib-only fake jobs. No production authorization exists.
    ps=original_root/'deployment/stage4_v2_2_a40/run_stage4_a40.ps1'
    env=os.environ.copy();env['PYTHONPATH']=str(original_root/'tmp/neural_build/pydeps')+os.pathsep+str(original_root)
    for scenario in ('success','failure','missing_completion'):
        out=root/('detached_'+scenario);out.mkdir()
        fixture_jobs=[{'job_id':'synthetic_source','depends_on':None,'fail':scenario=='failure','omit_completion':scenario=='missing_completion'},
                      {'job_id':'synthetic_adaptation','depends_on':'synthetic_source'}]
        if scenario=='success':
            fixture_jobs.extend({'job_id':f'synthetic_source_{i}','depends_on':None} for i in range(3))
            fixture_jobs.extend({'job_id':f'synthetic_adaptation_{i}','depends_on':f'synthetic_source_{i}'} for i in range(3))
        cfg={'purpose':'NON_SCIENTIFIC_CONTROLLER_STARTUP_FIXTURE','output_root':str(out),'jobs':fixture_jobs}
        cfgpath=root/(scenario+'.json');cfgpath.write_text(json.dumps(cfg))
        call=subprocess.run(['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',str(ps),'-Mode','launch','-Workers','4','-StartupFixture',str(cfgpath),'-FixturePython',sys.executable],cwd=original_root,env=env,capture_output=True,text=True,timeout=30,creationflags=subprocess.CREATE_NO_WINDOW)
        check('detached_launcher_exit_'+scenario,call.returncode==0)
        launch=json.loads(call.stdout);deadline=time.monotonic()+40
        while time.monotonic()<deadline and not any((out/n).exists() for n in ('controller_complete.json','controller_failed.json')):time.sleep(.25)
        evidence={'scenario':scenario,'launch':launch,'output_root':str(out)};integration.append(evidence)
        if scenario=='success':
            check('actual_detached_controller_startup',launch['CoordinatorPID']>0 and (out/'controller_complete.json').exists())
            exits=[json.loads(p.read_text()) for p in out.glob('*.exit.json')]
            check('successful_workers_EXITED_ZERO_WITH_COMPLETION',len(exits)==8 and all(e['status']=='EXITED_ZERO_WITH_COMPLETION' and e['exit_code']==0 for e in exits))
        else:
            check('controller_stops_'+scenario,(out/'controller_failed.json').exists() and not (out/'synthetic_adaptation.launch.json').exists())
            er=json.loads((out/'synthetic_source.exit.json').read_text())
            check('genuine_exit_or_missing_commit_detected_'+scenario,er['status']=='FAILED' and (er['exit_code']!=0 if scenario=='failure' else er['exit_code']==0 and er['validation_error'] is not None))
    check("controller_cli_matches_worker_parser",controller.worker_command(sources[0],'a'*64,{'path':'pf','sha256':'b'*64})[5:7]==['research.stage4_v2_2.a40','job'])
    tested=[p.relative_to(original_root).as_posix() for p in (original_root/'research/stage4_v2_2').glob('*.py')]+['deployment/stage4_v2_2_a40/run_stage4_a40.ps1']
    result={'status':'PASS','purpose':'SYNTHETIC_IMPLEMENTATION_GATE','check_count':len(checks),'checks':checks,'independent_ranking_tests':independent,
            'tested_code_sha256':{p:c.sha(p) for p in tested},
            'actual_detached_controller_integration':integration,'synthetic_optimizer_updates':2,'real_scientific_optimizer_updates':0,'real_scientific_evaluations':0,
            'final_target_labels_accessed':False,'final_experiment_started':False,'fixture_root':str(root),'timestamp_utc':c.utc()}
    (root/'test_report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':'PASS','check_count':len(checks),'report':str(root/'test_report.json')}),flush=True)
    return result


if __name__=='__main__':run()
