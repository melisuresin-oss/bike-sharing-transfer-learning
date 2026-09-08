"""Registered adversarial source-domain-invariant Graph-GRU method."""
import math
import torch
from torch import nn
from research.models.graph_gru import GraphGRU
from research.training.losses import log1p_target_mae
from research.training.trainer import NeuralTrainer,TrainerConfig
from research.stage4_v2_2 import core as c


class ReverseGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,coefficient):ctx.coefficient=float(coefficient);return x.view_as(x)
    @staticmethod
    def backward(ctx,gradient):return -ctx.coefficient*gradient,None


def grl(x,coefficient):return ReverseGradient.apply(x,coefficient)
def lambda_at(lam,schedule,j):
    if lam not in c.LAMBDAS or schedule not in c.SCHEDULES or not 0<=j<12000:raise ValueError("Unregistered lambda schedule/index")
    return lam if schedule=="constant" else lam*min(1.,j/2999)
def pool_current_mask(hidden,current_target_mask):
    if hidden.ndim!=3 or current_target_mask.shape!=hidden.shape[:2] or current_target_mask.dtype!=torch.bool:
        raise ValueError("Only current per-example target mask [B,N] is permitted")
    den=current_target_mask.sum(dim=1)
    if (den==0).any():raise ValueError("Empty current target mask; no denominator clamp")
    return (hidden*current_target_mask.unsqueeze(-1)).sum(dim=1)/den.unsqueeze(-1)


class SourceInvariantModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.forecast=GraphGRU(c.model_config())
        self.discriminator=nn.Sequential(nn.Linear(32,64),nn.ReLU(),nn.Dropout(.1),nn.Linear(64,7))
    def objectives(self,model_inputs,target,current_target_mask,domain_class,coefficient):
        output=self.forecast(**model_inputs)
        z=pool_current_mask(output.hidden,current_target_mask)
        logits=self.discriminator(grl(z,coefficient))
        labels=torch.full((len(z),),domain_class,dtype=torch.long,device=z.device)
        if not 0<=domain_class<7:raise ValueError("Source domain class outside seven-source map")
        forecast=log1p_target_mae(output.prediction,target,current_target_mask)
        domain=nn.functional.cross_entropy(logits,labels)
        accuracy=(logits.argmax(dim=-1)==labels).float().mean()
        return forecast,domain,accuracy


def source_optimizer(model):
    opt=torch.optim.AdamW(model.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=.0001)
    owned=[id(p) for g in opt.param_groups for p in g["params"]]
    c.require(len(owned)==len(set(owned)) and set(owned)=={id(p) for p in model.parameters()},"Joint optimizer ownership")
    return opt
def source_step(model,opt,batch,domain_class,coefficient):
    model.train();opt.zero_grad(set_to_none=True)
    fl,dl,accuracy=model.objectives(batch["model_inputs"],batch["target"],batch["mask"],domain_class,coefficient)
    loss=fl+dl
    if not torch.isfinite(loss):raise FloatingPointError("Nonfinite source objective")
    loss.backward()
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):raise FloatingPointError("Missing/nonfinite joint gradient")
    norm=nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
    return {"forecast_loss":float(fl.detach()),"domain_ce":float(dl.detach()),"domain_accuracy":float(accuracy.detach()),
            "lambda_t":coefficient,"gradient_norm_before_clip":float(norm),"valid_targets":int(batch["mask"].sum())}
def adaptation_trainer(forecast,seed):
    if not isinstance(forecast,GraphGRU):raise TypeError("Discriminator must be stripped before adaptation")
    trainer=NeuralTrainer(forecast,c.optimizer_config("adaptation"),TrainerConfig(seed=seed,output_mode="log1p_target",gradient_clip_global_norm=1.,checkpoint_mode="final"))
    c.require(not trainer.optimizer.state and trainer.step==0,"Fresh adaptation optimizer")
    return trainer
