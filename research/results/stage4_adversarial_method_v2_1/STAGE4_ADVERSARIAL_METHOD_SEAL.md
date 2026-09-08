# Adversarial Domain-Generalization Executable-Method Seal

**Status:** `FROZEN_VERIFIED`  
**Authorization:** executable method specification only; implementation and scientific execution are not authorized.

## Scientific identity and frozen context

The method is **GRL-based multi-source domain-generalized pretraining**, also called **adversarial source-domain-invariant pretraining**. It is not canonical source-target DANN: the unseen pseudo-target is excluded from source pretraining and is never a discriminator class.

The backbone is frozen `GGRU_K04_H032_D00` (`k=4`, hidden 32, backbone dropout 0.0) under `LOG1P_TARGET_MAE`. The development grid is `lambda_max` in `{0.01,0.10,0.50}` crossed with `{constant,linear}`, giving six configurations over Bilbao, Cardiff, Freiburg, and Goteborg folds with seeds 17, 29, and 43.

## Forecast and adversarial objective

Forecast loss remains `LOG1P_TARGET_MAE`. The discriminator uses ordinary unweighted seven-class cross-entropy with unit forward weight and no separate alpha. GRL is identity in the forward pass. In backward propagation it multiplies only the domain gradient entering the shared Graph-GRU representation by `-lambda_t`. Discriminator parameters receive ordinary unscaled CE gradients; the forecasting head receives no domain-loss gradient.

## Domain representation and classifier

For each city-hour graph, pool final station hidden states as `z=sum_n(M_target[n]*h[n])/sum_n(M_target[n])`. `M_target` is only the current-hour valid-station/coverage mask; no target demand value is supplied, `M_hist` is not used, and an empty mask raises an error rather than clamping the denominator.

The classifier is `Linear(32,64) -> ReLU -> explicit torch.nn.Dropout(0.10) -> Linear(64,7)`. Dropout is invoked after ReLU, active in train mode, and disabled in eval mode. It is not recurrent/inter-layer GRU dropout.

## Lambda schedules

There are 12,000 zero-indexed source optimizer updates `j=0,...,11999`. Constant uses `lambda(j)=lambda_max`. Linear uses `lambda(j)=lambda_max*min(1,j/2999)`, so `lambda(0)=0`, `lambda(2998)<lambda_max`, and `lambda(2999)=lambda(3000)=lambda_max` exactly. Ganin's sigmoid schedule is prohibited. The constant arm is not labeled degenerate and carries no assumed outcome.

## Optimizer, balancing, and diagnostics

One AdamW optimizer jointly owns the Graph-GRU encoder, forecast head, and discriminator: LR `1e-3`, weight decay `1e-4`, betas `(0.9,0.999)`, epsilon `1e-8`. Global gradient-norm clipping 1.0 covers all owned parameters. There is no separate discriminator optimizer or LR.

Source fitting must use `EqualCitySampler`; 12,000 updates across seven sources must differ by at most one exposure. CE is unweighted and has no inverse-frequency weights. Each future fit logs forecast loss, domain CE, domain accuracy, `lambda_t`, and cumulative exposure per source. Chance accuracy `1/7=0.142857142857` and uniform CE `ln(7)=1.94591014906` are descriptive only and cannot select, stop, alter lambda, or invalidate a run. The random-encoder diagnostic is omitted.

## Adaptation, inference, and workload

After source pretraining, discard the discriminator and retain Graph-GRU plus forecast head. Reset AdamW and apply the unchanged Stage 3 policy: full-network target-only adaptation on seven days for exactly 300 updates, no domain loss/discriminator, and no early stopping. The discriminator is absent at inference.

The sealed expected workload is `6 x 4 x 3 = 72` adversarial source fits, 72 dependent seven-day/300-update adaptations, and 72 post-adaptation evaluations: 144 parameter-changing fits. Stage 4 selection has zero zero-shot evaluations. This is an enumeration, not execution authorization.

## Future software-validation gate

Scientific execution is prohibited until every requirement in the machine-readable `future_software_gates` list passes, including a small non-scientific synthetic GRL/adversarial optimization test. In particular, the actual path must prove GRL gradients, exact pooling, discriminator dropout, schedule boundaries, optimizer ownership/clipping, EqualCitySampler balance, pseudo-target exclusion, discriminator stripping, fresh adaptation state, checkpoint round-trip, reproducibility, and the final-label firewall.

## Unresolved final-execution blockers

The final three-versus-five seed contradiction and exact seed-by-temporal-bootstrap aggregation convention remain unresolved. The frozen mapping `0/1/7/30/full -> 0/100/300/600/1200`, graph batch size 16, audited effective passes, no early stopping, zero-shot zero-day contract, Stage 2 reporting, and evaluation hierarchy remain unchanged. These blockers do not prevent this development-only method seal but do prevent later final execution/evaluation authorization.
