# Fase 4 - Treino de Camada Linear

Status: **em andamento**.

Esta fase testa aprendizado, nao apenas reconstrucao de matriz.

## Objetivo

Verificar se SAINT consegue aproximar uma funcao linear usando deltas com menos
parametros treinaveis que treino completo da matriz.

Tarefa:

```text
y = W_target x
W_target = W_base + delta_target
```

Durante o treino, `W_base` fica congelada. Cada metodo aprende apenas uma forma
de `delta`.

## Implementado

Modulo:

```text
saint.training
```

Script:

```text
scripts/benchmark_linear_phase4.py
```

Teste:

```text
tests/test_training_linear.py
```

## Baselines

Comparados no benchmark:

- `full_delta`: treina todos os valores do delta;
- `lora_rank_1`, `lora_rank_2`, `lora_rank_4`: delta low-rank;
- `block_scalar_2`: um escalar treinavel por bloco `2x2`;
- `codebook_delta_2`: prototipos `2x2` compartilhados por assinatura de gradiente;
- `saint_routed_delta`: mistura `freeze`, codebook e delta livre por sensibilidade;
- `sparse_sensitivity_delta`: treina apenas os valores mais sensiveis pelo gradiente inicial.

## Como Freeze Foi Modelado

Nesta fase, `freeze` passa a ter a semantica correta para SAINT:

```text
freeze = nao aplicar delta naquela regiao
```

Ou seja, o peso base continua existindo. A regiao congelada apenas nao recebe
parametros treinaveis de adaptacao.

## Benchmark Inicial

Configuracao:

```text
rows: 8
cols: 8
train_samples: 96
test_samples: 32
seed: 11
```

Resultado:

| Metodo | Test Loss | Erro W Rel L1 | Params | Estados Otimizador |
|---|---:|---:|---:|---:|
| full_delta | 0.0000003 | 0.0032 | 64 | 128 |
| codebook_delta_2 | 0.0003089 | 0.0678 | 68 | 136 |
| saint_routed_delta | 0.0004701 | 0.1062 | 56 | 112 |
| sparse_sensitivity_delta | 0.0010289 | 0.2188 | 16 | 32 |
| lora_rank_2 | 0.0034705 | 0.3771 | 32 | 64 |
| block_scalar_2 | 0.0041949 | 0.3829 | 16 | 32 |

## Leitura Inicial

`full_delta` vence em loss, como esperado, porque treina todos os 64 valores do
delta.

`saint_routed_delta` aprende a tarefa com menos parametros:

```text
params: 56 contra 64 do full_delta
optimizer states: 112 contra 128
```

E supera LoRA rank 2 neste caso sintetico:

```text
saint_routed_delta test_loss: 0.0004701
lora_rank_2 test_loss: 0.0034705
```

Isso e um sinal positivo inicial, mas ainda nao conclui a Fase 4. O benchmark
precisa variar sementes, tamanhos, rank de LoRA e orcamentos do roteador.

## Sweep de Sementes

Foi adicionado um sweep com:

```text
seeds: 11, 12, 13, 14, 15
LoRA ranks: 1, 2, 4
SAINT budgets:
  saint_routed_f25_c50
  saint_routed_f25_c25
  saint_routed_f50_c25
```

Arquivos gerados:

```text
runs/phase4_linear_sweep/linear_training_sweep_rows.json
runs/phase4_linear_sweep/linear_training_sweep_summary.json
runs/phase4_linear_sweep/linear_training_sweep_decisions.json
runs/phase4_linear_sweep/linear_training_sweep.md
```

Resumo medio:

| Metodo | Runs | Test Loss Medio | Erro W Rel L1 | Params Medios | Ganho/Parametro |
|---|---:|---:|---:|---:|---:|
| full_delta | 5 | 0.0000002 | 0.0022 | 64.0 | 0.00007163 |
| codebook_delta_2 | 5 | 0.0001678 | 0.0527 | 65.6 | 0.00006772 |
| saint_routed_f25_c50 | 5 | 0.0005749 | 0.0736 | 55.2 | 0.00007290 |
| saint_routed_f50_c25 | 5 | 0.0005749 | 0.0736 | 51.2 | 0.00007862 |
| sparse_sensitivity_delta | 5 | 0.0010238 | 0.1945 | 16.0 | 0.00022256 |
| saint_routed_f25_c25 | 5 | 0.0017572 | 0.1879 | 36.0 | 0.00007854 |
| lora_rank_4 | 5 | 0.0030129 | 0.3720 | 64.0 | 0.00002456 |
| lora_rank_2 | 5 | 0.0035034 | 0.3980 | 32.0 | 0.00003379 |
| block_scalar_2 | 5 | 0.0035227 | 0.3945 | 16.0 | 0.00006638 |
| lora_rank_1 | 5 | 0.0039388 | 0.4184 | 16.0 | 0.00004037 |

## Criterio Automatico

Foi adicionado um criterio automatico para comparar uma variante SAINT contra
uma baseline eficiente:

```text
loss_ratio <= 1.0
parameter_ratio <= 2.0
gain_per_parameter_ratio >= 1.0
```

Decisoes do sweep:

| SAINT | Baseline | Passou? | Leitura |
|---|---|---|---|
| saint_routed_f50_c25 | lora_rank_2 | sim | menor loss e maior ganho/parametro, com ate 2x parametros |
| saint_routed_f25_c50 | lora_rank_2 | sim | menor loss e maior ganho/parametro, com ate 2x parametros |
| saint_routed_f25_c25 | lora_rank_1 | nao | melhor loss e ganho/parametro, mas passou de 2x parametros |

Melhor variante SAINT neste sweep:

```text
saint_routed_f50_c25
test_loss medio: 0.0005749
params medios: 51.2
ganho/parametro: 0.00007862
```

Comparacao contra LoRA rank 2:

```text
saint_routed_f50_c25 test_loss: 0.0005749
lora_rank_2 test_loss: 0.0035034

saint_routed_f50_c25 ganho/parametro: 0.00007862
lora_rank_2 ganho/parametro: 0.00003379
```

Leitura:

```text
SAINT passou o primeiro criterio automatico contra LoRA rank 2.
Ainda nao bate full_delta em loss absoluta.
```

## Sweep de Regimes

Foi adicionado um segundo sweep para testar se o resultado se mantem fora do
caso inicial:

```text
sizes: 8x8, 16x16, 32x32
delta_modes: repeated, dense
seeds: 11, 12
train_samples: 32
test_samples: 16
steps: 90
lora_steps: 140
```

O modo `dense` e menos favoravel a blocos repetidos:

```text
delta_target = matriz densa aleatoria
```

Tambem foi adicionada a baseline:

```text
budgeted_full_delta_for_saint_routed_f50_c25
```

Ela treina valores individuais do delta usando o mesmo numero de parametros da
variante SAINT comparada. Isso mede se o roteador/codebook melhora ou perde para
um full delta esparso com orcamento equivalente.

Resumo agregado do sweep de regimes:

| Metodo | Runs | Test Loss Medio | Params Medios | Ganho/Parametro |
|---|---:|---:|---:|---:|
| codebook_delta_2 | 12 | 0.0029864 | 501.7 | 0.00002779 |
| full_delta | 12 | 0.0033214 | 448.0 | 0.00003320 |
| budgeted_full_delta_for_saint_routed_f50_c25 | 12 | 0.0035874 | 364.0 | 0.00003931 |
| saint_routed_f25_c50 | 12 | 0.0043488 | 391.3 | 0.00003197 |
| saint_routed_f50_c25 | 12 | 0.0043511 | 364.0 | 0.00003434 |
| sparse_sensitivity_delta | 12 | 0.0057709 | 112.0 | 0.00007714 |
| lora_rank_2 | 12 | 0.0098293 | 74.7 | 0.00000073 |
| lora_rank_4 | 12 | 0.0098339 | 149.3 | 0.00000030 |
| lora_rank_1 | 12 | 0.0098532 | 37.3 | 0.00000044 |

Decisoes contra LoRA rank 2 por regime:

| Regime | Passou? | Motivo |
|---|---|---|
| 8x8 dense | sim | menor loss, maior ganho/parametro, parametros dentro do limite |
| 8x8 repeated | sim | menor loss, maior ganho/parametro, parametros dentro do limite |
| 16x16 dense | nao | menor loss, mas parametro_ratio 3.25 > 2.0 |
| 16x16 repeated | nao | menor loss, mas parametro_ratio 3.25 > 2.0 |
| 32x32 dense | nao | menor loss, mas parametro_ratio 6.50 > 2.0 |
| 32x32 repeated | nao | menor loss, mas parametro_ratio 6.50 > 2.0 |

Decisoes contra `budgeted_full_delta` com orcamento equivalente:

| Regime | Passou? | Motivo |
|---|---|---|
| 8x8 dense | nao | loss_ratio 2.47 > 1.0 |
| 8x8 repeated | nao | loss_ratio 3.62 > 1.0 |
| 16x16 dense | nao | loss_ratio 1.43 > 1.0 |
| 16x16 repeated | nao | loss_ratio 1.14 > 1.0 |
| 32x32 dense | nao | loss_ratio 1.15 > 1.0 |
| 32x32 repeated | nao | loss_ratio 1.11 > 1.0 |

Leitura do sweep de regimes:

```text
SAINT manteve vantagem de loss contra LoRA rank 2,
mas nao manteve o limite de parametros em camadas maiores.

Contra full_delta esparso com mesmo orcamento,
SAINT ainda perde em todos os regimes testados.
```

Portanto, a Fase 4 ainda nao deve ser marcada como concluida.

## Criterio de Conclusao

Fase 4 so deve ser marcada como concluida quando SAINT empatar ou superar pelo
menos uma baseline eficiente em um eixo relevante, de forma reproduzivel:

- menor memoria;
- menos parametros;
- menor checkpoint;
- melhor loss para mesmo orcamento;
- melhor ganho por byte.

## Proximos Passos

1. Reduzir crescimento de parametros do `saint_routed_delta` em 16x16 e 32x32.
2. Testar codebook global por camada em vez de prototipos por regiao.
3. Adicionar mais compartilhamento entre regioes para reduzir `parameter_ratio`.
4. Melhorar o roteador para competir com `budgeted_full_delta`.
5. Repetir sweep de regimes antes de fechar a Fase 4.
