# Fase 15 - Escala 14B

Status: **em andamento**.

## Objetivo

Testar gargalos de offload, roteamento e latencia antes de qualquer salto para
70B.

Baseline herdado da Fase 14:

```text
Qwen/Qwen2.5-3B
gradient_sequential subset
budget 16384
delta_application functional
bfloat16
micro-batch 1
```

## Marco 1 - Ponte 14B Controlada

Status: **concluido com bloqueio de treino**.

### Modelo

Modelo escolhido:

```text
Qwen/Qwen2.5-14B
```

Motivo:

- causal LM base;
- mesma familia usada na ponte 3B;
- 14.7B parametros no model card;
- 48 camadas;
- tensor type BF16.

Checkpoint local:

```text
models/Qwen2.5-14B
tamanho local: 29.55 GB
```

### Mudancas

- dependencia `accelerate>=1.12` adicionada ao ambiente HF;
- helper `saint.adapters.huggingface_loading` centraliza:
  - dtype;
  - `device_map`;
  - `max_memory`;
  - `offload_folder`;
  - `low_cpu_mem_usage`;
- benchmark HF aceita:
  - `--hf-device-map`;
  - `--hf-max-memory`;
  - `--hf-offload-folder`;
- script `scripts/benchmark_huggingface_phase15_14b.py` mede load/forward
  controlado sem treino.

### Smoke 14B

Comando:

```bash
python scripts/benchmark_huggingface_phase15_14b.py \
  --model models/Qwen2.5-14B \
  --out runs/phase15_marco1_qwen25_14b_smoke.json \
  --device cuda \
  --model-dtype bfloat16 \
  --hf-device-map auto \
  --hf-max-memory "0=20GiB,cpu=64GiB" \
  --hf-offload-folder runs/phase15_marco1_offload \
  --max-length 8
```

Resultado:

| metrica | valor |
|---|---:|
| load_s | 26.964 |
| forward_s | 1.653 |
| loss | 7.261498 |
| load CUDA GB | 18.634 |
| forward CUDA GB | 19.834 |

Mapa de dispositivo:

```text
GPU: embeddings + camadas 0 a 32
CPU: camadas 33 a 47 + norm + rotary_emb + lm_head
```

### Tentativa de Treino SAINT

Configuracao:

```text
delta_application: inplace
routing_method: activation
budget: 4096
steps: 1
batch_size: 1
routing_max_length: 4
model_dtype: bfloat16
device_map: auto
max_memory: 0=20GiB,cpu=64GiB
max_cuda_gb: 23
```

Resultado:

```text
timeout apos 20 minutos
```

Conclusao:

- load e forward 14B sao viaveis com offload CPU;
- treino SAINT 14B ainda nao e viavel neste caminho;
- a latencia do offload CPU bloqueia o ciclo treino -> checkpoint -> merge;
- LoRA 14B tambem nao deve ser usado como baseline ate o caminho de treino
  parcial ficar mais barato.

### Veredito

```text
Fase 15 Marco 1 passou para smoke, mas falhou para treino.
```

O projeto demonstrou que consegue carregar e executar forward em 14B com
memoria controlada. Ainda nao demonstrou treino estavel, ganho mensuravel ou
tempo aceitavel em 14B.

## Proximo Marco

Marco 2 deve reduzir o custo antes de tentar treino novamente:

- criar roteamento por uma unica matriz alvo em camada residente na GPU;
- permitir selecionar explicitamente camadas baixas, por exemplo layer 0 ou 1;
- evitar tocar camadas offloadadas durante treino;
- testar delta somente em `model.layers.0.self_attn.q_proj.weight`;
- medir forward com `max_memory` menor e maior;
- comparar smoke 14B contra Qwen2.5-3B para custo relativo;
- so tentar LoRA 14B depois que SAINT fizer um step completo abaixo de 5 minutos.

## Marco 2 - Alvo Unico em Camada Residente

Status: **concluido como diagnostico, ainda sem treino viavel**.

### Mudancas

- `saint.adapters.huggingface_forward` aceita `target_names` explicito;
- o runtime SAINT aceita filtrar matriz alvo por `target_device`;
- o benchmark multiseed aceita:
  - `--saint-target-names`;
  - `--saint-target-device`;
- novo probe `scripts/benchmark_huggingface_phase15_target_probe.py` lista
  dispositivo, shape e tamanho das matrizes alvo.

### Probe de Alvos

Comando:

```bash
python scripts/benchmark_huggingface_phase15_target_probe.py \
  --model models/Qwen2.5-14B \
  --out runs/phase15_marco2_target_probe_14b.json \
  --device cuda \
  --model-dtype bfloat16 \
  --hf-device-map auto \
  --hf-max-memory "0=20GiB,cpu=64GiB" \
  --hf-offload-folder runs/phase15_marco2_offload \
  --target-names model.layers.0.self_attn.q_proj.weight,model.layers.33.self_attn.q_proj.weight
```

Resultado:

| matriz | device | shape | numel |
|---|---|---:|---:|
| `model.layers.0.self_attn.q_proj.weight` | `cuda:0` | 5120x5120 | 26214400 |
| `model.layers.33.self_attn.q_proj.weight` | `meta`/CPU offload | 5120x5120 | 26214400 |

Conclusao:

```text
camadas baixas podem ser escolhidas como alvo residente em GPU;
camadas altas ficam offloadadas e nao devem ser usadas no primeiro treino 14B.
```

### Smoke por Memoria

| modelo/config | load_s | forward_s | load CUDA GB | forward CUDA GB |
|---|---:|---:|---:|---:|
| Qwen2.5-3B sem offload | 15.242 | 0.339 | 5.854 | 5.874 |
| Qwen2.5-14B `0=18GiB,cpu=64GiB` | 33.053 | 1.353 | 16.579 | 17.783 |
| Qwen2.5-14B `0=20GiB,cpu=64GiB` | 26.964 | 1.653 | 18.634 | 19.834 |
| Qwen2.5-14B `0=22GiB,cpu=64GiB` | 23.044 | 1.098 | 20.685 | 21.885 |

Observacao:

```text
mais VRAM para o device_map reduz latencia de forward, mas aproxima o pico do
limite pratico da RTX 4090.
```

### Tentativa SAINT Limitada

Configuracao:

```text
target_names: model.layers.0.self_attn.q_proj.weight
target_device: cuda
delta_application: inplace
routing_method: activation
budget: 1024
steps: 1
batch_size: 1
routing_max_length: 4
max_memory: 0=18GiB,cpu=64GiB
max_cuda_gb: 23
```

Resultado:

```text
RuntimeError: CUDA budget exceeded during train: 29.856 GB
```

Leitura:

- a selecao por matriz alvo funcionou;
- a camada 0 `q_proj` estava residente em GPU;
- o caminho de validacao/treino ainda cria pico alto demais antes do step;
- LoRA 14B continua adiado, porque SAINT ainda nao completou um step abaixo de
  5 minutos.

### Veredito

```text
Marco 2 reduziu o escopo corretamente, mas 14B ainda nao esta pronto para treino.
```

O bloqueio agora e mais especifico: nao e escolher a matriz errada, e sim o
pico de memoria/custo do caminho de treino HF com offload.

## Proximo Marco

Marco 3 deve atacar o pico de memoria antes de tentar qualidade:

- evitar segunda carga completa no grid/validacao 14B;
- permitir modo `train-only` sem base eval, merge eval e generation;
- fazer checkpoint do delta sem recarregar o modelo;
- separar script 14B de treino minimo do grid multiseed;
- testar backward somente com uma janela curta e sem avaliacao final;
- medir memoria em subetapas: load, routing, train, checkpoint;
- so reativar LoRA 14B depois de um step SAINT completo abaixo de 5 minutos.

## Marco 3 - Train-Only 14B Sem Grid

Status: **concluido**.

### Mudancas

- criado `scripts/benchmark_huggingface_phase15_train_only.py`;
- o caminho HF aceita `train_only` para pular:
  - base eval;
  - merge eval;
  - generation;
  - validacao final separada;
- o adapter registra tempos de subetapas:
  - load;
  - routing;
  - train;
  - checkpoint payload;
- o adapter ativa `gradient_checkpointing` quando solicitado;
- o treino muda o modelo para `train()` durante o backward e volta para
  `eval()` depois;
- o checkpoint e escrito pelo runtime sem recarregar o modelo.

### Tentativa com 18 GiB

Configuracao:

```text
model: models/Qwen2.5-14B
target: model.layers.0.self_attn.q_proj.weight
device_map: auto
max_memory: 0=18GiB,cpu=64GiB
budget: 1024
steps: 1
max_length: 4
gradient_checkpointing: true
train_only: true
```

Resultado:

```text
RuntimeError: CUDA budget exceeded during train: 29.647 GB
```

Leitura:

```text
18 GiB ainda deixa camadas demais em GPU; o backward com offload cria pico acima
do limite da RTX 4090.
```

### Tentativa com 14 GiB

Comando:

```bash
python scripts/benchmark_huggingface_phase15_train_only.py \
  --model models/Qwen2.5-14B \
  --out runs/phase15_marco3_qwen25_14b_train_only_layer0_14gib \
  --device cuda \
  --model-dtype bfloat16 \
  --hf-device-map auto \
  --hf-max-memory "0=14GiB,cpu=64GiB" \
  --hf-offload-folder runs/phase15_marco3_offload_train_only_14gib \
  --target-names model.layers.0.self_attn.q_proj.weight \
  --target-device cuda \
  --budget 1024 \
  --steps 1 \
  --batch-size 1 \
  --train-texts 1 \
  --max-length 4 \
  --routing-method activation \
  --routing-max-length 4 \
  --routing-batch-size 1 \
  --learning-rate 0.001 \
  --max-cuda-gb 23 \
  --gradient-checkpointing
```

Resultado:

| metrica | valor |
|---|---:|
| status | ok |
| elapsed_s_total | 45.948 |
| train_s | 3.444 |
| routing_s | 2.219 |
| load_s | 37.341 |
| checkpoint_payload_s | 0.000463 |
| train_loss | 6.944987 |
| tokens/s | 0.871 |
| parametros treinaveis | 1024 |
| delta JSON bytes | 81818 |
| optimizer bytes | 163 |
| load CUDA GB | 13.402 |
| routing CUDA GB | 14.687 |
| train CUDA GB | 17.401 |

Checkpoint:

```text
runs/phase15_marco3_qwen25_14b_train_only_layer0_14gib
```

Arquivos:

```text
deltas.saintdelta.json
optimizer.saintopt
checkpoint.json
metrics.json
logs.jsonl
```

### Veredito

```text
Marco 3 passou.
```

SAINT completou um step autograd 14B em modo train-only abaixo de 5 minutos,
com pico de treino abaixo de 23 GB e checkpoint gerado sem merge/eval.

Isto ainda nao prova qualidade. Prova que o caminho minimo:

```text
load -> routing -> train -> checkpoint
```

funciona em 14B com offload agressivo.

## Proximo Marco

Marco 4 deve transformar o smoke em experimento comparavel:

- repetir o train-only com budgets 1024, 4096 e 8192;
- testar `max_memory` 12GiB, 14GiB e 16GiB;
- medir custo de load separado do custo de treino em runs repetidos;
- adicionar avaliacao posterior opcional em processo separado;
- testar LoRA 14B rank 1 somente se couber no mesmo limite CUDA;
- comparar ganho por parametro treinavel contra o primeiro baseline LoRA viavel;
- manter o criterio de abortar se `train_cuda_peak_bytes` passar de 23 GB.
