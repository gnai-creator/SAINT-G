# Fase 14 - Escala 3B

Status: **em andamento**.

## Objetivo

Preparar a passagem de modelos pequenos para modelos de escala maior em GPU
domestica, sem pular diretamente para 3B.

## Marco 1 - Ponte GPT-2 Small

Status: **concluido**.

Este marco usa `gpt2` como ponte entre `sshleifer/tiny-gpt2` e um modelo 3B.

### Modelo

```text
modelo: gpt2
parametros: 124.439.808
local: models/gpt2
```

### Comando

```bash
python scripts/benchmark_huggingface_multiseed_phase13.py \
  --model models/gpt2 \
  --corpus data/tinyshakespeare_phase13.txt \
  --out runs/phase14_marco1_gpt2_bridge \
  --device cuda \
  --steps 2 \
  --batch-size 4 \
  --seeds 31,32,33 \
  --saint-budgets 16 \
  --saint-lrs 0.001 \
  --lora-ranks 2 \
  --lora-lrs 0.001
```

### Resultado CUDA

| metodo | count | mean val loss | best val loss | mean gain/param |
|---|---:|---:|---:|---:|
| SAINT | 3 | 6.814889 | 6.814889 | 0.00000200 |
| LoRA | 3 | 6.808302 | 6.806123 | 0.00000399 |

Pico CUDA observado:

```text
SAINT: 2.083841536 GB
LoRA:  1.016675840 GB
```

Artefato LoRA carregado:

```text
lora_loaded_validation_loss: 6.80828857421875
lora_loaded_perplexity: 905.3200922133876
```

Geracao simples:

```text
SAINT      -> SAINT-RADIO-SOUTH
Checkpoint -> Checkpoint\n\nThe following is a list of
Training   -> Training.\n\nThe first step is to
```

### Veredito

```text
nao avancar ainda para 3B
```

Motivos:

- LoRA venceu SAINT em validation loss media;
- LoRA venceu em ganho por parametro;
- LoRA usou menos pico CUDA neste caminho atual;
- SAINT ainda carrega mais estado/runtime do que deveria para esse porte;
- o caminho SAINT precisa reduzir memoria e melhorar selecao de deltas antes do
  experimento 3B.

### Problema Corrigido

O smoke em GPT-2 small revelou um bug no merge quando o adapter Hugging Face
usava `max_dim` menor que a matriz real. O delta salvo estava no shape completo
da matriz, enquanto o peso base do merge estava fatiado.

Correcao:

```text
delta_payload agora corta o delta para o mesmo shape da base antes de salvar.
```

## Proximo Marco

Marco 2 deve otimizar o caminho SAINT antes do 3B:

- evitar segunda carga completa do modelo no `make_task`;
- reduzir payload de delta para salvar apenas valores treinaveis;
- permitir merge/eval sem materializar matrizes fatiadas desnecessarias;
- aumentar budget SAINT e comparar curva contra LoRA rank 2/4;
- medir memoria por etapa: load, train, checkpoint, merge;
- so avancar para 3B se SAINT ficar competitivo em GPT-2 small.
