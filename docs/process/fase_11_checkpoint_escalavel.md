# Fase 11 - Checkpoint Escalavel

Status: **concluida**.

Esta fase aprofunda o formato robusto da Fase 10 para permitir retomada real de
treino e payloads maiores.

## Objetivo

Escalar checkpoints SAINT para runs mais longos:

```text
treinar -> salvar AdamW + shards -> retomar -> continuar treino -> fundir
```

## Implementado

- estado completo de AdamW no caminho `drm_saint_autograd_smoke`;
- restauracao de AdamW com `optimizer.load_state_dict`;
- retomada real via `metadata.resume_run`;
- deltas cumulativos apos continuation;
- shards de payload por limite de bytes;
- leitura de shards com validacao SHA-256;
- leitura por `mmap` para payloads de matriz;
- dtypes `float32`, `float16`, `bfloat16` e `int8`;
- ponto de migracao por `format_version`;
- teste de checkpoint shardado em `float16`.

## Configuracao

Campos opcionais em `metadata`:

```text
checkpoint_dtype: float32 | float16 | bfloat16 | int8
checkpoint_shard_bytes: tamanho maximo aproximado por shard
resume_run: diretorio de run anterior
```

## Smoke DRM Autograd

Fluxo validado:

```text
run 1:
  train -> checkpoint AdamW + deltas

run 2:
  resume_run=run 1
  restaurar deltas
  restaurar AdamW
  continuar treino
  salvar checkpoint shardado float16
```

Resultado:

```text
first_loss: 4.1385
resume_initial_loss: 4.1385
second_loss: 4.1327
optimizer: AdamW
has_adamw_state: true
delta_format: saint_matrix_shards
dtype: float16
shards: 6
shape_validation: true
```

## Veredito

```text
Fase 11 concluida em escala smoke.
```

O resultado prova que o runtime consegue continuar um treino real do
`drm_transformer` a partir de deltas e estado AdamW salvos em checkpoint SAINT.

## Pendencias Futuras

- testar shards com checkpoints muito maiores;
- fazer merge lendo apenas subconjuntos necessarios;
- medir custo de I/O por dtype;
- adicionar migracoes reais quando `format_version` passar de 1 para 2;
- validar `bfloat16` e `int8` contra perda de qualidade em tarefa real.
