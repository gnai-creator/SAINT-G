# Fase DRM-G - DRM-SAINT-G

Status: **pendente**.

## Objetivo

Testar crescimento progressivo do `drm_transformer` por enxertos treinaveis.

A ideia central e evitar tentar nascer com um DRM grande demais para o hardware
disponivel. Em vez disso, o modelo cresce por ciclos pequenos, preservando um
nucleo congelado e treinando apenas enxertos compactos com DRM-SAINT-G-Phi.

## Hipotese

DRM-SAINT-G deve permitir:

```text
DRM pequeno estavel
  -> modulo novo anexado
  -> treino local por DRM-SAINT-G-Phi
  -> validacao global
  -> consolidacao progressiva
```

Se funcionar, o projeto ganha um caminho mais realista para sair de modelos
pequenos sem exigir pre-training full de um modelo gigante.

## Modelo Mental

O enxerto nao e apenas LoRA em uma matriz existente.

Ele e um modulo novo ou uma extensao estrutural do DRM. DRM-SAINT-G-Phi treina a ponte
entre o nucleo antigo e a capacidade nova:

```text
Delta W = A Phi B
```

onde:

- `A` projeta o estado do DRM para um espaco local;
- `Phi` e o operador relacional/geometrico treinavel;
- `B` projeta o resultado de volta para o fluxo do modelo.

## Entregas

- especificacao de `DRMGraftConfig`;
- registro de modulos enxertaveis do DRM;
- congelamento explicito do nucleo antigo;
- criacao de enxertos pequenos em camadas selecionadas;
- treino do enxerto com DRM-SAINT-G-Phi;
- checkpoint separado por enxerto;
- validacao antes/depois no mesmo corpus;
- criterio para consolidar ou descartar enxertos;
- comparacao contra adicionar parametros e treinar full no mesmo budget.

## Marcos

### Marco 1 - Enxerto Simulado

Criar um experimento pequeno, dependency-free ou PyTorch simples, em que um DRM
reduzido recebe um modulo novo e treina apenas o enxerto.

### Marco 2 - Enxerto Real no DRM Transformer

Integrar o mecanismo ao `drm_transformer`, congelando o nucleo e treinando um
enxerto em uma ou mais matrizes reais.

### Marco 3 - Consolidacao

Testar se o enxerto pode ser consolidado no checkpoint sem perder a melhoria de
validacao.

### Marco 4 - Crescimento Progressivo

Repetir o ciclo mais de uma vez:

```text
DRM-1 -> DRM-1+G1 -> DRM-1+G1+G2
```

O objetivo e medir se multiplos enxertos acumulam capacidade ou entram em
conflito.

## Metricas

- validation loss antes/depois;
- ganho por parametro treinavel;
- tamanho do checkpoint do enxerto;
- memoria CUDA por etapa;
- tempo de roteamento;
- tempo de treino;
- regressao em exemplos antigos;
- taxa de enxertos aprovados versus descartados.

## Criterio de Conclusao

A fase passa se pelo menos um ciclo completo demonstrar:

- melhoria de validacao contra base congelada;
- checkpoint recomponivel do enxerto;
- consolidacao sem regressao clara;
- ganho por parametro competitivo contra baseline full no mesmo budget;
- memoria controlada no hardware alvo.

## Relacao com Fase 16

Fase 16 deve escalar a estrategia que sair daqui.

Se DRM-SAINT-G funcionar, a escala 70B nao deve ser tratada apenas como adaptacao
de pesos existentes. Ela deve ser tratada como crescimento controlado por
enxertos, com o nucleo congelado e capacidade nova adicionada em partes.
