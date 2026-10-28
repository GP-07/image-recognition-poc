# Plan de implementación — Menu dish extraction POC

Basado en las decisiones de `decision_log.md` (Discusiones 1–6).

## Entorno detectado (2026-10-28)

- macOS, Apple M1 Pro, Python 3.10.11, pip (sin `uv`), git disponible (la carpeta aún no es repo).
- Sin `ANTHROPIC_API_KEY` en el entorno → el modo `ocr+llm` necesita que el usuario la exporte.
  Mientras tanto todo se desarrolla y testea con un LLM falso y respuestas grabadas.

## Cambio técnico propuesto: RapidOCR en lugar de PaddleOCR

RapidOCR corre **los mismos modelos PP-OCR** de PaddleOCR pero sobre ONNX Runtime:
`pip install rapidocr onnxruntime` (decenas de MB) vs. `paddlepaddle` (cientos de MB, instalación
más frágil en macOS arm64). Misma calidad, mucho más liviano → encaja con "POC liviano".
El motor queda detrás de una interfaz `OcrEngine`, así que cambiarlo es una clase.

## Estructura

```
image_recognition_poc/
  pyproject.toml            # deps + ruff + pytest
  README.md                 # cómo correr + respuestas a las 3 preguntas + resultados
  config/default.yaml       # configuración por defecto
  prompts/structure_v1.md   # prompt versionado del LLM
  data/golden/*.yaml        # golden set (un archivo por imagen, platos con alias)
  src/menu_poc/
    models.py               # contratos de datos (pydantic)
    interfaces.py           # Protocols de cada etapa
    config.py               # RunConfig (modo, política LLM, modelo, rutas)
    io/          dataset.py (zip/carpeta → imágenes), writers.py (JSON/CSV/Markdown)
    cache/       store.py   (artefactos con clave de contenido)
    preprocess/  image.py   (EXIF, resize, rotación de página)
    ocr/         rapid.py   (adaptador RapidOCR)
    parsing/     layout.py (columnas/filas), rules.py (parser anclado en precios), confidence.py
    llm/         client.py (protocolo + Anthropic + fake), structurer.py, grounding.py, pricing.py
    pipeline.py             # orquesta etapas según RunConfig
    evaluation/  golden.py, matching.py, metrics.py, report.py
    cli.py                  # `menu-poc run | eval | compare`
  tests/
    unit/  integration/  fixtures/
  artifacts/  runs/         # generados (gitignored)
```

## Fases

### F0 — Setup
venv + `pyproject.toml`, ruff, pytest, `git init`, `.gitignore`.
**Listo cuando:** `pytest` corre (vacío) y `ruff check` pasa.

### F1 — Estructura principal y contratos
- `models.py`: `OcrLine(text, bbox, confidence)`, `OcrResult`, `Dish(name, description, prices,
  section, source_line_ids, grounded)`, `ImageResult(image_id, dishes, confidence, llm_used,
  cost_usd, latency_s, tokens)`, `RunManifest`.
- `interfaces.py`: `Preprocessor`, `OcrEngine`, `Parser`, `Structurer`, `Verifier`, `LLMClient`.
- `config.py`: `mode: ocr | ocr+llm`, `llm_policy: fallback | always`, `llm_model` (uno por corrida),
  `confidence_threshold`, `only_low_confidence`, `reuse_from_run`.
- `cache/store.py`: clave = hash(entrada) + config de la etapa + versión; guarda JSON.
- `io/`: lectura del zip o carpeta (id restaurante/imagen), escritura de resultados.
**Listo cuando:** contratos con tests unitarios de serialización y claves de caché.

### F2 — Estrategias (sin LLM)
- Preprocesado: EXIF, resize a ~2000 px de lado largo, detección de página rotada 90°/270°
  (heurística por orientación de las cajas del OCR o reintento barato a baja resolución).
- OCR: adaptador RapidOCR → `OcrResult`, cacheado.
- Layout: agrupar líneas en columnas (clustering en x) y filas (solape en y), orden de lectura.
- Parser por reglas: señales de precio, numeración, líneas de puntos, tipografía → puntaje por línea →
  platos; filtros de ruido (teléfonos, horarios, alérgenos, URLs).
- Confianza: % de precios asociados a un plato, cobertura y cantidad de platos.
**Listo cuando:** `menu-poc run --mode ocr` produce JSON por imagen; tests unitarios con líneas OCR sintéticas.

### F3 — Conector LLM y configuración
- `LLMClient`: `AnthropicClient` (SDK oficial, structured outputs con JSON Schema, Haiku 5.5 por defecto,
  effort `low`) y `FakeLLMClient` (respuestas grabadas, para tests y modo offline).
- `structurer.py`: arma el prompt con el texto del OCR (con ids de línea y orden de lectura) → `Dish[]`.
- `grounding.py`: cada plato debe coincidir (rapidfuzz) con líneas del OCR; si no, se descarta y se cuenta.
- `pricing.py`: costo a partir de `usage` por modelo.
- Política `fallback` / `always`; un solo modelo por corrida; cache por hash(OCR) + modelo + versión de prompt.
**Listo cuando:** el pipeline corre con `FakeLLMClient`; con key real, una imagen de punta a punta.

### F4 — Pipeline y script de ejecución
- `pipeline.py` y CLI:
  - `menu-poc run --input <zip|dir> --mode ocr+llm --llm-model claude-haiku-5-5 --llm-policy fallback`
  - `menu-poc run ... --reuse-from runs/<id> --llm-model claude-sonnet-5-5 --only-low-confidence`
  - `menu-poc eval --run runs/<id>` y `menu-poc compare runs/a runs/b ...` → tabla Markdown.
- Cada corrida guarda `config.yaml`, `results.jsonl`, `metrics.json`, `summary.md`.

### F5 — Evaluación y golden set
- Formato: un YAML por imagen con platos y alias de idioma; split **dev** (para ajustar reglas y prompt)
  y **test** (sólo para medir) — evita sobreajustar a 19 imágenes.
- Matching normalizado (minúsculas, acentos, numeración) + rapidfuzz con umbral.
- Métricas: recall (tasa de reconocimiento), precision de control, % escalado al LLM, costo y latencia por imagen.
- Etiquetado: **borrador generado con IA a partir de las imágenes y revisión humana obligatoria**
  (riesgo de sesgo: el que etiqueta y el evaluado son de la misma familia de modelos → lo corrige el usuario).

### F6 — Tests
- Unitarios por módulo (dentro de cada fase, por quien escribe el módulo).
- Integración: pipeline completo con OCR falso + LLM falso; uno con RapidOCR real sobre una imagen
  (marcado `slow`); ninguno llama a la red.

### F7 — Revisión e iteración
- Revisión de código con contexto limpio (correctitud, simplicidad, consistencia de contratos).
- Análisis de errores **sobre el split dev**: platos no reconocidos y falsos positivos por imagen →
  ajustar reglas del parser, umbral de confianza y prompt (no se entrena nada). Medir en test.
- Máximo 2 iteraciones; cada cambio queda en `decision_log.md`.

### F8 — Documentación
README con instalación, uso, tabla comparativa de corridas y respuestas a: técnica, evaluación,
limitaciones y mejoras en producción. Actualizar la tabla de herramientas de IA.

## Orden y dependencias

```
F0 → F1 ─┬─ F2 (estrategias) ─┐
         ├─ F3 (LLM)          ├─ F4 (pipeline + CLI + integración) → F7 (revisión/iteración) → F8
         └─ F5 (eval + golden)┘
```
F2, F3 y F5 sólo dependen de los contratos de F1 y tocan archivos distintos → se pueden paralelizar.

## Organización de agentes

| Paso | Quién | Por qué |
|---|---|---|
| Plan | Yo (sesión principal) | Ya tengo todo el contexto de las 6 discusiones; un agente nuevo lo reconstruiría desde cero. |
| F0 + F1 | Yo, secuencial | Los contratos son la columna vertebral: los escribe una sola mano para que sean coherentes. |
| F2, F3, F5 | 3 subagentes en paralelo, cada uno con sus tests unitarios | Independientes una vez fijados los contratos; cada uno es dueño de sus archivos. |
| Borrador del golden set | Subagente con visión, en paralelo | Trabajo tedioso y paralelizable; el usuario lo revisa. |
| F4 + integración | Yo | Junta todas las piezas; requiere el contexto completo. |
| Revisión de código | Subagente con contexto limpio | Ojos nuevos detectan lo que el autor no ve. |
| Análisis de errores | Subagente, sobre dev | Propone ajustes de reglas/prompt/umbral; yo decido y mido en test. |
