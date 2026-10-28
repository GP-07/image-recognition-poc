# Decision log — Task 4: Image Processing POC (extracción de platos de cartas)

Registro de análisis, decisiones y herramientas (incluidas las de IA) usadas durante el POC.

## Enunciado

Diseñar un sistema que extraiga platos de imágenes de cartas (`google_images.zip`).
POC local y liviano: sin GPU, sin entrenar modelos, no se espera alta precisión.
Importa: arquitectura, selección de modelos, estrategia de evaluación, implementación limpia.

## Dataset observado (2026-10-27)

- 19 fotos JPG en 10 carpetas (una carpeta = un restaurante, id tipo Google Place; 2 fotos por carpeta, una tiene 1).
- Resoluciones entre 1385x620 y 5712x4284.
- Son **fotos de celular, no escaneos**: perspectiva, papel doblado (trípticos), reflejos en fundas
  plásticas, dedos, fotos tomadas desde la calle (pizarras/atriles), varias rotadas 90°.
- **Diseño complejo**: 2–4 columnas, precios en columnas separadas (tamaños de pizza), tipografías
  decorativas en títulos de sección, fotos de platos intercaladas, iconos de alérgenos.
- **Multilingüe**: el mismo plato en catalán / castellano / inglés (carta india, burgers).
- Tipos de cocina: india, pizza, sushi, libanesa, burgers/sándwiches, etc.
- **Los formatos no se repiten entre restaurantes**, y dentro de un mismo restaurante las dos fotos
  suelen ser tomas distintas (otra página, otra distancia, atril vs. primer plano).

## Discusión 1 — ¿OCR, LLM u otra cosa?

Postura inicial del usuario: OCR, porque no hay manuscrito ni errores ortográficos, y el OCR es
determinista y rápido.

Análisis:
- El OCR es una pieza muy buena (rápido, determinista, barato, no alucina), pero:
  1. "Sin errores ortográficos" ≠ "OCR sin errores": los errores vienen de la foto (ángulo, brillo,
     tipografía), no del texto. Tesseract sufrirá con estas imágenes.
  2. El problema principal no es leer sino **estructurar**: qué línea es nombre, descripción, precio,
     sección o traducción, y el orden de lectura en multi-columna.

Alternativas evaluadas:

| Enfoque | Ejemplos | Cuándo |
|---|---|---|
| OCR clásico + reglas | Tesseract + regex anclada en precios | Escaneos / plantillas fijas, sin red. Baseline. |
| OCR deep-learning + layout | PaddleOCR, docTR, Surya, Apple Vision (ocrmac) | Fotos reales, texto rotado, bboxes. CPU. |
| OCR + LLM de texto (estructurador) | OCR → LLM → JSON con esquema | Layouts variados; salida verificable contra el OCR. |
| VLM end-to-end | Claude / Gemini / GPT; local Qwen2.5-VL, MiniCPM-V | Máxima calidad, mínimo código; costo, latencia, alucinación. |
| Modelos OCR-free / fine-tune | Donut, Pix2Struct, LayoutLMv3 | Producción con datos etiquetados. Fuera de alcance (no entrenar). |
| Document AI gestionado | Google Document AI, AWS Textract, Azure DI | Producción rápida, lock-in. |
| Clasificador liviano sobre líneas OCR | Gradient boosting con features geométricas | Con unos cientos de líneas etiquetadas. |

Propuesta inicial: pipeline modular con estrategias intercambiables (reglas / LLM / VLM),
grounding contra el OCR y evaluación comparativa.

## Discusión 2 — Costo de estrategias intercambiables y agrupación por formato

Preocupaciones del usuario:
- Tener varias estrategias intercambiables parece costoso.
- Idea: comparar todas las imágenes (normalizadas en posición) para agrupar las que comparten formato.
- Recordatorio: se aceptan fallos; nada de entrenar; liviano y rápido; lo importante es la arquitectura
  (qué componente y cuándo).

Conclusiones:
- Las estrategias múltiples **no corren todas en producción**: se comparan sólo en la evaluación del POC.
  En ejecución se usa una **cascada**: lo barato siempre, lo caro sólo si la confianza es baja.
- Agrupar por formato / plantillas **no sirve para generalizar en este dataset**: ~10 layouts distintos
  sobre 19 fotos; cada "plantilla" tendría ≤2 ejemplos y habría que escribirla a mano.
  Aporte esperado sobre cartas nuevas ≈ 0%. Además la homografía sólo endereza superficies planas
  (no trípticos doblados ni fotos de atriles).
- Sí tiene valor como **caché / deduplicación** en producción (misma carta re-subida, cadenas con
  carta común): huella perceptual + huella de layout → reutilizar extracción previa.

Estimaciones (a validar con ground truth, recall de nombres de plato):
- OCR + reglas: ~35–55% global (mejor en una columna, peor en trípticos / multi-columna).
- OCR + LLM de texto (con grounding): ~70–85%.
- VLM end-to-end: ~85–95%, con mayor costo y riesgo de alucinación.
- Plantillas por formato sobre cartas nuevas: ~0%.

## Discusión 3 — Respuestas a las preguntas del enunciado (borrador del usuario + correcciones)

### Técnica
- Cascada por confianza: OCR → parser de reglas → confianza → (si baja) **LLM de texto** sobre el
  texto del OCR, validado contra el OCR.
- Corrección: el LLM procesa el **texto** del OCR, no la imagen (si procesa la imagen es un VLM).
- Corrección: el VLM no se descarta por costo (con un modelo chico ronda los milicentavos por imagen)
  sino porque su salida no se puede verificar contra nada y es menos determinista. Queda como último
  escalón opcional cuando el propio OCR falla (confianza de OCR baja), y para pre-etiquetar el golden set.

### Evaluación
- Base: **golden set etiquetado a mano** (6–8 imágenes) + métricas deterministas: precision / recall / F1
  de nombres (fuzzy match), exactitud de precio y sección, tasa de alucinación (ítems no presentes en
  el OCR), % de imágenes escaladas al LLM, latencia y costo por imagen.
- Un modelo fuerte puede **pre-etiquetar** el golden set (un humano corrige).
- LLM-as-judge sólo como complemento (equivalencias semánticas, traducciones) y para muestreo en
  producción sin etiquetas, calibrado contra humanos. No hace falta un agente: es una llamada simple.

### Modelo LLM elegido
- **Claude Haiku 5.5** (`claude-haiku-5-5`, $0.10 / $0.50 por MTok in/out) para estructurar: tarea de
  extracción simple, structured outputs con JSON Schema, effort `low`. ~4k tokens in + ~3k out ≈
  US$0.002 por imagen; las 19 imágenes ≈ US$0.04.
- Escalado a **Claude Sonnet 5.5** (`claude-sonnet-5-5`, $2 / $10) sólo si falla la validación (≈ US$0.04/imagen).
- Producción: Batch API (−50%) para ingesta no interactiva; prompt caching del system prompt + schema.
- Desde 2026-10-21 la API no acepta `temperature`: el determinismo se logra con schema estricto,
  grounding contra el OCR y caché de respuestas en disco (re-ejecución reproducible sin API key).
- Alternativa local sin costo: modelo abierto vía Ollama (menor calidad, más lento en CPU).

## Discusión 4 — Caché, normalización y estrategias configurables

### Caché (desde el inicio)
- Caché por contenido: clave = sha256 de la imagen + etapa + config (motor OCR, modelo, versión de prompt).
  Costo ~0; hace las re-ejecuciones instantáneas, reproducibles y sin API key.
- Huella perceptual para cartas "casi iguales": sólo útil en producción; en este dataset no hay duplicados.

### Normalización contra catálogo (desde el inicio, acotada)
- Barata en cómputo con embeddings multilingües chicos en CPU (evitar arrastrar torch: preferir
  fastembed/ONNX o model2vec).
- Problema real: no hay catálogo. En el POC: (1) deduplicar traducciones dentro de una misma carta,
  (2) mapear a un catálogo semilla chico hecho a mano, con umbral; siempre conservar el nombre original.
- Riesgo: fusiones falsas ("Pollo Tikka" vs "Pollo Tikka Masala") → umbral conservador + evaluar aparte.

### Estrategias habilitables
- Modos: `ocr` (sólo reglas) y `ocr+llm`, con política de LLM `fallback` (sólo si confianza baja) o
  `always` (techo de calidad del LLM); modelo LLM configurable. "Sólo LLM" no aplica: su entrada es el OCR.
- Interfaces (Protocol) por etapa + registro + config (CLI/YAML). La evaluación corre una matriz chica
  de configuraciones y produce una tabla comparativa.

## Discusión 5 — Alcance: reconocimiento y costo, sin normalización

Aclaración del usuario: lo que importa es **reconocer los platos de cada menú**; no importa que el mismo
plato quede con nombres distintos entre menús. No se pide normalizar. Métricas que importan:
**tasa de reconocimiento** y **costo de procesamiento**.

Decisiones (reemplazan la parte de normalización de la Discusión 4):
- **Fuera de alcance:** catálogo, embeddings y normalización entre menús → menos dependencias, menos costo.
- **Se mantiene el caché** (reduce costo y hace reproducible la evaluación).
- **Traducciones dentro de una carta:** no se deduplican en la salida. En el golden set cada plato se
  etiqueta una vez con sus variantes de idioma como alias: acertar cualquier alias = reconocido, y las
  traducciones extra no cuentan como falsos positivos.
- **Métricas principales:** recall (tasa de reconocimiento) y costo/latencia por imagen. Precision como
  control (que no se cuelen títulos, teléfonos, alérgenos). Precio: se extrae porque ayuda al parser,
  pero es métrica secundaria.

## Discusión 6 — Rol del precio, un modelo por corrida y reutilización del OCR

### Por qué el precio en el parser por reglas
- El OCR devuelve líneas sin semántica; las reglas necesitan señales baratas para decidir qué es un plato.
- El patrón de precio (`\d+[.,]\d{1,2}\s?€`) es la señal más fuerte: casi todo plato tiene uno y casi
  nada más lo tiene (teléfonos/fechas se excluyen por formato).
- Uso: línea con precio (o alineada en altura con un precio de otra columna) = plato; líneas siguientes
  sin precio = descripción; líneas grandes / mayúsculas sin precio = título de sección.
- No es la única señal: numeración ("46. LAMB CURRY"), líneas de puntos, tipografía (alto del bbox,
  mayúsculas). Se combinan en un score.
- Falla en: precios por sección o menú del día, tablas de tamaños separadas del nombre (pizzas),
  precios bajo fotos (sushi). Esos casos bajan la confianza y disparan el LLM.
- El "% de precios asociados a un plato" es además el indicador de confianza.

### Un solo modelo por corrida
- El modelo LLM es configurable, pero una corrida usa uno solo (se descarta el escalado
  Haiku → Sonnet dentro de la misma corrida). Comparar modelos = comparar corridas.

### Artefactos por etapa y re-ejecución parcial
- Cada etapa persiste su salida con clave de contenido:
  - OCR: hash(imagen) + motor + versión + parámetros + versión de preprocesado.
  - LLM: hash(salida OCR) + modelo + versión de prompt.
- Re-correr con Sonnet reutiliza el OCR guardado y sólo vuelve a llamar al LLM; se puede limitar a las
  imágenes con bajo reconocimiento / baja confianza.
- Determinismo del OCR: mismo motor, versión, parámetros y preprocesado → mismo resultado en la práctica.
  Puede variar al cambiar versión de librería/pesos, preprocesado o hardware (diferencias numéricas
  mínimas). Por eso esos datos van en la clave del caché; con el artefacto guardado no se depende de ello.
- El LLM no es determinista: por eso también se guarda su salida.
- En producción no hay golden set: "tasa baja" se detecta con proxies (confianza, % de ítems verificados
  contra el OCR, % de precios sin plato).

## Discusión 7 — Implementación (2026-10-28)

- Organización: F0/F1 (contratos) y F4 (pipeline/CLI) en la sesión principal; F2 (OCR + reglas),
  F3 (LLM + grounding), F5 (evaluación) y el borrador del golden set con 4 subagentes en paralelo,
  cada uno dueño de sus archivos. Revisión: un subagente de código (sólo lectura) y uno de análisis
  de errores del clasificador por reglas (ajusta sólo sobre dev).
- OCR: RapidOCR (modelos PP-OCR sobre ONNX Runtime) en lugar de PaddleOCR, por peso de instalación.
  Detecta páginas giradas 90° (re-OCR a 90/270) e inclinación leve (deskew).
- Bug encontrado por los tests de integración: imágenes con bytes idénticos compartían la entrada de
  caché del OCR con el `image_id` de la primera. Corregido (se re-etiqueta al leer de caché).
- Costo y tiempos reportados = los de producir el resultado, aunque venga de caché (corridas comparables);
  `llm_usage.cached` indica si hubo llamada real.
- Golden set: 19 YAML, 620 platos, split por restaurante (dev 10 imágenes / test 9), redactado por IA
  y pendiente de revisión humana. El dataset trae fotos duplicadas (idénticas o a media resolución)
  en 4 restaurantes: esas cartas pesan doble en las métricas.
- Primera corrida real, sólo OCR + reglas: recall micro 71,9% (dev 72,6% / test 70,1%), precision 72,4%,
  ~4,8 s por imagen en CPU (M1), costo $0. Superior a la estimación previa (35–55%).
- Hallazgo: la confianza de las reglas no está bien calibrada en algunos casos (p. ej. una página con
  recall 13,6% obtuvo 0,92) → se manda a calibrar al agente de análisis de errores (sólo con dev).
- Sin `ANTHROPIC_API_KEY` en el entorno: el LLM se probó con respuestas grabadas; las corridas reales
  con Haiku/Sonnet requieren que el usuario exporte la key.

## Discusión 8 — Revisión e iteración (2026-10-28)

### Revisión de código (agente de contexto limpio, sólo lectura)
Corregidos con tests de regresión: normalización que borraba alfabetos no latinos (grounding
descartaba esos platos); matching goloso → asignación óptima (húngaro, agrega `scipy`); predicciones
repetidas que sumaban TP; ids de imagen inconsistentes/duplicados; llamadas LLM fallidas pero cobradas
registradas como $0; `effort`/`max_tokens` fuera de la clave del caché; verificador fuera del manejo
de errores; señales de confianza de las reglas perdidas al usar LLM; re-corridas parciales que no
cubrían el mismo set de imágenes (ahora se arrastran los resultados previos); validaciones de config.
Pendientes menores (documentados): rutas relativas al directorio actual, versiones del OCR a mano,
dos funciones de normalización.

### Análisis de errores del clasificador por reglas (sólo dev, test medido una vez al final)
- Taxonomía en dev: bebidas, nombres cortados por el borde de la foto, fragmentos de descripción,
  códigos de alérgenos, notas de porción; misses por tableros multicolumna, grillas de precios
  compartidas (pizzas), perspectiva, menús sin precio.
- Cambios generalizables: número decimal suelto = precio; línea que empieza con "(" = nota; códigos
  de dieta/alérgenos = ruido; viñetas como numeración; descripciones en minúscula con precio;
  bebidas descartadas (política del golden, a confirmar); confianza con `recall_proxy`
  (precios sin plato), `name_gate` (nombres con forma de descripción) y `header_gate`.
- Resultado: dev recall 72,6% → 79,4%, precision 75,0% → 85,3%; test sin cambios (70,1%) → sin
  sobreajuste. Total 71,9% → 76,8% recall, 72,4% → 79,5% precision.
- Enrutamiento: ninguna página con recall < 60% queda fuera del LLM (antes 1 en test con conf 0,92).
  8/19 páginas (42%) irían al LLM con umbral 0,6. La confianza sigue más a F1/precision que a recall.

## Discusión 9 — Duplicados, LLM gratuito, I/O, .env y alcance completo (2026-10-28)

- **Duplicados tempranos:** ningún par duplicado del dataset compartía bytes (cambia la metadata),
  así que la caché por bytes no los detectaba. Nuevo paso antes del OCR: hash del archivo + dHash
  perceptual de 256 bits. En el dataset los duplicados quedan a 0–4 bits y las fotos distintas a
  ≥100 → umbral 10. Se conserva la imagen más grande (mejor OCR); el duplicado recibe el resultado
  del original con `duplicate_of` y no se puntúa (cada carta cuenta una vez). Costo ~0,1 s/imagen.
- **LLM gratuito:** `OllamaClient` (modelos locales, ej. `ollama/qwen2.5:7b`), sólo librería
  estándar, salida con JSON Schema, temperatura 0, costo $0 (se paga en latencia). No cambia el
  estructurador ni el pipeline: es otra implementación de `LLMClient` elegida por prefijo de modelo.
- **Escalado:** sólo con `--mode ocr+llm`; `fallback` = páginas con confianza < umbral (0,6,
  configurable), `always` = todas. En `ocr` nunca se llama al LLM.
- **I/O:** `--input` (zip o carpeta), `--output` (corridas) y `--artifacts` (caché) por CLI.
  Cada corrida escribe `results.jsonl`, `manifest.json` (warnings + errores), `config.yaml`
  efectiva y, si hay golden, `summary.md`/`metrics.json`.
- **.env y degradación:** `ANTHROPIC_API_KEY` desde `.env` (python-dotenv; `.env` en .gitignore,
  `.env.example` versionado). Sin key (o sin Ollama) una corrida `ocr+llm` sigue como `ocr`,
  exit 0, warning en consola y en `manifest.json`. Base para un job de GitHub Actions con la key
  como secret (bosquejado en el README).
- **Alcance:** pasa a ser **todo ítem pedible** (platos, postres, bebidas, extras/salsas con
  entrada propia). Se quitó el filtro de bebidas de las reglas (parser v3), prompt `structure_v2`,
  y el golden set se re-etiquetó con la nueva política.
- **Golden set (aclaración):** es la "hoja de respuestas" para medir; el pipeline nunca lo ve.
  Sólo influyó indirectamente al ajustar reglas a mano con el split dev; test nunca.

## Discusión 10 — Aliases, validación temprana, salida por defecto y CSV (2026-10-28)

- `--llm-model` acepta aliases sin distinguir mayúsculas: `haiku`, `sonnet`, `opus`, `ollama`
  (`ollama` = `qwen2.5:7b` por defecto, sin variable de entorno); los ids exactos siguen valiendo.
  Un nombre desconocido falla al cargar la config, con la lista de aliases.
- La conexión LLM se valida **antes de leer imágenes**: con key presente se hace `models.retrieve`
  (gratis) para probar que es válida y puede usar el modelo; si no, warning inmediato y la corrida
  sigue como `ocr`. Para Ollama: servidor alcanzable y modelo descargado.
- Sin `--output`, la salida va a `tmp/` del proyecto (gitignored). Las rutas del YAML son relativas
  a la raíz del proyecto y las de la CLI al directorio actual: el comando funciona desde cualquier lado.
- Salida principal: `items.csv` (una fila por ítem, UTF-8 con BOM para Excel). Se mantiene
  `results.jsonl` como registro completo anidado (señales, costo, tiempos), que usan `eval` y `--reuse-from`.
- GitHub Actions: pospuesto.
- Golden set: el usuario plantea borrarlo para no sumar trabajo manual; recomendación: conservarlo
  como "silver set" generado por IA sin revisión obligatoria (ver respuesta en la conversación). Pendiente de decisión.

## Discusión 11 — Selección del motor LLM por subclases (2026-10-28)

- Pedido del usuario: menos código de "preparación" y una selección tipo `LLMEngine.get_for(nombre)`.
- Implementado como factory method con subclases auto-registradas (cercano a chain of
  responsibility; no es el patrón State): clase abstracta `LLMEngine` con `get_for`/`resolve`;
  cada subclase (`AnthropicEngine`, `OllamaEngine`, `FakeEngine`) declara sus `aliases` y un
  `handles(nombre)`. `get_for` recorre las subclases y devuelve una instancia de la que lo maneja.
- Se eliminan `registry.py` y `make_llm_client`. Agregar un proveedor = agregar una subclase.
- Luego `get_for` + `resolve` se unificaron en `LLMEngine.get_for_alias(nombre)`: la config valida
  y resuelve el modelo con `get_for_alias(valor).model` (crear un motor no abre conexiones).
- Archivos: `llm/engine.py` (base, errores, `prompt_key`), `anthropic_engine.py`,
  `ollama_engine.py`, `fake_engine.py`; `llm/__init__.py` importa los motores para registrarlos.

## Discusión 12 — Pipeline como Chain of Responsibility + Builder (2026-10-28)

- Problema: la clase `Pipeline` concentraba OCR, reglas, política LLM, caché, verificación y
  duplicados; difícil de leer y de testear por partes.
- Cada paso es un `Step` en su módulo (`steps/dedup.py`, `ocr.py`, `rules.py`, `llm.py`,
  `verify.py`) con `process(ctx) -> bool` (¿sigue la cadena?) y un link `next`. El estado de cada
  imagen viaja en `ImageContext`. Una excepción corta la cadena sólo para esa imagen y conserva lo
  ya calculado (p. ej. si falla el LLM queda la salida de reglas).
- `PipelineBuilder`: `add()` en cualquier orden; `build()` enlaza en el orden fijo del sistema
  (dedup → ocr → rules → llm → verify), agrega `verify` siempre que haya `llm` y rechaza cadenas
  inválidas (LLM sin OCR/reglas, sin OCR). `from_config` arma las piezas según la configuración.
- `Pipeline` quedó como alimentador delgado: ordena por tamaño (para que el original de un grupo
  de duplicados sea la foto más grande), pasa cada imagen por la cadena y guarda resultados.
- Se eliminó `dedup.py` como paso por lotes: ahora es el primer step y guarda los originales vistos.
- Verificado: misma salida y mismas métricas que antes del refactor; 14 tests nuevos por step y
  del builder (210 en total).

## Discusión 13 — Sin `--mode`: OCR + reglas siempre, LLM opcional (2026-10-28)

- `ocr` vs `ocr+llm` era un detalle de implementación expuesto al usuario. Ahora OCR + reglas son
  el núcleo y corren siempre; el paso LLM se agrega sólo si hay `llm_model` (`--llm-model` o YAML).
- Sin modelo configurado: warning en consola y en `manifest.json` diciendo que se usa sólo OCR +
  reglas y cómo habilitar el LLM (alias + `ANTHROPIC_API_KEY`, u `ollama`). Con modelo pero sin key
  válida: igual que antes, degrada con warning (`requested_llm_model` vs `llm_model` en el manifest).
- OCR: el motor (modelos) se carga y verifica al inicio (`ensure_ready` del step); si no puede
  correr, la corrida termina con error (exit 3) antes de procesar imágenes. Un fallo de OCR en una
  imagen puntual (archivo corrupto) sigue aislado: queda como error de esa imagen y la corrida sigue.
- Reportes: el modo se deriva de `llm_model`; las corridas viejas conservan su `mode` guardado.

## Discusión 14 — Estilo: archivos por categoría, sin asserts, sin nombres de una letra (2026-10-28)

- Pedido del usuario: nada de `assert` en código productivo, nada de variables de una sola letra, y
  separar en archivos los modelos, dataclasses, typings y constantes (como `models.py` /
  `interfaces.py` del primer día).
- Cuatro módulos en la raíz del paquete, con dependencias en una sola dirección
  (`constants` ← `typings` ← `models` ← `data_classes`):
  - `constants.py`: todas las constantes (umbrales, versiones de caché, regex, precios por modelo,
    schema JSON del LLM, rutas por defecto), agrupadas por componente;
  - `typings.py`: aliases de tipos (`BBox`, `DishOrigin`, `LLMPolicy`, `Split`, `Candidate`...) y
    TypeVars; los Protocols siguen en `interfaces.py`;
  - `models.py`: todos los modelos pydantic, incluidos `RunConfig` y los de evaluación;
  - `data_classes.py` (no `dataclasses.py`, para no tapar el módulo estándar): `ImageRef`,
    `ImageContext`, `ItemRow`, `OcrItem`, `LineInfo`, `DishDraft`, `PriceMatch`, `ParseStats`,
    `ModelPrice`.
- `RunConfig` resuelve el alias del modelo con `LLMEngine`, que a su vez importa `models`: ese import
  va dentro del validador para no formar un ciclo.
- Asserts: los pasos leen la salida de pasos anteriores con `require(valor, "qué")`, que lanza
  `MissingInputError`. El builder garantiza el orden, así que sería un bug; igual que cualquier
  error de un paso, falla sólo esa imagen.
- Se mantiene `_` como descarte convencional (no es una variable que se lea).
- Verificado: `items.csv` byte a byte idéntico al de antes del refactor, mismas métricas, 212 tests.

## Discusión 15 — El comando corre siempre entero; validación como fase del pipeline (2026-10-28)

- El CLI tenía demasiadas opciones (`--config`, `--artifacts`, `--llm-policy`, `--offline`,
  `--threshold`, `--image-ids`, `--reuse-from`, `--only-low-confidence`, `--golden`). Quedan
  `--input`, `--output` y `--llm-model`; el resto vive en `config/default.yaml`.
- Sin corridas parciales: se eliminan `image_ids`, `reuse_from`, `only_low_confidence`,
  `select_images` y `merge_previous`. Re-correr con otro modelo sigue siendo barato porque el OCR
  y las respuestas del LLM salen de la caché por contenido.
- El archivo de config es del sistema: se asume que existe (sin chequeo ni flag).
- Las validaciones salen de `cli.py` y pasan al pipeline como una fase `validate_inputs()`, una
  sola vez antes de la cadena. No es un `Step` más porque los pasos corren por imagen, y "no hay
  imágenes" o "la API key no sirve" son condiciones de la corrida. Dentro de esa fase cada paso
  valida lo suyo (`Step.validate()`, que reemplaza a `ensure_ready`):
  - el OCR carga sus modelos; si no puede, `InputValidationError` y la corrida no arranca;
  - el LLM comprueba que puede llamar al modelo; si no, se desactiva y devuelve un warning;
  - el pipeline verifica que el input tenga imágenes y avisa si no hay LLM configurado.
- Exit codes: 0 cuando corre (aunque haya warnings o errores por imagen), 2 si no puede arrancar.
- Verificado: `items.csv` idéntico al anterior; con `--llm-model haiku` y sin key, la corrida usa
  OCR + reglas y el manifest guarda el modelo pedido y `llm_model: null`.
- Seguimiento (pedido del usuario): `Pipeline.run(refs, config)` crea el directorio de la corrida
  con la config efectiva (sin modelo LLM si su paso quedó desactivado) y devuelve un `RunOutcome`
  (`run_dir`, `config` efectiva, `results`). El CLI ya no decide qué config registrar.

## Herramientas de IA usadas durante el POC

| Fecha | Herramienta | Uso |
|---|---|---|
| 2026-10-27 | Claude Code (Claude Opus 5.5) | Inspección del dataset, análisis de alternativas, diseño de arquitectura, este log. |
| 2026-10-27 | Claude Haiku 5.5 / Sonnet 5.5 (elegidos, aún no usados) | Estructurador de texto OCR → JSON; un modelo por corrida (Haiku por defecto, Sonnet en una re-corrida). |
| 2026-10-28 | Claude Code (Opus 5.5), sesión principal | Plan de implementación, contratos, pipeline, CLI, integración, tests de integración. |
| 2026-10-28 | Subagentes de Claude Code (Opus 5.5) ×4 en paralelo | F2 OCR + reglas, F3 conector LLM + grounding, F5 evaluación, borrador del golden set mirando las imágenes (visión). |
| 2026-10-28 | Subagentes de Claude Code (Opus 5.5) ×2 | Revisión de código (sólo lectura) y análisis de errores / calibración del clasificador por reglas. |
| 2026-10-28 | RapidOCR 3.10 (modelos PP-OCRv6 det/rec small, ONNX Runtime) | Motor OCR local en CPU (modelos de deep learning preentrenados, sin entrenamiento propio). |
| 2026-10-28 | Ollama (modelos abiertos locales, ej. Qwen2.5 7B) — soportado, no ejecutado | Alternativa gratuita y local al LLM de Claude (no instalado en esta máquina). |
| 2026-10-28 | Subagente de Claude Code (Opus 5.5, visión) | Re-etiquetado del golden set con la nueva política (todos los ítems pedibles). |
| 2026-10-28 | Claude Code (Opus 5.5), sesión principal | Refactor de estilo: constantes, typings, modelos y dataclasses en archivos propios; sin asserts ni variables de una letra. |
| 2026-10-28 | Claude Code (Opus 5.5), sesión principal | Simplificación del CLI (corrida siempre completa) y validación de inputs como fase del pipeline. |
