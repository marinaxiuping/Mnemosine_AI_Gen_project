# Mnemósine — Asistente experta en mitología griega

Agente conversacional experto en **mitología griega**, construido con **RAG** (Retrieval-Augmented Generation), **Google Gemini**, **ChromaDB** y **LangGraph**, con memoria de conversación.

**App pública:** [mnemosine.streamlit.app](https://mnemosine.streamlit.app/)

> *Mnemósine* es la titánide de la memoria y la madre de las nueve Musas. El nombre encaja con un agente cuyo valor está en **recordar**: sus fuentes y la conversación.

---

## 1. Dominio elegido

**Mitología griega**: cosmogonía y titanes, dioses olímpicos, héroes (Heracles, Perseo, Teseo, Jasón, Orfeo), criaturas (Medusa, el Minotauro, Polifemo) y el ciclo troyano.

**¿Por qué este dominio?**

- **RAG aporta un valor real.** Cada mito tiene varias versiones según el autor (Hesíodo, Homero, Apolodoro, Ovidio…). Un LLM por sí solo tiende a mezclarlas o a inventar genealogías con total seguridad. Anclar cada respuesta a fuentes concretas, y citarlas, resuelve precisamente ese problema.
- **Existen las fuentes originales.** Los poemas de Homero y Hesíodo están disponibles en traducciones de dominio público, así que podemos trabajar con **fuentes primarias** y no solo con resúmenes.
- **Permite preguntas encadenadas**, ideales para demostrar la memoria: «¿Cómo derrotó Perseo a Medusa?» → «¿Y qué nació de su sangre?».

---

## 2. Base de conocimiento: triangulación de fuentes

La base de conocimiento combina **tres tipos de fuente**, como haría un historiador. Hay un documento por tipo:

| Documento | Tipo | Contenido | Idioma | Páginas aprox. |
|---|---|---|---|---|
| `01_fuentes_primarias.txt` | **Fuente primaria** | Hesíodo, *Teogonía* (origen del mundo, Crono, Prometeo, Titanomaquia) · Homero, *Ilíada* I y XXII · *Odisea* IX y XII | EN / ES | ~7 |
| `02_manual_clasico.txt` | **Manual clásico** | Thomas Bulfinch, *The Age of Fable* (1855): Prometeo, Perseo, Heracles, Teseo, Jasón, Orfeo, caída de Troya | EN | ~7 |
| `03_enciclopedia.txt` | **Enciclopedia** | 15 artículos de Wikipedia en español | ES | ~7 |
| **Total** | | **30 fuentes citables**, indexadas en **90 chunks** | | **~20** |

**¿Por qué tres tipos?**
- La **fuente primaria** es la voz de los propios griegos, con sus versiones y contradicciones.
- El **manual** narra los mitos completos y ordenados.
- La **enciclopedia** aporta síntesis y contexto moderno.

Con las tres, el agente puede responder «según Hesíodo…» y contrastar versiones, en lugar de repetir un único resumen.

**¿Por qué hay textos en inglés?** No existe una traducción española de la *Teogonía* que sea de dominio público, ni un manual de mitología en español de dominio público. Los embeddings de Gemini son **multilingües**: comparan significados, no palabras. Una pregunta en español recupera el pasaje de Hesíodo en inglés con una relevancia de 0,75, y el notebook lo verifica. Las respuestas son siempre en español.

| Obra | Traducción | Licencia | Enlace |
|---|---|---|---|
| Homero, *Ilíada* | Luis Segalá y Estalella (1908) | Dominio público | [Gutenberg #57654](https://www.gutenberg.org/ebooks/57654) |
| Homero, *Odisea* | Luis Segalá y Estalella (1910) | Dominio público | [Gutenberg #58221](https://www.gutenberg.org/ebooks/58221) |
| Hesíodo, *Teogonía* | Hugh G. Evelyn-White (1914) | Dominio público | [Gutenberg #348](https://www.gutenberg.org/ebooks/348) |
| Bulfinch, *The Age of Fable* | — (1855) | Dominio público | [Gutenberg #4925](https://www.gutenberg.org/ebooks/4925) |
| Wikipedia en español | — | CC BY-SA 4.0 | URL en cada sección |

### Preprocesado

El preprocesado lo hace `scripts/build_corpus.py`, que es reproducible con `python scripts/build_corpus.py`.

| Paso | Qué hace | Por qué |
|---|---|---|
| Limpieza de Gutenberg | Quita cabecera y licencia, reconstruye los párrafos (vienen cortados a 70 columnas), elimina números de verso, notas, ilustraciones y marcas de cursiva | Todo eso es ruido que empeora los embeddings |
| Ortografía de 1908 | «á», «ó», «é» átonas → «a», «o», «e» | La traducción de Segalá usa la norma antigua; así el texto se parece al de las preguntas |
| **Nombres latinos → griegos** | Segalá y Bulfinch usan los nombres romanos (Júpiter, Minerva, Ulises). Se anota el griego: «Minerva [Atenea]» | Una pregunta por «Atenea» recupera también los pasajes que dicen «Minerva». Es una anotación editorial: el texto original no se altera |
| Limpieza de Wikipedia | Texto plano sin Referencias, Bibliografía, Enlaces externos… | Ruido sin conocimiento del dominio |
| Selección de pasajes | Cada pasaje empieza por el párrafo exacto del episodio (p. ej. «Mi nombre es Nadie») y tiene un presupuesto de palabras | Corpus equilibrado (~7 págs. por tipo) que contiene los momentos clave |
| Cabecera de metadatos | Cada sección lleva `Título`, `Fuente`, `Tipo`, `Idioma` y `Obra` | Se guardan como metadatos de cada chunk: permiten citar, filtrar y mostrar el tipo de fuente |

---

## 3. Arquitectura

```
                      ┌──────────────────── LangGraph (StateGraph + MemorySaver) ─────────────────────┐
 pregunta ──► START ──► rewrite_query ──► retrieve ──────────► generate ──► END ──► respuesta + fuentes
                      │   (Gemini, T=0)     │                   (Gemini, T=0.3)                          │
                      │                     ▼                     ▲                                      │
                      │               ChromaDB (coseno)     system prompt + 4 fragmentos                 │
                      │               Gemini Embeddings     + últimos 10 mensajes                        │
                      └──────────────── estado por conversación (thread_id) ─────────────────────────────┘
```

| Nodo | Qué hace | Por qué |
|---|---|---|
| `rewrite_query` | Reescribe la pregunta como consulta autónoma usando el historial | Sin este paso, «¿y qué nació de su sangre?» se buscaría sin saber que «su» se refiere a Medusa. En el primer turno se omite, porque no hay nada que resolver y así se ahorra una llamada |
| `retrieve` | Recupera los 4 fragmentos más relevantes de ChromaDB, con su tipo de fuente | Garantiza que **todas** las consultas pasan por la base de conocimiento |
| `generate` | Gemini responde con system prompt + fragmentos + historial reciente | La respuesta queda anclada en las fuentes y es coherente con la conversación |

### Memoria

- El grafo se compila con un **checkpointer de LangGraph** (`MemorySaver`), que guarda el estado completo de cada conversación bajo un `thread_id`.
- **Cada conversación es independiente.** En Streamlit, cada sesión de navegador tiene su `thread_id`: dos usuarios nunca comparten historial, y cambiar de página no borra el chat.
- **Ventana de contexto:** al LLM solo se le envían los **últimos 10 mensajes**, para que una conversación larga no desborde el contexto. El nodo `generate` le indica al modelo si está en el primer turno o si su historial está recortado, para que no "rellene" lo que no ve.

---

## 4. Decisiones técnicas y su justificación

Todos los parámetros están centralizados y comentados en [`src/config.py`](src/config.py).

| Decisión | Valor | Justificación |
|---|---|---|
| Tamaño de chunk | 1000 caracteres | Equivale a 1-2 párrafos: un episodio completo (quién, qué, por qué). Si es mayor, mezcla episodios y diluye el embedding; si es menor, pierde el sujeto |
| Solapamiento | 150 caracteres (15 %) | Una frase cortada entre dos chunks aparece entera en al menos uno |
| Separadores | Sección → párrafo → frase → palabra, con *regex* de anchura cero | Se respeta la estructura del texto, y el corte de frase se hace *después* del punto, así ningún chunk empieza por «. » |
| Cabecera contextual | `[Homero, Odisea IX — El cíclope Polifemo] …` al inicio de cada chunk | Un fragmento como «le clavamos la estaca en el ojo» no dice de quién habla; con el título delante, su embedding sí lo sabe |
| Embeddings | `gemini-embedding-2` | Multilingüe (corpus ES + EN) y de 3072 dimensiones. La librería usa `RETRIEVAL_DOCUMENT` al indexar y `RETRIEVAL_QUERY` al buscar |
| Métrica | Similitud coseno | Compara la dirección de los vectores (el significado) y no su magnitud. Es la estándar para texto |
| k | 4 fragmentos | Unos 1000 tokens de contexto: caben dos tipos de fuente sobre el mismo tema sin meter ruido |
| LLM | `gemini-3.5-flash-lite` | En RAG la calidad depende sobre todo del contexto recuperado, así que priorizamos latencia (demo en vivo) y coste |
| Temperatura | 0.3 (respuesta) / 0.0 (reformulación) | Respuestas fieles con redacción natural; reformular es una tarea determinista |
| Indexación por lotes | 80 chunks por lote con reintento | La capa gratuita de Gemini admite 100 embeddings por minuto |
| Persistencia | `chroma_db/` en disco, versionado | Se indexa una vez y se incluye en el repositorio: el notebook y Streamlit Cloud cargan al instante sin gastar cuota |

### Código agnóstico al dominio

El motor (`knowledge_base.py` y `agent.py`) no sabe nada de mitología. Para convertir el asistente en experto de otro tema:

1. Poner los documentos (`.txt`, `.md` o `.pdf`) en `data/`, con o sin cabecera de metadatos.
2. Crear un nuevo `DomainProfile` en `src/domain.py`, con su nombre, su colección y su system prompt.
3. (Interfaz) Sustituir `app/domain_content.py`: panteón, árboles y tipos de fuente.

El agente recibe el buscador a través de un `Protocol` (`Searcher`), así que se podría sustituir ChromaDB sin tocar el grafo.

---

## 5. System prompt y justificación

El prompt completo está en [`src/prompts.py`](src/prompts.py). Se estructura en bloques con títulos (`# ROL`, `# TUS FUENTES`, `# CÓMO RESPONDES`…) porque los LLM siguen mejor las instrucciones agrupadas y jerarquizadas que un párrafo corrido.

| Bloque | Instrucción | Justificación |
|---|---|---|
| **Rol** | «Eres Mnemósine, titánide de la memoria… experta en mitología griega» | Una personalidad coherente con el dominio hace la experiencia memorable y **delimita el ámbito** |
| **Tus fuentes** | Explica los tres tipos de fuente y su autoridad, que hay pasajes en inglés que debe traducir y qué significa «Minerva [Atenea]» | El modelo solo puede ponderar bien las fuentes si sabe qué es cada una (el tipo llega con cada fragmento) |
| **1. Prioridad** | Basarse en los fragmentos y, si hay fuente primaria, apoyarse en ella primero | Es la esencia de la triangulación: el texto antiguo manda |
| **2. Citas** | Citar autor y obra: `[Hesíodo, Teogonía]`, `[Homero, Odisea IX]`, `[Bulfinch]`, `[Wikipedia — Zeus]` | Respuestas **verificables**; en la demo se ve de qué tipo de fuente sale cada dato |
| **3. Información insuficiente** | Puede usar conocimiento general, pero marcado como «(fuera de mis fuentes)», que nunca acompaña a una cita | Prohibir del todo el conocimiento general haría al agente inútil; permitirlo sin marcarlo haría invisible el origen |
| **4. Versiones del mito** | Contrastar las fuentes cuando difieren («según Hesíodo…, mientras que Bulfinch…») | Es el **error típico de los LLM en mitología** y el motivo principal de este diseño |
| **5. Memoria** | Resolver «él», «su padre», «y después» con los turnos anteriores | Coherencia conversacional. Complementa al nodo `rewrite_query` |
| **6. Preguntas sobre la conversación** | Responderlas solo con el historial visible y admitir si ya no lo recuerda | Añadido tras las pruebas (ver más abajo) |
| **Fuera de dominio** | No responder, explicarlo con amabilidad y sugerir una pregunta válida | Mantiene el foco y reconduce la conversación |
| **Tono y formato** | Evocador pero riguroso; de 1 a 4 párrafos | Personalidad sin sacrificar precisión |

### Iteración del prompt (lo que aprendimos probando)

1. **«Usa tu conocimiento general»** (primera versión). Debilitaba el RAG, porque el modelo mezclaba fuentes sin avisar. → Se sustituyó por la **etiqueta obligatoria** «(fuera de mis fuentes)».
2. **Sin bloque de fuera de dominio.** → Se añadió. Prueba: «¿Cuál es la capital de Australia?» → la rechaza con el tono del personaje.
3. **Memoria inventada.** Con la primera pregunta ya fuera de la ventana de 10 mensajes, el modelo *inventó* cuál había sido (dijo Teseo cuando había sido Perseo); en una conversación nueva inventaba preguntas que nunca existieron. Solución en dos partes:
   - **Prompt:** regla 6.
   - **Código:** el nodo `generate` le dice al modelo si es el primer turno o si su historial está recortado. Las instrucciones deterministas desde el código son más fiables que solo el prompt.
4. **Etiqueta contradictoria con fuentes en inglés.** Al traducir a Bulfinch, el modelo escribía «[Bulfinch] (fuera de mis fuentes)». No tenía claro si lo traducido "contaba". → La regla 3 aclara que lo traducido **sí** está en sus fuentes y que la etiqueta nunca acompaña a una cita.
5. **Copiaba las anotaciones.** Escribía «Aurora [Eos]» en sus respuestas. → El bloque *Tus fuentes* pide escribir directamente el nombre griego.
6. **Relleno y citas repetidas.** Empezaba con «Yo misma, Mnemósine…» y repetía «[Wikipedia — Zeus]» en cada frase. → *Tono y formato* pide entrar directamente en materia y citar cada fuente una sola vez por párrafo.
7. **Metacomentarios en las fichas.** Al pedirle «de quién desciende» (dato que el corpus no recoge) respondía «(fuera de mis fuentes) Aunque las fuentes disponibles…». → La presentación ya no pide la genealogía (está en la ficha) y prohíbe comentar qué fuentes tiene o le faltan.

---

## 6. Instalación

**Requisitos:** Python 3.11 (probado con 3.11.9), una API key de Google Gemini y conexión a internet.

```bash
# 1. Entorno virtual
python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # Linux/Mac

# 2. Dependencias (versiones fijadas)
pip install -r requirements-notebook.txt   # app + notebook
# pip install -r requirements.txt          # solo la app (lo que usa Streamlit Cloud)

# 3. API key
cp .env.example .env             # y edita .env con tu GOOGLE_API_KEY
```

**Cómo obtener la API key:** entra en [aistudio.google.com](https://aistudio.google.com), pulsa **Get API key** → **Create API key** y pégala en `.env`. La key **nunca** va en el código ni en `.env.example`.

---

## 7. Ejecución

### Notebook (entregable principal)

```bash
jupyter notebook notebooks/mnemosine.ipynb
```

Ejecuta las celdas en orden. El repositorio incluye la base vectorial ya construida (`chroma_db/`), así que se carga al instante. Si la borras o cambias `data/`, se reconstruye sola en unos 2 minutos (por el límite de la capa gratuita). El notebook incluye:

- el corpus y un ejemplo de la anotación de nombres;
- la **verificación de la recuperación** antes de conectar el agente (incluida la recuperación multilingüe);
- **8 ejemplos documentados con sus respuestas**, tres de ellos de memoria y uno de control con dos conversaciones;
- una **celda de chat interactiva**.

### Interfaz web Streamlit (bonus)

```bash
streamlit run app/streamlit_app.py
```

| Página | Qué contiene |
|---|---|
| **Inicio** | Presentación, panteón de 12 personajes (medallones con monograma y nombre griego), los tres tipos de fuente y un chat libre abajo |
| **Ficha de personaje** | Al pulsar un personaje: sus datos (padres, dominios, símbolos), una presentación generada por el agente **con RAG y fuentes citadas** (en caché tras la primera vez), cuatro preguntas sobre sus mitos y un chat libre sobre él. URL compartible: `?personaje=zeus` |
| **Oráculo** | El chat: respuesta en *streaming* a través del grafo de LangGraph; debajo, la consulta reformulada y las fuentes con su tipo, relevancia y enlace |
| **Genealogía** | Tres árboles (Orígenes, Olímpicos, Descendencia de Zeus), solo con parentescos y con las versiones alternativas marcadas |
| **Fuentes** | Catálogo de la base de conocimiento por tipo de fuente, idioma y nº de fragmentos |
| **Cómo funciona** | La arquitectura explicada |

La navegación superior permite volver al inicio **sin perder la conversación**; solo «Nueva conversación» abre un hilo nuevo.

Las 48 preguntas de las fichas se eligieron comprobando antes que la base de conocimiento las cubre: todas recuperan fuentes pertinentes (relevancia de 0,68 a 0,81).

**Diseño visual.** Estilo editorial de libro de museo, a propósito alejado de las plantillas habituales:
- Tipografía de libro (*Cormorant SC* y *EB Garamond*), latón envejecido y terracota de cerámica griega, bordes finos y movimiento mínimo.
- Avatares e iconos propios en SVG, en lugar de emojis.
- De fondo, el cielo nocturno con las **12 constelaciones del zodiaco** (`app/assets/zodiac.svg`). Un velo difuminado las atenúa detrás del contenido para que no compitan con el texto.

### Regenerar el corpus (opcional)

```bash
python scripts/build_corpus.py
```

Después, reconstruye la base desde el notebook con `knowledge_base.load_or_build(force_rebuild=True)`.

### Despliegue en Streamlit Cloud

El repositorio está preparado para desplegarse tal cual:

| Pieza | Para qué |
|---|---|
| `requirements.txt` | Solo las dependencias de la app, con **versiones fijadas** (las probadas); Jupyter va aparte, en `requirements-notebook.txt` |
| `chroma_db/` versionado | El índice vectorial ya construido. En Streamlit Cloud el disco se borra en cada reinicio; sin él, la app reindexaría (~2 min y cuota de la API) cada vez que despierta |
| `.streamlit/config.toml` | El tema oscuro de la app |
| `.streamlit/secrets.toml.example` | La plantilla del secreto que hay que configurar |
| `src/` y `app/` | Las rutas se calculan desde la raíz del proyecto y la API key se lee de `st.secrets` si no hay `.env` |

**Pasos:**

1. Crea un repositorio en GitHub y sube el proyecto (el `.gitignore` deja fuera `.env`, `venv/` y las cachés):
   ```bash
   git remote add origin https://github.com/<tu-usuario>/<tu-repo>.git
   git push -u origin main
   ```
2. En [share.streamlit.io](https://share.streamlit.io), pulsa **Create app** → **Deploy a public app from GitHub** y elige:
   - **Repository:** tu repositorio · **Branch:** `main`
   - **Main file path:** `app/streamlit_app.py`
   - **Advanced settings → Python version:** `3.11`
3. En **Advanced settings → Secrets** (o, después, en **Settings → Secrets**), pega:
   ```toml
   GOOGLE_API_KEY = "tu_api_key"
   ```
4. **Deploy.** La primera construcción tarda unos minutos (instalación de dependencias); después la app arranca con el índice ya cargado.

**A tener en cuenta:**
- **Cuota compartida.** Todas las visitas usan tu API key; con la capa gratuita de Gemini, muchas preguntas seguidas pueden toparse con el límite por minuto.
- **Primera visita a cada ficha.** Su presentación se genera la primera vez y después queda en caché mientras la app no se reinicie.
- **Inactividad.** La app se duerme si nadie la usa durante un tiempo; al despertar tarda unos segundos y la memoria de las conversaciones empieza de cero.

---

## 8. Estructura del proyecto

```
├── data/                          # Corpus: 3 documentos, uno por tipo de fuente (~20 págs.)
├── scripts/build_corpus.py        # Descarga, limpieza y anotación del corpus (Gutenberg + Wikipedia)
├── src/                           # Motor RAG, agnóstico al dominio
│   ├── config.py                  # Parámetros técnicos, cada uno justificado
│   ├── domain.py                  # Perfil del dominio (único punto que cambia por tema)
│   ├── prompts.py                 # System prompt, prompt de reformulación y notas de estado
│   ├── knowledge_base.py          # Carga → chunking → Gemini Embeddings → ChromaDB
│   └── agent.py                   # Grafo LangGraph: rewrite_query → retrieve → generate
├── app/                           # Interfaz Streamlit
│   ├── streamlit_app.py           # Entrada: configuración, navegación y barra lateral
│   ├── domain_content.py          # Panteón (fichas y preguntas), árboles genealógicos y tipos de fuente
│   ├── ui/resources.py            # Agente, base vectorial y sesión (compartidos)
│   ├── ui/components.py           # Piezas visuales reutilizables
│   ├── views/                     # Una página por archivo
│   ├── assets/                    # Avatares SVG propios y cielo del zodiaco
│   └── theme.css                  # Estilos
├── notebooks/mnemosine.ipynb      # Notebook con ejemplos documentados
├── chroma_db/                     # Índice vectorial ya construido (versionado para el despliegue)
├── .streamlit/config.toml         # Tema de Streamlit (+ secrets.toml.example)
├── .env.example                   # Plantilla de variables de entorno
├── requirements.txt               # Dependencias de la app (versiones fijadas)
└── requirements-notebook.txt      # + Jupyter, para el notebook
```

---

## 9. Limitaciones y mejoras futuras

- **Memoria con ventana fija.** Lo dicho hace más de 10 mensajes se "olvida". Mejora posible: un nodo que resuma los turnos antiguos.
- **Memoria en RAM.** `MemorySaver` se pierde al reiniciar. Para producción: `SqliteSaver` o `PostgresSaver`, cambiando una línea gracias a que el checkpointer se inyecta.
- **Sin umbral de relevancia.** En preguntas fuera del dominio la relevancia cae a ~0,5, frente a 0,7-0,8 en las del dominio. Un umbral permitiría detectarlas antes de llamar al LLM.
- **Pasajes seleccionados.** Las fuentes primarias son extractos de episodios clave, no las obras completas. Ampliarlas es añadir un `Passage` en `scripts/build_corpus.py`.

---

## Créditos

- Homero, *Ilíada* y *Odisea*, trad. Luis Segalá y Estalella. Hesíodo, *Teogonía*, trad. H. G. Evelyn-White. T. Bulfinch, *The Age of Fable*. Todas de dominio público, vía [Project Gutenberg](https://www.gutenberg.org).
- [Wikipedia en español](https://es.wikipedia.org), licencia [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/deed.es).
