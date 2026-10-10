# AI Video File Assistant

A Windows desktop app that renames and organises (video) files from **plain-language commands in English or Bangla**.
You pick a folder, type what you want — *"Remove 1080p and WEB-DL from all filenames"*, *"সব ভিডিওর নামের শেষে 01 থেকে numbering দাও"* —
review a colour-coded **preview**, click **Apply**, and can **Undo** the whole batch afterwards.

* **Any number of AI models**: Gemini, OpenAI, any OpenAI-compatible server (OpenRouter, Groq, LM Studio, Ollama, vLLM …) or a custom endpoint, each with its own
  base URL and API key. A router picks the model per task and **falls back** automatically (429, quota, 5xx, timeouts, invalid answers).
  The AI only *describes* the change; the app validates and performs it.
* Simple commands run **offline** (no AI request, no cost). Big folders need **one** request, not one per file.
* Nothing is changed without a preview. Every batch is journaled, atomic and undoable.

## Contents
1. [Quick start (using the .exe)](#quick-start)
2. [Building the .exe](#building-the-exe)
3. [How a command is processed](#how-a-command-is-processed)
4. [Safety model](#safety-model)
5. [AI models, routing and fallback](#ai-models-routing-and-fallback)
6. [Title Generator (unique titles, Google Search context, presets)](#title-generator)
7. [Command examples](#command-examples)
8. [Undo, history and the undo-trash](#undo-history-and-the-undo-trash)
9. [Optional: durations and thumbnails (ffmpeg)](#optional-durations-and-thumbnails-ffmpeg)
10. [Tests](#tests) · [Project layout](#project-layout) · [Extending](#extending-the-app) · [Troubleshooting](#troubleshooting)

## Quick start
1. Start **AI Video File Assistant.exe**. On first start click **Add Model** in the welcome banner (or *Settings → AI → AI Models → Add Model*),
   pick a provider type, paste the API key, choose a model (or **Load Models**), press **Test Connection**, then **Save**.
2. **Browse** (or drag & drop a folder). 3. Type a command, pick the **AI mode** (default *Auto + Fallback*) and **Model** (default *Automatic*),
   press **Generate Preview** (Ctrl+Enter). The line under the command shows `AI: Gemini / gemini-2.5-flash` before the request and what was used afterwards.
4. Check the preview, then **Apply Changes**. 5. Changed your mind? **Undo** (top right) or open **History**.

Offline commands need no key at all. Only file **names** (never file contents, never full paths) are sent to an AI.

## Building the .exe
Requirements: Windows 10/11 and **Python 3.12+** (python.org installer, "Add to PATH" + "py launcher").

```bat
cd AI-Video-File-Assistant
build.bat              rem one-folder build  -> dist\AI Video File Assistant\AI Video File Assistant.exe
build.bat onefile      rem single-file build -> dist\AI Video File Assistant.exe
```

`build.bat` creates `.venv`, installs `requirements-dev.txt`, **runs the test-suite** (the build stops if a test fails), runs PyInstaller with
`AI Video File Assistant.spec`, then **starts the new exe with `--self-test`** (scan → plan → apply → undo in a temp folder) to prove the bundle is complete.
`set SKIP_TESTS=1` skips the test run. Running from source: `python -m pip install -r requirements.txt` then `python main.py`.

### Packaging trade-offs (why the default is one-folder)
| | **One-folder** (default) | One-file (`build.bat onefile`) |
|---|---|---|
| Start-up | fast (nothing to unpack) | slower: unpacks ~60–150 MB to `%TEMP%` on **every** launch |
| Antivirus | fewer false positives | self-extracting exes are flagged more often |
| ffmpeg/ffprobe | bundled by dropping the `.exe`s into `app\resources\bin` before building | would be re-extracted each launch, so **not bundled** — put `bin\ffprobe.exe`/`ffmpeg.exe` next to the exe or use *Settings → General* |
| Distribution | zip the folder | one file |

Both builds are verified in CI-style runs of `--self-test`. Ship the one-folder build as a zip (or wrap it with an installer such as Inno Setup).

## How a command is processed
```
command ──► offline parser ─(simple? done, 0 requests)
              │ no
              ▼
   task analyzer ──► model registry ──► capability filter ──► strategy / priority
              ▼
        AI router ── model 1 ─(429/5xx/timeout/bad JSON)─► model 2 … ──► JSON only
              ▼
        strict response parser   (untrusted data!)
              ▼
        planner  ── rule actions applied locally to ALL selected files
                 ── explicit actions resolved, every path validated
              ▼
        batch validator ── conflicts, duplicates, overwrites, missing folders
              ▼
        PREVIEW ──(you)──► operation manager: journal → two-phase rename → commit / rollback
```
* **Rule actions** (`remove_text`, `replace_text`, `add_prefix`, `add_suffix`, `numbering`, `change_case`, `sort`, and `move`/`copy`/`delete` with `match`) are described
  once by the AI and applied locally to every file — so a 5 000-file folder costs **one** request.
* **Explicit actions** (`rename`, `move`, `copy`, `delete`, `create_folder`) are used when each file needs its own interpretation; those files are sent in
  **batches** (*Settings → Files per AI request*, default 50) that also respect a character budget (*Max characters per request*), so 500 files become
  a handful of requests, never 500.
* `sort` reorders the working list (affects numbering order and the table); it never touches files.

## Safety model
* The AI **never executes anything**. It can only return the 12 whitelisted action types; anything else — or any answer containing keys like `command`, `shell`, `powershell`,
  `script`, `exec` — is rejected as a whole. No shell, CMD, PowerShell, registry or process is ever used on AI output.
* AI output is **untrusted**: JSON is parsed strictly; sources must be among the *selected* files; targets must be plain relative names — no absolute paths, drive letters, UNC paths, `..`,
  illegal Windows characters, reserved names (`CON`, `NUL`…), trailing dots/spaces, or paths that leave the workspace through symlinks/junctions.
* File names in the prompt are marked as **data, not instructions** (prompt-injection hardening), and a deleting plan is only ever a *preview* until you confirm.
* **Extensions are preserved** unless the command explicitly asks otherwise; an unrequested change (`.mp4`→`.mkv`) is rejected.
* **Previews first**: conflicts and invalid rows are shown and skipped; valid rows can be un-ticked individually (the plan is re-validated).
* **Deletes** always need an extra confirmation ("Delete 3 files? This action may not be reversible."), even with *Ask before applying* off or *Auto Apply* on.
  Auto Apply (off by default) never applies plans containing deletes, conflicts or invalid rows.
* **Two-phase renames** (`A→B`, `B→C`, swaps, case-only changes) go through `.tmp_ai_*` temp names so nothing is ever overwritten.
* **All-or-nothing**: every step is journaled in SQLite first; if one file fails (locked, disk full, cancelled) everything done so far is rolled back.
* API keys are stored **per model**, encrypted with **Windows DPAPI** (only your Windows user can decrypt), shown as `************ABCD`, sent in HTTP *headers* (never URLs),
  never written to the settings table, backups or logs. Base URLs must be `https://` (plain `http://` is allowed only for `localhost`/private-network servers).

## AI models, routing and fallback
**Configure several models** in *Settings → AI → AI Models*. The table shows Provider, Model, Base URL, Status (Online / Cooldown / No key / Disabled / Untested / Error),
Enabled, Priority and Capabilities; selecting a row shows its statistics (`✓ 98% success · avg 1.8s · used 43×`, last success/failure, last error).
Buttons: **Add Model, Edit, Remove, Test Connection, Load Models, Enable/Disable, Move Up/Down**. Each model has:

| Field | Notes |
|---|---|
| Provider type | Gemini · OpenAI · OpenAI-compatible · Custom (chat-completions format, no JSON mode / listing) |
| Display name, Base URL | Base URL is pre-filled for Gemini/OpenAI and editable (proxies, regional endpoints) |
| API key | Its **own** key per model (two models may use different accounts). When editing: **Keep Existing Key** / **Replace Key** — an empty field never wipes a key. **Remove Key** asks *"Remove this saved API key from this computer?"* and deletes only the key. |
| Model | Typed, or picked with **Load Models** |
| Priority, Timeout, Max retries, Enabled | Lower priority number = tried first |
| Capabilities | Text, JSON, Long context, Fast, Vision — used by the task-aware routing |
| Optional limits / prices | Requests per minute/day and price per 1M tokens — **only what you enter**; the app never guesses prices |

**Load Models** calls the provider's model-list endpoint with the form's base URL and key (Gemini `GET /models`, OpenAI-style `GET /models`), shows the list,
and every model you tick is saved as its own configuration (same URL, key and settings). Providers without a list endpoint (or a `404`) show
*"Model listing is not available for this provider. Enter the model ID manually."* **Test Connection** validates URL + key, sends one tiny request,
measures the response time and checks the answer is JSON.

**AI mode (routing strategy)** — in the command area and in *Settings → AI → Routing Strategy*:
* **Auto + Fallback** (default): the task analyzer classifies the command (simple / complex rename / titles / descriptions / organisation / large batch / JSON),
  the router keeps enabled models that have a key and the needed capabilities, ranks them (capability match, then priority; *cost-aware* adds price after
  capability), puts a chosen **Model** first, and skips models in **cooldown** or over their rate limit. On HTTP 429/quota, 5xx, timeout, network failure or an
  invalid/unstructured answer it moves to the next model. Transient errors are retried on the same model up to its *Max retries*; the number of models
  tried is capped (*Max models tried per request*), so there are no endless loops. An **unsafe** answer stops immediately — it never shops for another model.
* **Auto**: same choice, no fallback. **Manual**: only the selected model. **Cheapest / Fastest / Highest Priority**: rank by your prices / measured latency / list order, with fallback.
* **Health**: every call records success/failure, latency and tokens. A rate limit puts a model in cooldown at once; repeated failures (default 2) start a cooldown
  that doubles while it keeps failing. After the cooldown it gets its normal place back.

What happened is always visible: `AI: Gemini / gemini-2.5-flash` before the request; afterwards e.g.
`Gemini / gemini-2.5-flash (HTTP 429) → OpenAI / gpt-4o-mini ✓` with *Primary / Failed / Fallback / Success* lines, and the preview header shows
`Completed · Model: … · Requests: 2 · Fallbacks: 1 · Files: 500`. Simple commands are still handled **offline** first.

**Persistence & migration.** Models, priorities, capabilities, strategy, preferred model and all preferences survive restarts; the AI mode / model chosen in the
command area is saved automatically. Keys from the previous single-Gemini/OpenAI version are **migrated automatically** (no re-entry).
*Settings → General → Backup & Restore* exports preferences, model configurations and saved prompts **without API keys**.

*File Operations*: ask before applying, operation history, prevent overwrite, include subfolders, **Auto Apply (off by default)**, duplicate-name policy, delete mode (undo-trash / permanent).
*Appearance*: Dark / Light / System. *General*: reopen last folder, **language (English / বাংলা)**, default filter, log level, ffmpeg folder, backup & restore.
Data folder: `%APPDATA%\AI Video File Assistant` (SQLite database, logs, thumbnail cache). Override with the `AIVFA_DATA_DIR` environment variable.

## Title Generator
Open **Title Generator** in the AI Command card, tick **Use for the next preview**, select 20 / 50 / 100+ videos and press **Generate Preview**
(a command is optional — if you type one it can override the panel, e.g. *"Do not use the original filenames. Make them cinematic and suspenseful,
10–16 words each. Start every title with Blue Bloods."*). Every video gets its **own** title, e.g. `Blue Bloods - One Hidden Clue Changes Everything for the Reagan Family.mp4`.

| Option | Values |
|---|---|
| Title mode | **Independent Creative** (default — the file name is only an identifier and is *never sent* to the AI), Partially Connected (loose inspiration), Connected to Filename |
| Information source | **Gemini / AI only** (default), Google Search + Gemini / AI, Google Search only, Custom prompt |
| Category · Content/Topic · Keywords | Crime … Emotional or a custom category; free-text topic; comma-separated keywords (used naturally, not forced into every title) |
| Style · Tone | Cinematic, Story, Dramatic, Suspenseful, Professional, Simple, Emotional, YouTube, Facebook, custom · Neutral … Professional, custom |
| Word capacity | Short 5–8 · Medium 8–12 · Long 12–18 · Custom min/max. The **series prefix does not count** unless *Prefix counts as words* is ticked |
| Series prefix · Variation | `Blue Bloods` → `Blue Bloods - <title>` · Low / Medium / **High** |
| Checks · Custom instructions | Unique title for every video, Do not use original filename, Prevent duplicate titles · free text that is always sent with the request |

**How it works.** Titles are requested in batches (25 per request — 100 videos ≈ 4 requests) through the normal AI router, so the configured models,
AI mode and **fallback** apply unchanged. The AI answers with titles per item number; the app adds the prefix, keeps the **original extension** and
builds ordinary rename actions, which go through the usual validation, **preview**, Apply and **Undo**. Before the preview a local checker compares the
whole batch — exact duplicates, near-duplicates (shared vocabulary, almost identical wording), repeated openings, repeated sentence structures,
over-used words, word range, and titles that copy the file name or a search-result title — and only the flagged titles are regenerated (at most 3 rounds;
anything left is reported, and an exact duplicate can never be applied because the preview marks it as a conflict).
The AI is told it has **not seen the videos**: specific events, characters or places are only allowed when the search context supports them.

**Google Search.** Set the Google API key (Custom Search API enabled) and the Programmable Search Engine ID under *Settings → AI → Google Search*
(key encrypted with DPAPI, sent as a header). Per file a concise query is built (*Automatic*: `Blue Bloods S05E03 episode` from the file name if
*Use filename for information lookup* is on, otherwise series + topic; or *Filename*, *Filename + Series*, *Custom* template such as
`{series} {season} {episode} episode plot`). Identical queries run **once per batch** and are cached locally (default 24 h, **Clear Search Cache**).
Only titles/snippets of up to 3/5/10 results are kept: HTML stripped, instruction-like text dropped, reliable sources (Wikipedia, IMDb, TVmaze …) first,
terms confirmed by several sources and disagreements marked — and sent as **context only**. If the search fails, generation continues with the AI only
(*If Google Search fails, continue with AI only*, on by default) and the preview says so. *Google Search only* uses no AI: it can name episodes from
search results (Connected modes) but refuses Independent Creative titles because those need an AI model. *Custom prompt* lets your instructions lead
(mention "search" to add search context). The preview's note column shows `Search: ✓ 5 results · AI: Gemini / gemini-2.5-flash` per file.

**Presets.** *Save Preset / Load Preset / Delete Preset* store the complete configuration (e.g. "Blue Bloods Facebook Titles") in the local database;
the current panel settings are also restored automatically at start-up.

## Command examples
| You type | What happens |
|---|---|
| `Remove 1080p and WEB-DL from all filenames.` | offline: `remove_text` |
| `Replace WEB-DL with WEB.` / `Replace underscores with spaces` | offline: `replace_text` |
| `Add prefix "S02 - "` / `Blue Bloods নামটা শুরুতে যোগ করো।` | offline: `add_prefix` |
| `সব ভিডিওর নামের শেষে 01 থেকে numbering দাও।` | offline: `numbering` |
| `Convert filenames to lowercase` · `Sort by size descending` | offline: `change_case` · `sort` |
| `Clean these filenames and rename them sequentially as Blue Bloods Episode 01, 02 and 03.` | AI → one `numbering` rule → `Blue Bloods - Episode 01.mp4` … |
| `Season 2-এর সব ভিডিও আলাদা folder-এ রাখো।` | AI → `create_folder` + `move` by pattern |
| `Give each video a clean title from its name` | AI → explicit `rename` per file, in batches |

The five built-in **Saved Prompts** (Blue Bloods Rename, Clean Video Filename, Episode Numbering, Remove Quality Tags, YouTube Title Format) can be edited or deleted; save your own with **Save as prompt**.
**Recent** commands are one click away, can be favourited, reused or deleted.

## Undo, history and the undo-trash
Every applied batch is recorded (*History*: operation #, time, command, provider, summary, status, every item). **Undo** first checks the disk; if it cannot be done safely
(a file was renamed again, its original name is taken, a copy was edited, the trash was emptied…) it **explains why and changes nothing**. Deleted files go to
`.ai_video_assistant_trash` inside the folder (same drive → instant) so Undo can restore them; **History → Empty undo-trash** frees the space. Choose *Delete permanently* in Settings if you prefer — those files can't be restored.
Operations interrupted by a crash or power loss are marked *Interrupted* on the next start and can still be undone (temp files included).

## Optional: durations and thumbnails (ffmpeg)
With `ffprobe`/`ffmpeg` available (bundled in `app\resources\bin`, on `PATH`, or via *Settings → General*), the **Duration** column fills in **lazily for visible rows only**, results are cached in SQLite,
and selecting a video shows a cached **thumbnail**, resolution, fps and codecs. Thumbnails are never generated in bulk. Without ffmpeg these features just stay hidden.

## Tests
```bash
python -m pip install -r requirements-dev.txt
python -m pytest            # headless (QT_QPA_PLATFORM=offscreen is set automatically)
```
Over 600 tests cover validation, path traversal, collisions, two-phase renames, rollback, undo, recovery, JSON parsing, AI/offline command handling, every provider
adapter, Load Models, routing strategies, fallback (429/5xx/timeout/invalid JSON), cooldowns, migration, backup without keys and persistence across restarts
(all against a fake local HTTP server — no real API keys are needed), Bangla/Unicode names, the GUI workflow and 20 000-row performance. ffmpeg tests skip automatically if ffmpeg is absent.

## Project layout
```
main.py                      entry point (--self-test for packaging checks)
build.bat / *.spec           build + PyInstaller recipe
app/config/                  constants (models, limits), typed settings
app/ai/                      model config/registry/health, provider adapters, task analyzer, router, prompt builder, parser, pipeline,
                             title generator (title_config, title_quality, title_generator, web_search)
app/files/                   scanner, planner, validators, rename/move engines, operation manager, metadata, thumbnails
app/database/                SQLite access
app/workers/                 QRunnable workers (scan, AI, operations, ffprobe) - the UI never blocks
app/ui/                      main window, panels, dialogs, theme, icons
app/i18n/                    English + Bangla strings
tests/                       pytest suite
```

## Extending the app
* **New AI provider type**: subclass `AIProvider` (`app/ai/base_provider.py`), add it to `app/ai/providers.py:ADAPTERS` and `app/ai/model_config.py:PROVIDER_TYPES`.
  Most hosted services already work as *OpenAI-compatible*.
* **New action type**: add a dataclass in `app/ai/schemas.py`, a builder in `app/ai/response_parser.py`, handle it in `RuleEngine`/`Planner`, and describe it in `prompt_builder.SYSTEM_PROMPT`.
* **New language**: add `app/i18n/<code>.py` (same keys) and register it in `app/i18n/__init__.py`; a test enforces key parity.
* The architecture leaves room for the roadmap items (YouTube titles/descriptions, subtitle management, duplicate detection, categorisation, batch metadata editing, workflows):
  they are new action types + planner/executor handlers over the same preview → journal → undo pipeline.

## Troubleshooting
| Message | Meaning |
|---|---|
| *Authentication error* | The key was rejected — re-enter it (Settings → AI → Test Connection). |
| *Rate limit or quota reached* | The model rests in cooldown; with *Auto + Fallback* the next model is used. Add another model or wait. |
| *No usable AI model is configured* | Add a model (Settings → AI → AI Models) or give the existing one its API key. |
| *Model listing is not available…* | The provider has no model-list endpoint — type the model ID. |
| *Network error / Timeout* | No connection or a slow reply; raise *Request timeout* or retry. |
| *The file is in use by another program* | A player/Explorer pane holds the file; close it and apply again (nothing was changed). |
| *The folder changed since the preview* | Files were added/removed after the preview; generate a new one. |
| *Undo is not possible right now* | The dialog lists exactly why (e.g. a file was renamed again). |

Logs: *Settings → General → Open log folder* (never contain API keys).
