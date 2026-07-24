# pdf2post

Break a local PDF into discrete "posts" and publish them to a WordPress site via the WP REST API. Uses a local LLM (**gemma4** via Ollama) to decide where each post begins and to title it, and pulls embedded artwork out of the PDF to use as each post's featured image.

The LLM only chooses **boundaries and titles** — post bodies are assembled **verbatim** from the extracted text, so nothing in the body is hallucinated.

## Setup

```bash
git clone https://github.com/jeffehobbs/pdf2post.git
cd pdf2post
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
cp .env.example .env            # then edit .env (see below)
```

Requires [Ollama](https://ollama.com) running locally with a gemma4 model
pulled (`ollama pull gemma4:latest`).

### `.env`

| key | meaning |
|-----|---------|
| `WP_URL` | site base URL |
| `WP_USER` | your WordPress login username |
| `WP_APP_PASSWORD` | an **Application Password** — create at *WP Admin → Users → Profile → Application Passwords*. Paste the spaced value as-is. |
| `WP_AUTHOR` | default author credited on posts — username/slug or numeric user id. The `WP_USER` account needs permission to edit others' posts (Author/Editor/Admin) to attribute to a *different* user. |
| `WP_STATUS` | default post status: `publish` (default), `draft`, `pending`, `private` |
| `WP_CATEGORY` | optional category name to file every post under (created if missing) |
| `OLLAMA_MODEL` | model for segmentation (default `gemma4:latest`) |
| `OLLAMA_NUM_CTX` | context window in tokens (default `32768`) — must be big enough to hold the whole document |

## Usage

Always preview first with `--dry-run` — it writes an HTML + image preview of
each post to a local folder and posts nothing:

```bash
./.venv/bin/python pdf2post.py ~/Desktop/issue001.pdf --dry-run
```

Then publish:

```bash
./.venv/bin/python pdf2post.py ~/Desktop/issue001.pdf
```

### Options

```
--status {publish,draft,pending,private}   post status (default from .env / publish)
--author AUTHOR         author username/slug or user id (overrides .env)
--model MODEL           Ollama model (default gemma4:latest)
--category CATEGORY     force this one category on every post (created if missing)
--auto-category         let the LLM assign each post to the best-fitting existing category
--fix-spacing           restore missing spaces in run-together text (see below)
--ocr / --no-ocr        OCR pages whose text is vectorized (default on; see below)
--ocr-dpi N             render DPI for OCR pages (default 200)
--ocr-threshold N       OCR pages with fewer than N chars of text (default 60)
--ocr-draw-threshold N  also OCR sparse-text pages with >= N vector drawings (default 150)
--jpeg-quality N        JPEG quality for featured images (default 85)
--max-image-width PX    downscale featured images wider than this (default 2000; 0 disables)
--min-image-side PX     ignore embedded images smaller than this for featured art (default 200)
--num-ctx N             Ollama context window in tokens (default 32768)
--max-llm-chars N       cap on characters sent to the LLM (default 120000)
--limit N               only process the first N posts (handy for testing)
--dry-run               extract + segment only; write local previews, post nothing
--out DIR               output dir for --dry-run previews
```

### `--fix-spacing`

Some PDFs encode certain text runs with **no space glyphs**, so extraction produces run-together words like `performsattheCambridgeFactory`. With `--fix-spacing`, each affected line is sent through the LLM to reinsert **only** spaces. A guard checks that the letters/digits are unchanged (spaces are the only allowed edit); if a line comes back altered, the original is kept. Repair is done line-by-line so one bad line never spoils the rest of a post. Body prose that already has spaces is left untouched, so this is safe to leave on.

### OCR (vectorized-text pages)

Zines often set headlines, posters, and whole articles in outlined/vectorized
display fonts. The PDF text layer reports those pages as **empty**, so their
content would silently vanish. With OCR on (the default), a page is transcribed
by the vision model when its text layer is near-empty (`--ocr-threshold`) **or**
it has many vector drawings but little text (`--ocr-draw-threshold`) — the
signature of vectorized text. Pages with a healthy text layer (normal prose,
diagrams-with-captions) keep their exact text and are never OCR'd. OCR is
best-effort: transcription of heavily stylized/glitchy pages can contain small
errors, so review OCR'd posts. Uses the same model (`gemma4:latest` is
vision-capable); `--no-ocr` disables it.

### Auto-categorization

With `--auto-category` (or `WP_AUTO_CATEGORY=1`), the script fetches the site's
existing categories and asks the LLM to file each post under the single
best-fitting one — it won't invent new categories, and leaves a post
uncategorized if none genuinely fit. `--category NAME` is the manual
alternative: it forces one category on every post (and creates it if missing).
The two can be combined — the forced category plus each post's auto-assigned
one. On electricant.space this reliably sorts posts into `build` / `essays` /
`shows` / `artists`.

### Featured images → web-friendly JPEG

Extracted art is often at full print resolution (many MB, thousands of pixels
wide). Featured images are re-encoded to JPEG (`--jpeg-quality`, transparency
flattened onto white) and downscaled to `--max-image-width` (default 2000px).
In practice this turns multi-MB PNGs into ~200–300 KB JPEGs.

## How it works

1. **Extract** per-page text and embedded images with PyMuPDF; OCR any
   vectorized-text pages (see above) so no content is lost.
2. **Segment** — gemma4 reads the full text (with page markers) and returns,
   for each post, a title, a verbatim opening anchor, and a start page.
3. **Assemble** — each anchor is located in the original text; each post's body
   is the verbatim text from its anchor to the next post's anchor.
4. **Art** — for each post, the largest embedded image within its page span
   (above `--min-image-side`) becomes the featured image, re-encoded as a
   web-friendly JPEG.
5. **Publish** — images are uploaded to the WP media library, then posts are
   created with the chosen status, author, and optional category.

## Notes / gotchas

- **Fully scanned/image-only PDFs** aren't a target case: the OCR pass above is
  meant for vectorized *text*, and a PDF with no extractable text at all exits
  with a message rather than OCRing cover to cover.
- If a PDF page is set in a spaceless display font, use `--fix-spacing`.
- If Ollama throws a Metal/`MTLCompilerService` error (common after sleep/wake),
  restart the Ollama server and retry.
- Only one image per post is used (the featured image). Inline-embedding every
  image is a possible future flag.

## License

MIT — see [LICENSE](LICENSE).
