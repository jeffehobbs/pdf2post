#!/usr/bin/env python3
"""
pdf2post — break a local PDF into discrete "posts" and publish them to a
WordPress site (electricant.space by default) via the REST API.

Pipeline:
  1. Extract per-page text + embedded images with PyMuPDF.
  2. Ask a local LLM (gemma4 via Ollama) where each post begins and what to
     title it. The LLM only decides *boundaries and titles* — the post body is
     assembled verbatim from the extracted text, so nothing is hallucinated.
  3. Pick a featured image for each post (largest embedded image within the
     post's page span).
  4. Upload images to the WP media library and create posts, crediting a
     configurable author.

Config lives in .env (see .env.example). Run with --dry-run first to preview.
"""

from __future__ import annotations

import argparse
import html
import io
import json
import mimetypes
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF
import requests
from dotenv import load_dotenv

try:
    import ollama
except Exception:  # pragma: no cover
    ollama = None


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #

@dataclass
class PageText:
    number: int          # 1-indexed
    text: str
    char_start: int      # offset of this page's text within the joined blob
    char_end: int
    ocr: bool = False    # text came from vision OCR, not the PDF text layer


# A page with at least draw_threshold vector drawings but fewer than this many
# extractable chars is treated as having text set in outlined/vectorized fonts.
OCR_DRAW_TEXT_CAP = 2000


@dataclass
class OcrConfig:
    model: str
    dpi: int
    threshold: int       # OCR a page whose extractable text is shorter than this
    draw_threshold: int  # ...or that has >= this many vector drawings + sparse text
    num_ctx: int


@dataclass
class ExtractedImage:
    page: int            # 1-indexed
    xref: int
    data: bytes
    ext: str
    width: int
    height: int

    @property
    def area(self) -> int:
        return self.width * self.height


@dataclass
class Post:
    title: str
    body: str
    page_start: int
    page_end: int
    featured: ExtractedImage | None = None
    # populated after categorization
    category_id: int | None = None
    category_name: str | None = None
    # populated after posting
    wp_id: int | None = None
    wp_link: str | None = None
    media_id: int | None = None


# --------------------------------------------------------------------------- #
# PDF extraction
# --------------------------------------------------------------------------- #

PAGE_MARKER = "\n\n===== PAGE {n} =====\n\n"


OCR_PROMPT = (
    "You are transcribing a page from a print zine. Transcribe ALL legible "
    "text in natural reading order — for multi-column layouts, read each "
    "column top-to-bottom, left-to-right. Preserve paragraph breaks. Do NOT "
    "describe images or graphics, do NOT invent, correct, or translate text, "
    "and ignore purely decorative marks that aren't words. If a page has no "
    "readable text, output nothing. Output only the transcription."
)


def ocr_page(page: "fitz.Page", cfg: OcrConfig) -> str:
    """Transcribe a page image with a vision-capable local model via Ollama."""
    if ollama is None:
        return ""
    png = page.get_pixmap(dpi=cfg.dpi).tobytes("png")
    resp = ollama.chat(
        model=cfg.model,
        messages=[{"role": "user", "content": OCR_PROMPT, "images": [png]}],
        options={"temperature": 0.0, "num_ctx": cfg.num_ctx},
    )
    return (resp["message"]["content"] or "").strip()


def extract_pdf(
    path: Path, ocr_cfg: OcrConfig | None = None
) -> tuple[list[PageText], str, list[ExtractedImage]]:
    """Return per-page text, a single joined text blob, and embedded images.

    Pages whose extractable text is shorter than ocr_cfg.threshold (and that
    have visual content) are transcribed via vision OCR — this recovers pages
    whose text is set in outlined/vectorized display fonts, which the PDF text
    layer reports as empty.
    """
    doc = fitz.open(path)
    pages: list[PageText] = []
    images: list[ExtractedImage] = []
    blob_parts: list[str] = []
    cursor = 0

    seen_xrefs: set[int] = set()
    for i, page in enumerate(doc):
        n = i + 1
        raw = page.get_text("text") or ""
        raw = raw.strip()

        did_ocr = False
        if ocr_cfg:
            n_draw = len(page.get_drawings())
            low_text = len(raw) < ocr_cfg.threshold
            vectorized = n_draw >= ocr_cfg.draw_threshold and len(raw) < OCR_DRAW_TEXT_CAP
            if low_text or vectorized:
                why = "sparse text" if low_text else f"{n_draw} vector drawings, sparse text"
                print(f"  · OCR page {n} ({len(raw)} chars text layer; {why}) …")
                ocr_text = ocr_page(page, ocr_cfg)
                # Prefer OCR only when it recovers more than the text layer had.
                if len(ocr_text) > len(raw):
                    raw = ocr_text
                    did_ocr = True

        # A page marker precedes each page's text. It is counted in the offset
        # cursor (so offsets stay accurate) but sits *before* the page's
        # char_start, so post bodies never begin with a marker. text_to_html
        # strips any markers that fall inside a body.
        marker = PAGE_MARKER.format(n=n)
        blob_parts.append(marker)
        cursor += len(marker)

        start = cursor
        blob_parts.append(raw)
        cursor += len(raw)
        end = cursor

        pages.append(PageText(number=n, text=raw, char_start=start, char_end=end, ocr=did_ocr))

        for img in page.get_images(full=True):
            xref = img[0]
            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)
            try:
                info = doc.extract_image(xref)
            except Exception:
                continue
            data = info.get("image")
            if not data:
                continue
            images.append(
                ExtractedImage(
                    page=n,
                    xref=xref,
                    data=data,
                    ext=info.get("ext", "png"),
                    width=info.get("width", 0),
                    height=info.get("height", 0),
                )
            )

    doc.close()
    return pages, "".join(blob_parts), images


def page_for_offset(pages: list[PageText], offset: int) -> int:
    """Which 1-indexed page does a char offset fall in?"""
    for p in pages:
        if p.char_start <= offset < p.char_end:
            return p.number
    # offset may land in a separator; snap to the nearest preceding page
    last = 1
    for p in pages:
        if p.char_start <= offset:
            last = p.number
    return last


# --------------------------------------------------------------------------- #
# LLM segmentation
# --------------------------------------------------------------------------- #

SEGMENT_SYSTEM = (
    "You are an editor preparing a print PDF (a zine/newsletter) for a blog. "
    "You split the raw extracted text into a SMALL number of discrete posts. "
    "Each post is one complete top-level piece — a whole article, a whole "
    "essay, or a whole short story — the kind of thing that would run as a "
    "single blog entry with its own title. You only choose where each piece "
    "begins and write a title for it. You never rewrite, summarize, translate, "
    "or invent body text."
)

SEGMENT_INSTRUCTIONS = """\
Below is the full extracted text of a PDF. Page breaks are marked with
"===== PAGE N =====".

Identify the discrete top-level posts in reading order. For EACH post return:
  - "title": a concise, human title (you may invent this).
  - "start_anchor": the FIRST 6-12 words of that post, copied EXACTLY and
    VERBATIM from the text (same words, same order). This marks where the
    post begins. Do not paraphrase the anchor.
  - "start_page": the page number (integer) where the post begins.

CRITICAL rules about granularity:
  - A continuous work is ONE post. A short story, essay, or article is a
    SINGLE post even when it is long, spans multiple pages, or contains scene
    breaks, chapter breaks, section headings, or blank lines. Do NOT split a
    single narrative or argument into pieces.
  - Only start a new post when the piece genuinely changes: a different
    author, a different topic, or an obviously separate item (e.g. an article,
    then a short story, then a list of performer bios).
  - Most zines have only a handful of posts (often 2-6). Prefer fewer, larger
    posts over many small fragments. When unsure whether two adjacent chunks
    are the same piece, keep them together.
  - Skip pure boilerplate (mastheads, page numbers, ads, tables of contents)
    unless it is genuinely the content.
  - Anchors must appear literally in the text so the boundary can be located.

Respond with ONLY a JSON object of this exact shape:
{"posts": [{"title": "...", "start_anchor": "...", "start_page": 1}, ...]}

TEXT:
---
%s
---
"""


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def segment_with_llm(blob: str, model: str, max_chars: int, num_ctx: int) -> list[dict]:
    if ollama is None:
        raise RuntimeError("ollama python package not available")

    text = blob
    if len(text) > max_chars:
        text = text[:max_chars]
        print(f"  ! text truncated to {max_chars} chars for the LLM", file=sys.stderr)

    prompt = SEGMENT_INSTRUCTIONS % text
    resp = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": SEGMENT_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        format="json",
        # num_ctx must be large enough to hold the whole document, or Ollama
        # silently truncates the prompt and the model only "sees" the start.
        options={"temperature": 0.0, "num_ctx": num_ctx},
    )
    content = resp["message"]["content"]
    data = json.loads(content)
    posts = data.get("posts") if isinstance(data, dict) else None
    if not isinstance(posts, list) or not posts:
        raise ValueError("LLM returned no posts")
    return posts


def build_posts(
    blob: str,
    pages: list[PageText],
    seg: list[dict],
) -> list[Post]:
    """Turn LLM boundary decisions into posts with verbatim bodies."""
    norm_blob = _normalize(blob)

    # Locate each anchor's offset in the original blob.
    located: list[tuple[int, str]] = []  # (offset, title)
    for item in seg:
        title = (item.get("title") or "").strip() or "Untitled"
        anchor = (item.get("start_anchor") or "").strip()
        offset = None
        if anchor:
            offset = _find_anchor(blob, norm_blob, anchor)
        if offset is None:
            # fall back to the start of the given page
            sp = item.get("start_page")
            if isinstance(sp, int):
                pg = next((p for p in pages if p.number == sp), None)
                if pg:
                    offset = pg.char_start
        if offset is None:
            offset = 0
        located.append((offset, title))

    # Sort by position, dedupe identical offsets.
    located.sort(key=lambda t: t[0])
    deduped: list[tuple[int, str]] = []
    for off, title in located:
        if deduped and off == deduped[-1][0]:
            continue
        deduped.append((off, title))

    posts: list[Post] = []
    for idx, (off, title) in enumerate(deduped):
        end = deduped[idx + 1][0] if idx + 1 < len(deduped) else len(blob)
        body = blob[off:end].strip()
        if not body:
            continue
        posts.append(
            Post(
                title=title,
                body=body,
                page_start=page_for_offset(pages, off),
                page_end=page_for_offset(pages, max(off, end - 1)),
            )
        )
    return posts


def _find_anchor(blob: str, norm_blob: str, anchor: str) -> int | None:
    """Find an anchor's char offset in blob, tolerant of whitespace/case."""
    # 1) direct case-insensitive search
    idx = blob.lower().find(anchor.lower())
    if idx != -1:
        return idx
    # 2) whitespace-normalized search, then map back to a real offset
    na = _normalize(anchor)
    if not na:
        return None
    pos = norm_blob.find(na)
    if pos == -1:
        # 3) try just the first few words
        words = na.split()
        if len(words) > 4:
            short = " ".join(words[:4])
            pos = norm_blob.find(short)
        if pos == -1:
            return None
    # Map normalized position back to a blob offset by counting non-space chars.
    target_nonspace = len(re.sub(r"\s", "", norm_blob[:pos]))
    seen = 0
    for i, ch in enumerate(blob):
        if not ch.isspace():
            if seen == target_nonspace:
                return i
            seen += 1
    return None


# --------------------------------------------------------------------------- #
# Auto-categorization (assign each post to an existing WordPress category)
# --------------------------------------------------------------------------- #

def fetch_categories(base_url: str, session: "requests.Session | None" = None) -> list[dict]:
    """Fetch the site's categories (public GET; no auth required)."""
    s = session or requests.Session()
    if session is None:
        s.headers.update({"User-Agent": "pdf2post/1.0"})
    r = s.get(
        base_url.rstrip("/") + "/wp-json/wp/v2/categories",
        params={"per_page": 100, "_fields": "id,name,slug,description"},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


CATEGORY_SYSTEM = (
    "You file a blog post into one of a fixed set of existing categories. You "
    "pick the single best fit. If none of them genuinely fit, you answer NONE. "
    "You never invent a category that is not in the provided list."
)

CATEGORY_INSTRUCTIONS = """\
Available categories (choose exactly one, or NONE):
%s

Post title: %s

Post excerpt:
%s

Respond with ONLY a JSON object: {"category": "<one category name exactly as \
listed above, or NONE>"}."""


def choose_category(
    title: str, body: str, categories: list[dict], model: str, num_ctx: int
) -> dict | None:
    """Ask the LLM to pick the best-fitting existing category for a post."""
    if ollama is None or not categories:
        return None
    lines = []
    for c in categories:
        desc = f" — {c['description']}" if c.get("description") else ""
        lines.append(f"- {c['name']}{desc}")
    prompt = CATEGORY_INSTRUCTIONS % ("\n".join(lines), title, body[:800])
    resp = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": CATEGORY_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        format="json",
        options={"temperature": 0.0, "num_ctx": num_ctx},
    )
    try:
        choice = (json.loads(resp["message"]["content"]).get("category") or "").strip()
    except Exception:
        return None
    for c in categories:
        if c["name"].lower() == choice.lower():
            return c
    return None


def assign_categories(
    posts: list[Post], categories: list[dict], model: str, num_ctx: int
) -> None:
    # "Uncategorized" is the fallback, not a real choice — don't offer it.
    choosable = [c for c in categories if c.get("slug") != "uncategorized"]
    for p in posts:
        cat = choose_category(p.title, p.body, choosable, model, num_ctx)
        if cat:
            p.category_id = cat["id"]
            p.category_name = cat["name"]


# --------------------------------------------------------------------------- #
# Optional space repair (for PDFs whose display fonts encode no space glyphs)
# --------------------------------------------------------------------------- #

# A "word" of this many characters with no space is almost certainly run-together.
_RUNON_RE = re.compile(r"\S{25,}")

SPACING_SYSTEM = (
    "You restore missing spaces in text extracted from a PDF. The words are "
    "correct but spaces between them were lost. Insert spaces so it reads "
    "normally. Do NOT add, remove, reorder, correct, or translate any words, "
    "punctuation, or line breaks — only insert spaces. Return the repaired text."
)


def needs_spacing_repair(text: str) -> bool:
    return bool(_RUNON_RE.search(text))


def _repair_chunk(text: str, model: str, num_ctx: int) -> str:
    resp = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": SPACING_SYSTEM},
            {"role": "user", "content": text},
        ],
        options={"temperature": 0.0, "num_ctx": num_ctx},
    )
    fixed = resp["message"]["content"].strip()
    # Guard against the model rewriting content: the repaired text must have
    # the exact same letters/digits as the original — spaces are the only
    # allowed change. If not, discard the repair and keep the original.
    return fixed if _alnum(fixed) == _alnum(text) else text


def repair_spacing(text: str, model: str, num_ctx: int) -> str:
    """Reinsert only spaces into run-together text, line by line.

    Repairing per line limits the blast radius: if the model corrupts one
    line (fails the letters-must-match guard), only that line falls back to
    the original instead of the whole post.
    """
    if ollama is None or not needs_spacing_repair(text):
        return text
    out_lines: list[str] = []
    for line in text.split("\n"):
        out_lines.append(_repair_chunk(line, model, num_ctx) if needs_spacing_repair(line) else line)
    return "\n".join(out_lines)


def _alnum(s: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]", "", s).lower()


# --------------------------------------------------------------------------- #
# Featured image selection
# --------------------------------------------------------------------------- #

def to_web_image(img: ExtractedImage, quality: int, max_width: int) -> tuple[bytes, str]:
    """Re-encode an extracted image as a web-friendly JPEG.

    Flattens transparency onto white and optionally downscales to max_width.
    Returns (jpeg_bytes, "jpeg"). Falls back to the original bytes/ext if
    Pillow is unavailable or the image can't be decoded.
    """
    try:
        from PIL import Image
    except Exception:
        return img.data, img.ext
    try:
        im = Image.open(io.BytesIO(img.data))
        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
            im = im.convert("RGBA")
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.split()[-1])
            im = bg
        else:
            im = im.convert("RGB")
        if max_width and im.width > max_width:
            new_h = round(im.height * max_width / im.width)
            im = im.resize((max_width, new_h), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=quality, optimize=True)
        return buf.getvalue(), "jpeg"
    except Exception:
        return img.data, img.ext


def choose_featured(
    post: Post, images: list[ExtractedImage], min_side: int
) -> ExtractedImage | None:
    candidates = [
        img
        for img in images
        if post.page_start <= img.page <= post.page_end
        and img.width >= min_side
        and img.height >= min_side
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda i: i.area)


# --------------------------------------------------------------------------- #
# Body -> HTML
# --------------------------------------------------------------------------- #

def text_to_html(body: str) -> str:
    # Drop page markers from the body.
    body = re.sub(r"=====\s*PAGE\s*\d+\s*=====", "", body)
    # Split into paragraphs on blank lines.
    chunks = re.split(r"\n\s*\n", body)
    out: list[str] = []
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        # collapse internal single newlines into spaces
        para = re.sub(r"\s*\n\s*", " ", chunk)
        para = html.escape(para)
        out.append(f"<p>{para}</p>")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# WordPress client
# --------------------------------------------------------------------------- #

class WordPress:
    def __init__(self, base_url: str, user: str, app_password: str):
        self.base = base_url.rstrip("/") + "/wp-json/wp/v2"
        self.session = requests.Session()
        self.session.auth = (user, app_password)
        self.session.headers.update({"User-Agent": "pdf2post/1.0"})

    def check(self) -> str:
        r = self.session.get(f"{self.base}/users/me", timeout=30)
        r.raise_for_status()
        me = r.json()
        return f"{me.get('name')} (id={me.get('id')}, slug={me.get('slug')})"

    def resolve_author(self, who: str | None) -> int | None:
        if not who:
            return None
        if str(who).isdigit():
            return int(who)
        # try slug, then search
        r = self.session.get(f"{self.base}/users", params={"slug": who}, timeout=30)
        if r.ok and r.json():
            return r.json()[0]["id"]
        r = self.session.get(f"{self.base}/users", params={"search": who}, timeout=30)
        if r.ok and r.json():
            return r.json()[0]["id"]
        raise ValueError(f"could not resolve author '{who}' to a user id")

    def upload_media(self, data: bytes, ext: str, filename: str) -> dict:
        ctype = mimetypes.types_map.get(f".{ext}", f"image/{ext}")
        headers = {
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Type": ctype,
        }
        r = self.session.post(
            f"{self.base}/media", headers=headers, data=data, timeout=120
        )
        r.raise_for_status()
        return r.json()

    def create_post(
        self,
        title: str,
        content: str,
        status: str,
        author: int | None,
        featured_media: int | None,
        categories: list[int] | None,
    ) -> dict:
        payload: dict = {"title": title, "content": content, "status": status}
        if author:
            payload["author"] = author
        if featured_media:
            payload["featured_media"] = featured_media
        if categories:
            payload["categories"] = categories
        r = self.session.post(f"{self.base}/posts", json=payload, timeout=120)
        r.raise_for_status()
        return r.json()

    def resolve_category(self, name: str) -> int:
        r = self.session.get(f"{self.base}/categories", params={"search": name}, timeout=30)
        if r.ok:
            for c in r.json():
                if c.get("name", "").lower() == name.lower():
                    return c["id"]
        r = self.session.post(f"{self.base}/categories", json={"name": name}, timeout=30)
        r.raise_for_status()
        return r.json()["id"]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def slugify(s: str, maxlen: int = 40) -> str:
    s = re.sub(r"[^\w\s-]", "", s.lower())
    s = re.sub(r"[\s_-]+", "-", s).strip("-")
    return (s or "post")[:maxlen]


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    load_dotenv()
    ap = argparse.ArgumentParser(description="Break a PDF into posts and publish to WordPress.")
    ap.add_argument("pdf", type=Path, help="path to the local PDF")
    ap.add_argument("--status", default=os.getenv("WP_STATUS", "publish"),
                    choices=["publish", "draft", "pending", "private"],
                    help="post status (default: publish)")
    ap.add_argument("--author", default=os.getenv("WP_AUTHOR"),
                    help="author username/slug or numeric user id (overrides .env WP_AUTHOR)")
    ap.add_argument("--model", default=os.getenv("OLLAMA_MODEL", "gemma4:latest"),
                    help="Ollama model for segmentation")
    ap.add_argument("--category", default=os.getenv("WP_CATEGORY"),
                    help="force a single category name on every post (created if missing)")
    ap.add_argument("--auto-category", action="store_true",
                    default=os.getenv("WP_AUTO_CATEGORY", "").lower() in ("1", "true", "yes"),
                    help="let the LLM assign each post to the best-fitting existing category")
    ap.add_argument("--min-image-side", type=int, default=200,
                    help="ignore embedded images smaller than this (px) for featured art")
    ap.add_argument("--jpeg-quality", type=int, default=85,
                    help="JPEG quality for featured images (default 85)")
    ap.add_argument("--max-image-width", type=int, default=2000,
                    help="downscale featured images wider than this (px); 0 disables")
    ap.add_argument("--max-llm-chars", type=int, default=120000,
                    help="cap on characters sent to the LLM")
    ap.add_argument("--num-ctx", type=int, default=int(os.getenv("OLLAMA_NUM_CTX", "32768")),
                    help="Ollama context window in tokens (must fit the whole doc)")
    ap.add_argument("--fix-spacing", action="store_true",
                    help="use the LLM to restore missing spaces in run-together text "
                         "(some PDFs encode display text with no space glyphs)")
    ap.add_argument("--ocr", action=argparse.BooleanOptionalAction, default=True,
                    help="OCR pages whose text layer is near-empty (vectorized/outlined "
                         "fonts) with a vision model; --no-ocr to disable")
    ap.add_argument("--ocr-dpi", type=int, default=200,
                    help="render DPI for OCR pages (default 200)")
    ap.add_argument("--ocr-threshold", type=int, default=60,
                    help="OCR a page whose extractable text is shorter than this (chars)")
    ap.add_argument("--ocr-draw-threshold", type=int, default=150,
                    help="also OCR a sparse-text page with at least this many vector "
                         "drawings (outlined/vectorized text)")
    ap.add_argument("--limit", type=int, default=0, help="only process the first N posts")
    ap.add_argument("--dry-run", action="store_true",
                    help="segment + extract only; write previews locally, do not post")
    ap.add_argument("--out", type=Path, default=None, help="output dir for --dry-run previews")
    args = ap.parse_args()

    if not args.pdf.exists():
        print(f"error: {args.pdf} not found", file=sys.stderr)
        return 2

    print(f"→ extracting {args.pdf} …")
    ocr_cfg = None
    if args.ocr:
        ocr_cfg = OcrConfig(model=args.model, dpi=args.ocr_dpi,
                            threshold=args.ocr_threshold,
                            draw_threshold=args.ocr_draw_threshold, num_ctx=args.num_ctx)
    pages, blob, images = extract_pdf(args.pdf, ocr_cfg)
    ocr_pages = [p.number for p in pages if p.ocr]
    print(f"  {len(pages)} pages, {len(blob):,} chars of text, {len(images)} embedded images"
          + (f"  (OCR'd pages: {ocr_pages})" if ocr_pages else ""))
    if not blob.strip():
        print("error: no extractable text (is this a scanned/image-only PDF?)", file=sys.stderr)
        return 3

    print(f"→ segmenting with {args.model} …")
    seg = segment_with_llm(blob, args.model, args.max_llm_chars, args.num_ctx)
    posts = build_posts(blob, pages, seg)
    print(f"  LLM found {len(seg)} boundaries → {len(posts)} posts")

    if args.fix_spacing:
        n = sum(1 for p in posts if needs_spacing_repair(p.title) or needs_spacing_repair(p.body))
        if n:
            print(f"→ repairing spacing on {n} post(s) with {args.model} …")
        for p in posts:
            p.title = repair_spacing(p.title, args.model, args.num_ctx)
            p.body = repair_spacing(p.body, args.model, args.num_ctx)

    for p in posts:
        p.featured = choose_featured(p, images, args.min_image_side)

    if args.limit and args.limit > 0:
        posts = posts[: args.limit]

    if args.auto_category:
        base_url = os.getenv("WP_URL", "https://electricant.space")
        print(f"→ auto-categorizing against {base_url} …")
        try:
            cats = fetch_categories(base_url)
            print(f"  categories: {', '.join(c['name'] for c in cats)}")
            assign_categories(posts, cats, args.model, args.num_ctx)
        except Exception as e:
            print(f"  ! could not fetch/assign categories: {e}", file=sys.stderr)

    # -------- dry run --------
    if args.dry_run:
        out = args.out or (args.pdf.parent / f"{args.pdf.stem}_pdf2post")
        out.mkdir(parents=True, exist_ok=True)
        for stale in list(out.glob("*.html")) + list(out.glob("*.jpeg")) + \
                list(out.glob("*.png")) + list(out.glob("*.jpg")):
            stale.unlink()
        for i, p in enumerate(posts, 1):
            base = f"{i:02d}-{slugify(p.title)}"
            html_doc = f"<h1>{html.escape(p.title)}</h1>\n"
            fe = "none"
            if p.featured:
                data, ext = to_web_image(p.featured, args.jpeg_quality, args.max_image_width)
                imgname = f"{base}.{ext}"
                (out / imgname).write_bytes(data)
                html_doc += f'<p><img src="{imgname}" style="max-width:100%"></p>\n'
                fe = (f"{p.featured.width}x{p.featured.height} "
                      f"{len(p.featured.data)//1024}KB→{len(data)//1024}KB {ext}")
            html_doc += text_to_html(p.body)
            (out / f"{base}.html").write_text(html_doc, encoding="utf-8")
            cat = f"  cat:{p.category_name}" if p.category_name else ("  cat:—" if args.auto_category else "")
            print(f"  [{i:02d}] {p.title!r}  pages {p.page_start}-{p.page_end}  "
                  f"{len(p.body):,} chars  art:{fe}{cat}")
        print(f"\n✓ dry run written to {out}")
        return 0

    # -------- publish --------
    base_url = os.getenv("WP_URL", "https://electricant.space")
    user = os.getenv("WP_USER")
    app_pw = os.getenv("WP_APP_PASSWORD")
    if not user or not app_pw:
        print("error: WP_USER and WP_APP_PASSWORD must be set in .env", file=sys.stderr)
        return 4

    wp = WordPress(base_url, user, app_pw)
    print(f"→ connecting to {base_url} …")
    try:
        print(f"  authenticated as {wp.check()}")
    except Exception as e:
        print(f"error: WordPress auth/connection failed: {e}", file=sys.stderr)
        return 5

    author_id = wp.resolve_author(args.author) if args.author else None
    if args.author:
        print(f"  author '{args.author}' → user id {author_id}")

    cat_ids = None
    if args.category:
        cat_ids = [wp.resolve_category(args.category)]
        print(f"  category '{args.category}' → {cat_ids}")

    print(f"→ publishing {len(posts)} post(s) as status='{args.status}' …")
    for i, p in enumerate(posts, 1):
        media_id = None
        if p.featured:
            data, ext = to_web_image(p.featured, args.jpeg_quality, args.max_image_width)
            fn = f"{slugify(p.title)}.{ext}"
            try:
                m = wp.upload_media(data, ext, fn)
                media_id = m["id"]
                p.media_id = media_id
            except Exception as e:
                print(f"  [{i:02d}] media upload failed ({e}); posting without art", file=sys.stderr)
        # per-post categories: forced --category (all posts) + auto-assigned one
        post_cats = list(cat_ids or [])
        if p.category_id and p.category_id not in post_cats:
            post_cats.append(p.category_id)
        content = text_to_html(p.body)
        try:
            res = wp.create_post(p.title, content, args.status, author_id, media_id,
                                 post_cats or None)
            p.wp_id = res.get("id")
            p.wp_link = res.get("link")
            catnote = f", cat {p.category_name}" if p.category_name else ""
            print(f"  [{i:02d}] ✓ {p.title!r} → {p.wp_link}  (id {p.wp_id}"
                  + (f", media {media_id}" if media_id else "") + catnote + ")")
        except Exception as e:
            print(f"  [{i:02d}] ✗ FAILED to post {p.title!r}: {e}", file=sys.stderr)

    ok = sum(1 for p in posts if p.wp_id)
    print(f"\n✓ done — {ok}/{len(posts)} posted.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
