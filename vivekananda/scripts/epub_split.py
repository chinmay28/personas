"""Reusable epub-by-TOC splitter.

Splits an epub into sections, one per toc.ncx navPoint, by walking the spine in
document order and assigning every block-level element to the most recent TOC
anchor seen. Sections may span spine files. Works on calibre-generated epubs
(anchors are <a id="..."> inside <p>) and generic EPUB2 with an NCX.

Usage as a library:
    from epub_split import split_epub
    entries, dropped = split_epub("book.epub")
    for e in entries: print(e.order, e.depth, e.path, len(e.paragraphs))

Usage as a CLI (dump a quick report):
    python epub_split.py book.epub
"""
from __future__ import annotations

import html
import posixpath
import re
import sys
import zipfile
from dataclasses import dataclass, field
from urllib.parse import unquote
import xml.etree.ElementTree as ET

import warnings
from bs4 import BeautifulSoup, NavigableString, Tag, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

NS = {
    "c": "urn:oasis:names:tc:opendocument:xmlns:container",
    "o": "http://www.idpf.org/2007/opf",
    "n": "http://www.daisy.org/z3986/2005/ncx/",
}
BLOCK_TAGS = {"p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "blockquote", "pre", "dd", "dt"}


@dataclass
class TocEntry:
    order: int            # position in TOC (playOrder-like, 1-based)
    depth: int            # 0 = top level
    path: list[str]       # labels from root to this entry
    src: str              # spine file (zip path)
    anchor: str | None    # fragment id, None = file start
    paragraphs: list[str] = field(default_factory=list)
    heading_blocks: list[str] = field(default_factory=list)  # blocks dropped as title repeats
    images: int = 0
    first_file: str | None = None
    doc_seq: int | None = None   # order in which the entry was first reached in document order

    @property
    def label(self) -> str:
        return self.path[-1]


def _norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return s.strip()


def _block_text(tag: Tag) -> tuple[list[str], int]:
    """Paragraph texts of a block: tags stripped, single <br> kept as newline, double <br> splits paragraphs."""
    parts: list[str] = []
    images = 0

    def walk(node):
        nonlocal images
        for ch in node.children:
            if isinstance(ch, NavigableString):
                parts.append(str(ch))
            elif isinstance(ch, Tag):
                if ch.name == "br":
                    parts.append("\n")
                elif ch.name == "img":
                    images += 1
                    parts.append(" ")
                else:
                    walk(ch)

    walk(tag)
    raw = html.unescape("".join(parts))
    lines = [re.sub(r"[ \t\r\f\v\u00a0]+", " ", ln).strip() for ln in raw.split("\n")]
    # a blank line (two consecutive <br>) separates paragraphs; a single <br> is a line break
    paras: list[str] = []
    cur: list[str] = []
    for ln in lines:
        if ln:
            cur.append(ln)
        elif cur:
            paras.append("\n".join(cur)); cur = []
    if cur:
        paras.append("\n".join(cur))
    return paras, images


def _read_zip_text(z: zipfile.ZipFile, name: str) -> str:
    return z.read(name).decode("utf-8", errors="replace")


def load_toc_and_spine(z: zipfile.ZipFile) -> tuple[list[str], list[TocEntry]]:
    container = ET.fromstring(z.read("META-INF/container.xml"))
    opf_path = container.find(".//c:rootfile", NS).get("full-path")
    opf_dir = posixpath.dirname(opf_path)
    opf = ET.fromstring(z.read(opf_path))
    manifest = {i.get("id"): i.get("href") for i in opf.find("o:manifest", NS)}
    spine_el = opf.find("o:spine", NS)
    spine = [posixpath.normpath(posixpath.join(opf_dir, manifest[r.get("idref")])) for r in spine_el]
    toc_id = spine_el.get("toc")
    ncx_href = manifest.get(toc_id) if toc_id else next(h for h in manifest.values() if h.endswith(".ncx"))
    ncx = ET.fromstring(z.read(posixpath.normpath(posixpath.join(opf_dir, ncx_href))))

    entries: list[TocEntry] = []

    def walk(np, depth, path):
        for p in np.findall("n:navPoint", NS):
            label = (p.find("n:navLabel/n:text", NS).text or "").strip()
            src = p.find("n:content", NS).get("src")
            file_part, _, frag = src.partition("#")
            file_part = posixpath.normpath(posixpath.join(opf_dir, unquote(file_part)))
            e = TocEntry(order=len(entries) + 1, depth=depth, path=path + [label], src=file_part, anchor=frag or None)
            entries.append(e)
            walk(p, depth + 1, path + [label])

    walk(ncx.find("n:navMap", NS), 0, [])
    return spine, entries


def split_epub(epub_path: str, drop_title_blocks: bool = True) -> tuple[list[TocEntry], dict]:
    """Return TOC entries with their paragraphs, plus a dict describing dropped content."""
    with zipfile.ZipFile(epub_path) as z:
        spine, entries = load_toc_and_spine(z)
        by_anchor: dict[tuple[str, str | None], TocEntry] = {}
        for e in entries:
            by_anchor[(e.src, e.anchor)] = e
        dropped = {"pre_toc_chars": 0, "pre_toc_blocks": 0, "unresolved_anchors": []}
        seen_anchor_keys: set = set()
        current: TocEntry | None = None
        seq = 0

        def activate(e: TocEntry, fname: str):
            nonlocal current, seq
            current = e
            if e.doc_seq is None:
                seq += 1
                e.doc_seq = seq
                e.first_file = fname

        for fname in spine:
            if fname not in z.namelist():
                continue
            soup = BeautifulSoup(_read_zip_text(z, fname), "lxml")
            body = soup.body or soup
            if (fname, None) in by_anchor:
                activate(by_anchor[(fname, None)], fname)
                seen_anchor_keys.add((fname, None))
            for blk in body.find_all(list(BLOCK_TAGS)):
                if blk.find(list(BLOCK_TAGS)):
                    continue  # container of other blocks; children will be visited
                ids = [a.get("id") for a in blk.find_all(True) if a.get("id")]
                if blk.get("id"):
                    ids.insert(0, blk.get("id"))
                hit = None
                for i in ids:
                    key = (fname, i)
                    if key in by_anchor:
                        hit = by_anchor[key]
                        seen_anchor_keys.add(key)
                if hit is not None:
                    activate(hit, fname)
                paras, images = _block_text(blk)
                if current is None:
                    dropped["pre_toc_chars"] += sum(len(p) for p in paras)
                    dropped["pre_toc_blocks"] += 1
                    continue
                current.images += images
                for k, text in enumerate(paras):
                    is_anchor_para = hit is not None and k == 0
                    if drop_title_blocks and not current.paragraphs and (is_anchor_para or not current.heading_blocks):
                        nt, nl = _norm(text), _norm(current.label)
                        if nt == nl or (is_anchor_para and len(text) <= len(current.label) + 10 and nl in nt):
                            current.heading_blocks.append(text)
                            continue
                    current.paragraphs.append(text)
        dropped["unresolved_anchors"] = [f"{e.src}#{e.anchor}" for e in entries if (e.src, e.anchor) not in seen_anchor_keys]
    return entries, dropped


if __name__ == "__main__":
    ents, dropped = split_epub(sys.argv[1])
    total = sum(len(p) for e in ents for p in e.paragraphs)
    print(f"entries={len(ents)} chars={total} dropped={dropped}")
    for e in ents[:40]:
        print(e.order, e.depth, " / ".join(e.path), len(e.paragraphs), sum(len(p) for p in e.paragraphs))
